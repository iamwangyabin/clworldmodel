#!/usr/bin/env python3
"""Compare an active run's early scalars with a preserved ARROW reference."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Iterable


def _points_by_step(points: Iterable[dict[str, Any]]) -> dict[int, float]:
    values: dict[int, float] = {}
    for point in points:
        values[int(point["step"])] = float(point["value"])
    return values


def compare_progress(
    current_metrics: dict[str, list[dict[str, Any]]],
    reference: dict[str, Any],
    *,
    through_step: int | None = None,
) -> dict[str, Any]:
    """Return a finite, JSON-serializable diagnostic and guard decision."""
    guard = reference["guard"]
    start_step = int(guard["start_step"])
    configured_end = int(guard["end_step"])
    requested_end = configured_end if through_step is None else through_step
    end_step = min(configured_end, requested_end)
    required_points = int(guard["required_aligned_points"])
    rules = guard["metric_rules"]
    observed_steps = [
        int(point["step"])
        for points in current_metrics.values()
        for point in points
    ]
    observed_through_step = max(observed_steps, default=-1)
    metric_results: dict[str, Any] = {}
    failures: list[str] = []
    insufficient: list[str] = []

    for tag, reference_points in reference["metrics"].items():
        reference_by_step = _points_by_step(reference_points)
        current_by_step = _points_by_step(current_metrics.get(tag, []))
        aligned_steps = sorted(
            step
            for step in reference_by_step.keys() & current_by_step.keys()
            if start_step <= step <= end_step
        )
        non_finite_steps = [
            step
            for step in aligned_steps
            if not math.isfinite(current_by_step[step])
        ]
        finite_steps = [
            step
            for step in aligned_steps
            if math.isfinite(current_by_step[step])
            and math.isfinite(reference_by_step[step])
        ]
        rule = rules.get(tag)
        floor = float(rule.get("reference_floor", 0.0)) if rule else 0.0
        ratios = [
            abs(current_by_step[step])
            / max(abs(reference_by_step[step]), floor)
            for step in finite_steps
        ]
        median_ratio = statistics.median(ratios) if ratios else None
        latest_step = finite_steps[-1] if finite_steps else None
        result: dict[str, Any] = {
            "status": "informational" if rule is None else "pending",
            "aligned_steps": finite_steps,
            "non_finite_steps": non_finite_steps,
            "median_absolute_ratio": median_ratio,
            "latest": (
                {
                    "step": latest_step,
                    "current": current_by_step[latest_step],
                    "reference": reference_by_step[latest_step],
                    "absolute_ratio": ratios[-1],
                }
                if latest_step is not None
                else None
            ),
        }
        if rule is not None:
            result["rule"] = rule
            if non_finite_steps:
                result["status"] = "fail"
                failures.append(f"{tag}: non-finite values at {non_finite_steps}")
            elif len(ratios) < required_points:
                missing = f"{tag}: only {len(ratios)}/{required_points} aligned points"
                if observed_through_step >= end_step:
                    result["status"] = "fail"
                    failures.append(missing)
                else:
                    result["status"] = "insufficient_data"
                    insufficient.append(missing)
            else:
                minimum = rule.get("median_ratio_min")
                maximum = rule.get("median_ratio_max")
                below = minimum is not None and median_ratio < float(minimum)
                above = maximum is not None and median_ratio > float(maximum)
                if below or above:
                    result["status"] = "fail"
                    failures.append(
                        f"{tag}: median absolute ratio {median_ratio:.6g} "
                        f"outside [{minimum}, {maximum}]"
                    )
                else:
                    result["status"] = "pass"
        metric_results[tag] = result

    if failures:
        status = "fail"
    elif insufficient:
        status = "insufficient_data"
    else:
        status = "pass"
    return {
        "schema_version": 1,
        "status": status,
        "observed_through_step": observed_through_step,
        "comparison_window": {
            "start_step": start_step,
            "end_step": end_step,
            "required_aligned_points": required_points,
        },
        "failures": failures,
        "insufficient_data": insufficient,
        "metrics": metric_results,
        "reference_provenance": reference["provenance"],
        "comparison_class": reference["comparison_class"],
        "interpretation": reference["interpretation"],
    }


def _event_files(run_dir: Path, explicit: list[Path]) -> list[Path]:
    paths = [path.resolve() for path in explicit]
    if run_dir:
        paths.extend(sorted(run_dir.resolve().glob("events.out.tfevents.*")))
    unique = list(dict.fromkeys(paths))
    if not unique:
        raise FileNotFoundError("No TensorBoard event files found")
    return unique


def _load_tensorboard_scalars(
    event_files: list[Path], tags: Iterable[str]
) -> dict[str, list[dict[str, float | int]]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorBoard is required to read event files in the training runtime"
        ) from exc

    merged: dict[str, dict[int, tuple[float, float]]] = {tag: {} for tag in tags}
    for event_file in event_files:
        accumulator = EventAccumulator(
            str(event_file), size_guidance={"scalars": 0}
        )
        accumulator.Reload()
        available = set(accumulator.Tags().get("scalars", []))
        for tag in merged.keys() & available:
            for event in accumulator.Scalars(tag):
                previous = merged[tag].get(int(event.step))
                candidate = (float(event.wall_time), float(event.value))
                if previous is None or candidate[0] >= previous[0]:
                    merged[tag][int(event.step)] = candidate
    return {
        tag: [
            {"step": step, "value": wall_and_value[1]}
            for step, wall_and_value in sorted(points.items())
        ]
        for tag, points in merged.items()
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--event-file", type=Path, action="append", default=[])
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--through-step", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--enforce-guard",
        action="store_true",
        help="Exit 2 when the predeclared diagnostic guard fails.",
    )
    args = parser.parse_args()
    if args.run_dir is None and not args.event_file:
        parser.error("one of --run-dir or --event-file is required")
    if args.through_step is not None and args.through_step < 0:
        parser.error("--through-step must be non-negative")

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    event_files = _event_files(args.run_dir, args.event_file)
    current = _load_tensorboard_scalars(event_files, reference["metrics"])
    result = compare_progress(
        current, reference, through_step=args.through_step
    )
    result["current_event_files"] = [str(path) for path in event_files]
    if args.output is not None:
        _write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 2 if args.enforce_guard and result["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
