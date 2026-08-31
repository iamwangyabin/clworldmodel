#!/usr/bin/env python3
"""Select the preregistered Task-0 profile without reading final evaluation."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from run_evolving_task0_sweep import (
    BASELINE_HPARAMETERS,
    DURATION_PROFILE_EPOCHS,
    DURATION_PROTOCOL,
    PROFILE_OVERRIDES,
    PROTOCOL,
    TASK_ORDER,
)


LR_EXPECTED_PROFILES = frozenset(("fixed_v1", *PROFILE_OVERRIDES))
DURATION_EXPECTED_PROFILES = frozenset(
    ("fixed_v1", *DURATION_PROFILE_EPOCHS)
)
ALL_PROFILES = LR_EXPECTED_PROFILES | DURATION_EXPECTED_PROFILES


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        action="append",
        required=True,
        help="Repeat once for the fixed_v1 control and each of four sweep runs.",
    )
    parser.add_argument(
        "--family",
        choices=("learning_rate", "duration"),
        default="learning_rate",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _profile_distance(config: dict[str, Any]) -> float:
    return float(
        sum(
            abs(math.log(float(config.get(name, baseline)) / baseline))
            for name, baseline in BASELINE_HPARAMETERS.items()
        )
    )


def _candidate_from_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    launch = _load_json(run_dir / "launch.json")
    config = _load_json(run_dir / "resolved_training_config.json")
    profile = str(config.get("evolving_task0_profile", "fixed_v1"))
    if profile not in ALL_PROFILES:
        raise ValueError(f"Run has an undeclared Task-0 profile: {profile!r}")
    expected_hparams = dict(BASELINE_HPARAMETERS)
    expected_hparams.update(PROFILE_OVERRIDES.get(profile, {}))
    mismatches = {
        name: (config.get(name, BASELINE_HPARAMETERS[name]), expected)
        for name, expected in expected_hparams.items()
        if config.get(name, BASELINE_HPARAMETERS[name]) != expected
    }
    if mismatches:
        raise ValueError(f"Run {run_dir} drifted from profile {profile}: {mismatches}")
    if tuple(launch.get("task_order", ())) != TASK_ORDER:
        raise ValueError(f"Run {run_dir} does not use the fixed task order")
    if launch.get("seed_index") != 0:
        raise ValueError(f"Run {run_dir} is not the preregistered seed-0 pilot")
    expected_task0_epochs = DURATION_PROFILE_EPOCHS.get(profile, 90)
    schedule = config.get("esc", {}).get("kwargs", {})
    task_durations = schedule.get("task_durations")
    actual_task0_epochs = (
        task_durations[0]
        if isinstance(task_durations, list) and task_durations
        else schedule.get("swap_sched")
    )
    if actual_task0_epochs != expected_task0_epochs:
        raise ValueError(
            f"Run {run_dir} has Task-0 duration {actual_task0_epochs}, "
            f"expected {expected_task0_epochs}"
        )
    if profile != "fixed_v1":
        status = _load_json(run_dir / "run_status.json")
        if status.get("complete") is not True:
            raise ValueError(f"Sweep run is not complete and eligible: {run_dir}")
        if int(config.get("epochs", -1)) != expected_task0_epochs:
            raise ValueError(f"Sweep run did not stop at the Task-0 boundary: {run_dir}")

    pre_path = (
        run_dir
        / "evolving_core_consolidation"
        / "task_00_pre_validation.json"
    )
    if pre_path.is_file():
        pre = _load_json(pre_path)
        validation = pre.get("validation")
        if not isinstance(validation, dict):
            raise ValueError(f"Pre-validation payload is malformed: {pre_path}")
        raw_means = validation.get("raw_mean")
        task_seeds = validation.get("task_seeds")
        rollouts = pre.get("rollouts_per_task")
        if pre.get("heldout_final_data_used") is not False:
            raise ValueError(f"Pre-validation did not exclude held-out data: {pre_path}")
    else:
        # Compatibility for the already-running fixed_v1 control, which was
        # launched before the standalone pre-validation artifact was added.
        boundary_path = (
            run_dir
            / "evolving_core_consolidation"
            / "task_00_boundary.json"
        )
        boundary = _load_json(boundary_path)
        validation = boundary.get("validation")
        if not isinstance(validation, dict):
            raise ValueError(f"Boundary payload is malformed: {boundary_path}")
        raw_means = validation.get("pre_raw_mean")
        task_seeds = validation.get("task_seeds")
        rollouts = 16
        pre_path = boundary_path
    if not isinstance(raw_means, list) or len(raw_means) < 1:
        raise ValueError(f"Task-0 pre-consolidation return is missing: {pre_path}")
    score = float(raw_means[0])
    if not math.isfinite(score):
        raise ValueError(f"Task-0 score is non-finite: {pre_path}")
    if not isinstance(task_seeds, list) or len(task_seeds) < 1:
        raise ValueError(f"Fixed validation seed is missing: {pre_path}")
    return {
        "profile": profile,
        "run_dir": str(run_dir),
        "project_git_commit": launch.get("project_git", {}).get("commit"),
        "score": score,
        "score_name": "task0_pre_consolidation_raw_return_mean",
        "task0_acquisition_epochs": expected_task0_epochs,
        "validation_task_seeds": task_seeds[:1],
        "rollouts_per_task": rollouts,
        "profile_log_distance_from_fixed_v1": _profile_distance(config),
        "source_artifact": str(pre_path),
    }


def _select(
    candidate_dirs: list[Path], *, family: str = "learning_rate"
) -> dict[str, Any]:
    if family == "learning_rate":
        expected_profiles = LR_EXPECTED_PROFILES
        protocol = PROTOCOL
    elif family == "duration":
        expected_profiles = DURATION_EXPECTED_PROFILES
        protocol = DURATION_PROTOCOL
    else:
        raise ValueError(f"Unknown Task-0 selection family: {family!r}")
    candidates = [_candidate_from_run(path) for path in candidate_dirs]
    by_profile = {candidate["profile"]: candidate for candidate in candidates}
    if len(by_profile) != len(candidates):
        raise ValueError("Task-0 selection received duplicate profiles")
    observed_profiles = frozenset(by_profile)
    if observed_profiles != expected_profiles:
        raise ValueError(
            "Task-0 selection requires the complete preregistered profile set: "
            f"missing={sorted(expected_profiles - observed_profiles)}, "
            f"unexpected={sorted(observed_profiles - expected_profiles)}"
        )
    cohorts = {
        tuple(candidate["validation_task_seeds"]) for candidate in candidates
    }
    if len(cohorts) != 1:
        raise ValueError("Task-0 candidates did not use the same validation cohort")
    score_ranking = sorted(
        candidates,
        key=lambda candidate: (
            -candidate["score"],
            candidate["profile_log_distance_from_fixed_v1"],
            candidate["profile"],
        ),
    )
    if family == "duration":
        maximum_score = score_ranking[0]["score"]
        tolerance = 0.05 * max(abs(maximum_score), 1.0)
        near_best_threshold = maximum_score - tolerance
        near_best = [
            candidate
            for candidate in candidates
            if candidate["score"] >= near_best_threshold
        ]
        winner = min(
            near_best,
            key=lambda candidate: (
                candidate["task0_acquisition_epochs"],
                -candidate["score"],
                candidate["profile"],
            ),
        )
        selection_rule = (
            "choose the shortest Task-0 duration whose raw mean is within five "
            "percent of the maximum observed mean; then higher mean and lexical name"
        )
    else:
        maximum_score = score_ranking[0]["score"]
        tolerance = None
        near_best_threshold = None
        winner = score_ranking[0]
        selection_rule = (
            "maximize raw mean; exact ties use smaller log-distance from fixed_v1, "
            "then lexical profile name"
        )
    return {
        "schema_version": 1,
        "artifact_kind": f"evolving_core_task0_{family}_selection",
        "protocol": protocol,
        "selection_family": family,
        "task_order": list(TASK_ORDER),
        "seed_index": 0,
        "selection_metric": "task0_pre_consolidation_raw_return_mean",
        "maximize": True,
        "validation_task_seeds": list(next(iter(cohorts))),
        "heldout_final_data_read": False,
        "selection_rule": selection_rule,
        "maximum_observed_score": maximum_score,
        "near_best_relative_tolerance": 0.05 if family == "duration" else None,
        "near_best_absolute_tolerance": tolerance,
        "near_best_score_threshold": near_best_threshold,
        "winner": winner,
        "score_ranking": score_ranking,
        "learning_curve": sorted(
            candidates,
            key=lambda candidate: (
                candidate["task0_acquisition_epochs"], candidate["profile"]
            ),
        ),
        "claim_limit": (
            "seed-0 Task-0 acquisition selection only; launch the winner from "
            "scratch and use fresh seeds for confirmation"
        ),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    args = _parser().parse_args()
    selection = _select(args.candidate_dir, family=args.family)
    output = args.output.expanduser().resolve()
    _write_json_atomic(output, selection)
    print(json.dumps(selection, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
