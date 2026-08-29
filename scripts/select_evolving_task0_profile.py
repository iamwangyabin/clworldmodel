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
    PROFILE_OVERRIDES,
    PROTOCOL,
    TASK_ORDER,
)


EXPECTED_PROFILES = frozenset(("fixed_v1", *PROFILE_OVERRIDES))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        action="append",
        required=True,
        help="Repeat once for the fixed_v1 control and each of four sweep runs.",
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
    if profile not in EXPECTED_PROFILES:
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
    if profile != "fixed_v1":
        status = _load_json(run_dir / "run_status.json")
        if status.get("complete") is not True:
            raise ValueError(f"Sweep run is not complete and eligible: {run_dir}")
        if int(config.get("epochs", -1)) != 90:
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
        "validation_task_seeds": task_seeds[:1],
        "rollouts_per_task": rollouts,
        "profile_log_distance_from_fixed_v1": _profile_distance(config),
        "source_artifact": str(pre_path),
    }


def _select(candidate_dirs: list[Path]) -> dict[str, Any]:
    candidates = [_candidate_from_run(path) for path in candidate_dirs]
    by_profile = {candidate["profile"]: candidate for candidate in candidates}
    if len(by_profile) != len(candidates):
        raise ValueError("Task-0 selection received duplicate profiles")
    observed_profiles = frozenset(by_profile)
    if observed_profiles != EXPECTED_PROFILES:
        raise ValueError(
            "Task-0 selection requires the complete preregistered profile set: "
            f"missing={sorted(EXPECTED_PROFILES - observed_profiles)}, "
            f"unexpected={sorted(observed_profiles - EXPECTED_PROFILES)}"
        )
    cohorts = {
        tuple(candidate["validation_task_seeds"]) for candidate in candidates
    }
    if len(cohorts) != 1:
        raise ValueError("Task-0 candidates did not use the same validation cohort")
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -candidate["score"],
            candidate["profile_log_distance_from_fixed_v1"],
            candidate["profile"],
        ),
    )
    return {
        "schema_version": 1,
        "artifact_kind": "evolving_core_task0_hparam_selection",
        "protocol": PROTOCOL,
        "task_order": list(TASK_ORDER),
        "seed_index": 0,
        "selection_metric": "task0_pre_consolidation_raw_return_mean",
        "maximize": True,
        "validation_task_seeds": list(next(iter(cohorts))),
        "heldout_final_data_read": False,
        "tie_break": (
            "smaller log-distance from fixed_v1, then lexical profile name"
        ),
        "winner": ranked[0],
        "ranking": ranked,
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
    selection = _select(args.candidate_dir)
    output = args.output.expanduser().resolve()
    _write_json_atomic(output, selection)
    print(json.dumps(selection, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
