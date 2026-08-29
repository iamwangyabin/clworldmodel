#!/usr/bin/env python3
"""Run one preregistered MsPacman-only Evolving-Core acquisition profile."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from git_provenance import git_state, require_synced_training_git_state
from run_arrow_ar50_atari import (
    ARROW_ROOT,
    ROOT,
    SEEDS,
    THREAD_ENV_KEYS,
    UPSTREAM_COMMIT,
    _arrow_replay_storage_budget,
    _config_path,
    _run_and_tee,
    _runtime_info,
    _verify_primary_config,
    _write_json,
)
from run_cnn_projector_lora_incremental import _prepare_replay_symlink
from run_evolving_atomic_rssm import (
    TASK_DURATION_EPOCHS,
    TASK_ORDERS,
    _resolved_config,
)


PROTOCOL = "Evolving-Core-Atomic-RSSM-ARROW-v1-Task0-HParamSweep-v1"
DURATION_PROTOCOL = (
    "Evolving-Core-Atomic-RSSM-ARROW-v1-Task0-DurationSweep-v1"
)
TASK_ORDER_NAME = "mspacman-boxing-crazyclimber"
TASK_ORDER = TASK_ORDERS[TASK_ORDER_NAME]
SELECTION_SEED_INDEX = 0
BASELINE_HPARAMETERS = {
    "first_task_shared_core_lr": 2e-4,
    "task_private_lr": 2e-4,
    "ac_lr": 1e-4,
}
PROFILE_OVERRIDES = {
    "task0_shared_lr_1e4": {"first_task_shared_core_lr": 1e-4},
    "task0_shared_lr_3e4": {"first_task_shared_core_lr": 3e-4},
    "task0_private_lr_3e4": {"task_private_lr": 3e-4},
    "task0_actor_lr_2e4": {"ac_lr": 2e-4},
}
DURATION_PROFILE_EPOCHS = {
    "task0_epochs_120": 120,
    "task0_epochs_150": 150,
    "task0_epochs_180": 180,
    "task0_epochs_240": 240,
}
ALL_PROFILES = (*PROFILE_OVERRIDES, *DURATION_PROFILE_EPOCHS)
PROFILE_RATIONALE = {
    "task0_shared_lr_1e4": "test slower shared representation learning",
    "task0_shared_lr_3e4": "test faster shared representation learning",
    "task0_private_lr_3e4": "test faster private projector/atom/head learning",
    "task0_actor_lr_2e4": "test faster policy/value learning",
    "task0_epochs_120": "test 33 percent more Task-0 acquisition",
    "task0_epochs_150": "test 67 percent more Task-0 acquisition",
    "task0_epochs_180": "test twice the Task-0 acquisition budget",
    "task0_epochs_240": "test 2.67 times the Task-0 acquisition budget",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=ALL_PROFILES, required=True)
    parser.add_argument(
        "--seed",
        type=int,
        choices=(SELECTION_SEED_INDEX,),
        default=SELECTION_SEED_INDEX,
        help="The sweep is a seed-0 selection pilot; confirmation seeds are separate.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--replay-mmap-root", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cpu-threads", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _resolved_sweep_config(source: dict, *, profile: str) -> dict:
    """Apply exactly one declared Task-0 change and stop at its boundary."""

    if profile not in ALL_PROFILES:
        raise ValueError(f"Unknown Task-0 sweep profile: {profile!r}")
    config = _resolved_config(source, task_order=TASK_ORDER_NAME)
    task0_epochs = DURATION_PROFILE_EPOCHS.get(profile, TASK_DURATION_EPOCHS)
    config["epochs"] = task0_epochs
    config["evolving_task0_profile"] = profile
    config.update(PROFILE_OVERRIDES.get(profile, {}))
    if profile in DURATION_PROFILE_EPOCHS:
        schedule = config["esc"]["kwargs"]
        schedule.pop("swap_sched")
        schedule["task_durations"] = [task0_epochs, 90, 90]
    return config


def _training_command(
    *,
    python: Path,
    config_path: Path,
    output_dir: Path,
    task_snapshot_dir: Path,
    project_commit: str,
) -> list[str]:
    # Deliberately omit --evaluate-final: held-out-final data may not influence
    # the Task-0 hyperparameter choice.
    return [
        str(python),
        "Code/ARROW_and_DV3/Atari/train.py",
        "--config",
        str(config_path),
        "--arrow-replay-ratio",
        "50-50",
        "--log-dir",
        str(output_dir),
        "--task-bank-snapshot-dir",
        str(task_snapshot_dir),
        "--project-git-commit",
        project_commit,
        "--fused-adam",
        "--tf32",
        "--profile-stages",
    ]


def _budget_manifest(config: dict) -> dict:
    decisions_per_epoch = int(config["n_sync"]) * int(config["gen_seq_len"])
    raw_frames_per_epoch = decisions_per_epoch * int(config["env_repeat"])
    online_updates = int(config["epochs"]) * int(config["steps_per_batch"])
    consolidation_updates = int(config["boundary_consolidation_steps"])
    return {
        "task_duration_epochs": int(config["epochs"]),
        "tasks_trained": 1,
        "raw_environment_frames": raw_frames_per_epoch * int(config["epochs"]),
        "online_world_model_updates": online_updates,
        "boundary_consolidation_world_model_updates": consolidation_updates,
        "total_world_model_optimizer_steps": online_updates
        + consolidation_updates,
        "actor_critic_updates": int(config["epochs"])
        * int(config["ac_train_steps"]),
        "online_current_sequences": online_updates * int(config["mb_n_size"]),
        "online_memory_sequences": 0,
        "consolidation_sequences": consolidation_updates
        * int(config["mb_n_size"]),
        "online_sequence_batch_total": int(config["mb_n_size"]),
        "replay": _arrow_replay_storage_budget(config),
        "evaluation_transitions_enter_replay": False,
        "heldout_final_evaluation_performed": False,
        "consolidation_is_extra_compute": True,
    }


def _profile_distance(config: dict) -> float:
    """Log-space distance used only as an exact-score tie breaker."""

    return float(
        sum(
            abs(math.log(float(config.get(name, baseline)) / baseline))
            for name, baseline in BASELINE_HPARAMETERS.items()
        )
    )


def _selection_candidate(
    *, output_dir: Path, launch: dict, config: dict
) -> dict:
    path = (
        output_dir
        / "evolving_core_consolidation"
        / "task_00_pre_validation.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact_kind") != "evolving_core_pre_consolidation_validation":
        raise ValueError(f"Unexpected pre-validation artifact: {path}")
    if payload.get("completed_task_id") != 0:
        raise ValueError("Task-0 selection artifact addresses a different task")
    validation = payload.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("Task-0 selection artifact is missing validation data")
    raw_means = validation.get("raw_mean")
    if not isinstance(raw_means, list) or len(raw_means) != 1:
        raise ValueError("Task-0 selection requires exactly one raw return mean")
    score = float(raw_means[0])
    if not math.isfinite(score):
        raise ValueError("Task-0 raw return mean must be finite")
    if payload.get("heldout_final_data_used") is not False:
        raise ValueError("Task-0 selection artifact must exclude held-out-final data")
    profile = config["evolving_task0_profile"]
    duration_sweep = profile in DURATION_PROFILE_EPOCHS
    return {
        "schema_version": 1,
        "artifact_kind": "evolving_core_task0_sweep_candidate",
        "protocol": launch["protocol"],
        "selection_family": "duration" if duration_sweep else "learning_rate",
        "profile": profile,
        "profile_overrides": PROFILE_OVERRIDES.get(profile, {}),
        "task0_acquisition_epochs": int(config["epochs"]),
        "profile_log_distance_from_fixed_v1": _profile_distance(config),
        "project_git_commit": launch["project_git"]["commit"],
        "seed_index": launch["seed_index"],
        "seed": launch["seed"],
        "task_order": launch["task_order"],
        "score_name": "task0_pre_consolidation_raw_return_mean",
        "score": score,
        "validation_task_seeds": validation.get("task_seeds"),
        "rollouts_per_task": payload.get("rollouts_per_task"),
        "source_artifact": str(path),
        "eligible": True,
        "heldout_final_data_used": False,
        "selection_rule": (
            "shortest duration within five percent of the best raw mean, then "
            "higher raw mean and lexical profile name"
            if duration_sweep
            else "maximum raw mean; exact ties use smaller log-distance from "
            "fixed_v1, then lexical profile name"
        ),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")
    project_git = (
        git_state(ROOT) if args.dry_run else require_synced_training_git_state(ROOT)
    )
    source_path = _config_path("original", args.seed)
    source = _verify_primary_config(source_path, "original", args.seed)
    config = _resolved_sweep_config(source, profile=args.profile)
    duration_sweep = args.profile in DURATION_PROFILE_EPOCHS
    protocol = DURATION_PROTOCOL if duration_sweep else PROTOCOL
    python = args.python.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else ROOT
        / "runs"
        / f"evolving_task0_{args.profile}_s{args.seed}_pilot"
    )
    config_path = output_dir / "resolved_training_config.json"
    task_snapshot_dir = output_dir / "task_boundary_snapshots"
    command = _training_command(
        python=python,
        config_path=config_path,
        output_dir=output_dir,
        task_snapshot_dir=task_snapshot_dir,
        project_commit=str(project_git["commit"]),
    )
    env = os.environ.copy()
    thread_env = {key: str(args.cpu_threads) for key in THREAD_ENV_KEYS}
    env.update(thread_env)
    project_pythonpath = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
    env["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (project_pythonpath, env.get("PYTHONPATH"))
        if value
    )
    launch = {
        "schema_version": 1,
        "method": "Evolving-Core Atomic RSSM",
        "protocol": protocol,
        "classification": "pilot",
        "status": "dry_run" if args.dry_run else "launching",
        "project_git": project_git,
        "upstream_arrow_commit": UPSTREAM_COMMIT,
        "source_config": str(source_path),
        "seed_index": args.seed,
        "seed": SEEDS[args.seed],
        "task_order": list(TASK_ORDER),
        "tasks_trained": [TASK_ORDER[0]],
        "task_identity_exposed_to_agent": True,
        "task_agnostic_claimed": False,
        "from_scratch": True,
        "sweep_profile": args.profile,
        "selection_family": "duration" if duration_sweep else "learning_rate",
        "profile_rationale": PROFILE_RATIONALE[args.profile],
        "profile_overrides": PROFILE_OVERRIDES.get(args.profile, {}),
        "task0_acquisition_epochs": int(config["epochs"]),
        "selection_metric": "Task0 pre-consolidation fixed-validation raw mean",
        "selection_rule": (
            "shortest duration within five percent of the best raw mean, then "
            "higher raw mean and lexical profile name"
            if duration_sweep
            else "maximum raw mean; exact ties use smaller log-distance from "
            "fixed_v1, then lexical profile name"
        ),
        "heldout_final_data_used_for_selection": False,
        "post_selection_requirement": (
            "launch the selected duration from scratch for the fixed-order full "
            "curriculum; confirmation seeds remain required"
            if duration_sweep
            else "launch the selected profile from scratch for the fixed 270-epoch "
            "MsPacman-Boxing-CrazyClimber protocol; confirmation seeds remain required"
        ),
        "budgets": _budget_manifest(config),
        "resolved_training_config": str(config_path),
        "output_dir": str(output_dir),
        "replay_mmap_root": (
            None
            if args.replay_mmap_root is None
            else str(args.replay_mmap_root.expanduser().resolve())
        ),
        "project_pythonpath_prepend": project_pythonpath,
        "world_model_compile": False,
        "command": command,
    }
    print(json.dumps(launch, indent=2))
    rendered_env = [f"{key}={value}" for key, value in thread_env.items()]
    rendered_env.append(f"PYTHONPATH={env['PYTHONPATH']}")
    print(f"command: {shlex.join([*rendered_env, *command])}")
    if args.dry_run:
        return 0

    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite run directory: {output_dir}")
    output_dir.mkdir(parents=True)
    replay_backing = _prepare_replay_symlink(output_dir, args.replay_mmap_root)
    _write_json(config_path, config)
    launch["status"] = "running"
    launch["started_at_utc"] = datetime.now(timezone.utc).isoformat()
    launch["runtime_environment"] = _runtime_info(python, env)
    launch["replay_mmap_backing"] = (
        None if replay_backing is None else str(replay_backing)
    )
    _write_json(output_dir / "launch.json", launch)

    return_code = _run_and_tee(
        command,
        cwd=ARROW_ROOT,
        env=env,
        log_path=output_dir / "train.log",
    )
    required = [
        "save_wm.pt",
        "save_ac.pt",
        "save_ac_bank.pt",
        "model_parameter_accounting.json",
        "actor_critic_parameter_accounting.json",
        "evolving_core_checkpoints/task_00_pre_consolidation.pt",
        "evolving_core_checkpoints/task_00_post_consolidation.pt",
        "evolving_core_consolidation/task_00_pre_validation.json",
    ]
    missing = [name for name in required if not (output_dir / name).is_file()]
    success = output_dir / "evolving_core_consolidation/task_00_boundary.json"
    failure = (
        output_dir
        / "evolving_core_checkpoints/task_00_consolidation_failure.json"
    )
    missing_consolidation_record = not success.is_file() and not failure.is_file()
    forbidden_final_evaluation = (output_dir / "final_evaluation.json").exists()
    candidate = None
    candidate_error = None
    if return_code == 0 and not missing:
        try:
            candidate = _selection_candidate(
                output_dir=output_dir, launch=launch, config=config
            )
            _write_json(output_dir / "task0_selection_candidate.json", candidate)
        except Exception as exc:  # persist exact eligibility failure below
            candidate_error = f"{type(exc).__name__}: {exc}"
    status = {
        "complete": return_code == 0
        and not missing
        and not missing_consolidation_record
        and not forbidden_final_evaluation
        and candidate is not None,
        "return_code": return_code,
        "missing_required_outputs": missing,
        "missing_consolidation_record": missing_consolidation_record,
        "forbidden_final_evaluation_present": forbidden_final_evaluation,
        "selection_candidate_error": candidate_error,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output_dir / "run_status.json", status)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    if not status["complete"]:
        raise RuntimeError(f"Task-0 sweep omitted required evidence: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
