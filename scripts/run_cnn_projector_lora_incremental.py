#!/usr/bin/env python3
"""Launch true Task-2/3 acquisition from a frozen Task-1 CNN/RSSM core."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
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
    THREAD_ENV_KEYS,
    UPSTREAM_COMMIT,
    _run_and_tee,
    _runtime_info,
    _write_json,
)


PROTOCOL = "CNN-Projector-RSSM-LoRA-ARROW-v1-Task1SnapshotSeeded-Atari-TaskAware"
EXPECTED_TASKS = (
    "ALE/MsPacman-v5",
    "ALE/Boxing-v5",
    "ALE/CrazyClimber-v5",
)
LORA_RANKS = (128, 128, 32)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task1-boundary-snapshot", type=Path, required=True)
    parser.add_argument("--source-config", type=Path)
    parser.add_argument(
        "--epochs-after-task1",
        type=int,
        default=180,
        help="Train this many epochs after the completed Task-1 boundary.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cpu-threads", type=int, default=12)
    parser.add_argument("--replay-mmap-root", type=Path)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_config_path(snapshot: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    inferred = snapshot.parent.parent / "resolved_training_config.json"
    if not inferred.is_file():
        raise FileNotFoundError(
            "Could not infer the source config next to the boundary snapshot: "
            f"{inferred}"
        )
    return inferred


def _incremental_config(source: dict, *, epochs_after_task1: int) -> dict:
    if epochs_after_task1 < 1:
        raise ValueError("--epochs-after-task1 must be positive")
    config = copy.deepcopy(source)
    task_names = tuple(task["name"] for task in config["esc"]["env_configs"])
    if task_names != EXPECTED_TASKS:
        raise ValueError(
            f"Expected the frozen three-task curriculum {EXPECTED_TASKS}, got {task_names}"
        )
    task_duration = int(config["esc"]["kwargs"]["swap_sched"])
    if task_duration != 90:
        raise ValueError("The first pilot fixes each task at 90 epochs")
    config.update(
        {
            "epochs": task_duration + epochs_after_task1,
            "continual_method": "cnn_projector_lora_arrow",
            "rssm_num_experts": 3,
            "dino_fullbank_current_task_fraction": 1.0,
            "observation_objective": "reconstruction",
            "observation_encoder": "cnn",
            "task_banked_image_encoder": False,
            "task_projected_image_encoder": True,
            "task_projector_bottleneck_features": 64,
            "task_lora_recurrent_rank": LORA_RANKS[0],
            "task_lora_representation_rank": LORA_RANKS[1],
            "task_lora_transition_rank": LORA_RANKS[2],
            "data_parallel_world_size": 1,
            "compute_dtype": "bfloat16",
            "replay_observation_dtype": "uint8",
            "random_policy": "new",
            "fresh_ac": False,
            "residual_correction": "none",
            "residual_consolidation": "none",
            "shared_core_mode": "task1_frozen_projector_lora",
            "independent_expert_original_task_index": None,
        }
    )
    for replay_config in config["replay_buffers"]:
        replay_config["rb_device"] = "cpu"
    return config


def _prepare_replay_symlink(output_dir: Path, mmap_root: Path | None) -> Path | None:
    if mmap_root is None:
        return None
    root = mmap_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    backing = root / output_dir.name
    if backing.exists() or backing.is_symlink():
        raise FileExistsError(f"Replay backing already exists: {backing}")
    backing.mkdir()
    link = output_dir / "mmap_replay"
    link.symlink_to(backing, target_is_directory=True)
    return backing


def main() -> int:
    args = _parser().parse_args()
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")
    snapshot = args.task1_boundary_snapshot.expanduser().resolve()
    if not snapshot.is_file():
        raise FileNotFoundError(f"Task-1 boundary snapshot does not exist: {snapshot}")
    checksum_path = snapshot.with_suffix(snapshot.suffix + ".sha256")
    if not checksum_path.is_file():
        raise FileNotFoundError(f"Task-1 checksum is missing: {checksum_path}")
    snapshot_sha = _sha256(snapshot)
    checksum_fields = checksum_path.read_text(encoding="ascii").split()
    if not checksum_fields or checksum_fields[0] != snapshot_sha:
        raise RuntimeError("Task-1 boundary checksum does not match the snapshot")

    source_config_path = _source_config_path(snapshot, args.source_config)
    source = json.loads(source_config_path.read_text(encoding="utf-8"))
    config = _incremental_config(
        source, epochs_after_task1=args.epochs_after_task1
    )
    python = args.python.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else snapshot.parent.parent.parent
        / (
            "cnn_projector_rssm_lora_incremental_"
            f"r{LORA_RANKS[0]}_{LORA_RANKS[1]}_{LORA_RANKS[2]}_"
            f"posttask1_e{args.epochs_after_task1}"
        )
    )
    config_path = output_dir / "resolved_training_config.json"
    evaluation_snapshot_dir = output_dir / "evaluation_snapshots"
    task_boundary_snapshot_dir = output_dir / "task_boundary_snapshots"

    project_git = (
        git_state(ROOT) if args.dry_run else require_synced_training_git_state(ROOT)
    )
    env = os.environ.copy()
    thread_env = {key: str(args.cpu_threads) for key in THREAD_ENV_KEYS}
    env.update(thread_env)
    project_pythonpath = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
    inherited_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (project_pythonpath, inherited_pythonpath) if value
    )

    command = [
        str(python),
        "Code/ARROW_and_DV3/Atari/train.py",
        "--config",
        str(config_path),
        "--arrow-replay-ratio",
        "50-50",
        "--log-dir",
        str(output_dir),
        "--evaluation-snapshot-dir",
        str(evaluation_snapshot_dir),
        "--task-bank-snapshot-dir",
        str(task_boundary_snapshot_dir),
        "--project-git-commit",
        str(project_git["commit"]),
        "--init-task1-boundary-snapshot",
        str(snapshot),
        "--fused-adam",
        "--tf32",
        "--profile-stages",
        "--evaluate-final",
    ]
    if not args.no_compile:
        command.append("--compile-world-model")

    post_task_epochs = args.epochs_after_task1
    decisions_per_epoch = int(config["n_sync"]) * int(config["gen_seq_len"])
    raw_frames_per_epoch = decisions_per_epoch * int(config["env_repeat"])
    budgets = {
        "source_completed_task1_epochs": 90,
        "new_training_epochs": post_task_epochs,
        "new_raw_environment_frames": raw_frames_per_epoch * post_task_epochs,
        "new_world_model_updates": int(config["steps_per_batch"])
        * post_task_epochs,
        "new_actor_critic_updates": int(config["ac_train_steps"])
        * post_task_epochs,
        "replay_initial_state": "empty",
        "source_replay_reused": False,
        "old_task_update_fraction": 0.0,
        "current_task_update_fraction": 1.0,
    }
    launch = {
        "schema_version": 1,
        "method": "CNN-Projector-RSSM-LoRA-ARROW",
        "protocol": PROTOCOL,
        "classification": "single_seed_snapshot_seeded_incremental_pilot",
        "status": "dry_run" if args.dry_run else "launching",
        "project_git": project_git,
        "upstream_arrow_commit": UPSTREAM_COMMIT,
        "source_config": str(source_config_path),
        "source_task1_boundary_snapshot": str(snapshot),
        "source_task1_boundary_sha256": snapshot_sha,
        "source_snapshot_resumable": False,
        "continuation_equivalence_claimed": False,
        "frozen_core": "completed Task-1 CNN encoder and RSSM route",
        "later_task_modules": {
            "per_task_projector": "residual 256x4x4 bottleneck-64 spatial projector",
            "per_task_rssm_lora_ranks": list(LORA_RANKS),
            "per_task_actor_critic": "fresh deterministic weights and optimizer",
            "per_task_decoder_reward_continue_heads": "copied from Task 1 then trained",
        },
        "task_identity_exposed_to_agent": True,
        "training_data_flow": "current-task ARROW replay only",
        "evaluation_transitions_enter_replay": False,
        "budgets": budgets,
        "resolved_training_config": str(config_path),
        "output_dir": str(output_dir),
        "replay_mmap_root": (
            str(args.replay_mmap_root.expanduser().resolve())
            if args.replay_mmap_root is not None
            else None
        ),
        "cpu_threads": args.cpu_threads,
        "environment": thread_env,
        "project_pythonpath_prepend": project_pythonpath,
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
        str(replay_backing) if replay_backing is not None else None
    )
    _write_json(output_dir / "launch.json", launch)

    return_code = _run_and_tee(
        command,
        cwd=ARROW_ROOT,
        env=env,
        log_path=output_dir / "train.log",
    )
    required_outputs = (
        "save_wm.pt",
        "save_ac.pt",
        "save_ac_bank.pt",
        "final_evaluation.json",
        "model_parameter_accounting.json",
        "actor_critic_parameter_accounting.json",
        "task1_seed_initialization.json",
    )
    missing_outputs = [
        name for name in required_outputs if not (output_dir / name).is_file()
    ]
    status = {
        "complete": return_code == 0 and not missing_outputs,
        "return_code": return_code,
        "missing_required_outputs": missing_outputs,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output_dir / "run_status.json", status)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    if missing_outputs:
        raise RuntimeError(f"Training omitted required outputs: {missing_outputs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
