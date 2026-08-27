#!/usr/bin/env python3
"""Launch compact Task-2/3 RSSM adaptation with one rehearsed shared actor."""

from __future__ import annotations

import argparse
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
from run_cnn_projector_lora_incremental import (
    EXPECTED_TASKS,
    _incremental_config as _projector_incremental_config,
    _prepare_replay_symlink,
    _sha256,
    _source_config_path,
)


PROTOCOL = (
    "CNN-Projector-CompactRSSM-SharedActor-ARROW-v1-"
    "Task1SnapshotSeeded-Atari-TaskAware"
)
ADAPTER_SIZES = {
    "recurrent_lora_rank": 0,
    "representation_lora_rank": 32,
    "transition_lora_rank": 32,
    "gru_output_adapter_features": 32,
}
DISTILLATION = {
    "scale": 1.0,
    "interval": 4,
    "n_sync": 128,
    "burnin_steps": 16,
    "dream_steps": 16,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task1-boundary-snapshot", type=Path, required=True)
    parser.add_argument("--source-config", type=Path)
    parser.add_argument("--epochs-after-task1", type=int, default=180)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cpu-threads", type=int, default=12)
    parser.add_argument("--replay-mmap-root", type=Path)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _incremental_config(source: dict, *, epochs_after_task1: int) -> dict:
    config = _projector_incremental_config(
        source, epochs_after_task1=epochs_after_task1
    )
    config.update(
        {
            "continual_method": "cnn_compact_shared_actor_arrow",
            "task_lora_recurrent_rank": ADAPTER_SIZES["recurrent_lora_rank"],
            "task_lora_representation_rank": ADAPTER_SIZES[
                "representation_lora_rank"
            ],
            "task_lora_transition_rank": ADAPTER_SIZES["transition_lora_rank"],
            "task_recurrent_output_adapter_features": ADAPTER_SIZES[
                "gru_output_adapter_features"
            ],
            "shared_actor_imagination_distillation": True,
            "shared_actor_distill_scale": DISTILLATION["scale"],
            "shared_actor_distill_interval": DISTILLATION["interval"],
            "shared_actor_distill_n_sync": DISTILLATION["n_sync"],
            "shared_actor_distill_burnin_steps": DISTILLATION[
                "burnin_steps"
            ],
            "shared_actor_distill_steps": DISTILLATION["dream_steps"],
            "shared_core_mode": "task1_frozen_projector_compact_rssm",
        }
    )
    return config


def _budgets(config: dict, *, epochs_after_task1: int) -> dict:
    decisions_per_epoch = int(config["n_sync"]) * int(config["gen_seq_len"])
    raw_frames_per_epoch = decisions_per_epoch * int(config["env_repeat"])
    ac_steps = int(config["ac_train_steps"])
    distill_batches_per_epoch = (
        ac_steps + DISTILLATION["interval"] - 1
    ) // DISTILLATION["interval"]
    distill_batches = distill_batches_per_epoch * epochs_after_task1
    distilled_states = (
        distill_batches
        * DISTILLATION["n_sync"]
        * DISTILLATION["dream_steps"]
    )
    burnin_state_uses = (
        distill_batches
        * DISTILLATION["n_sync"]
        * DISTILLATION["burnin_steps"]
    )
    return {
        "source_completed_task1_epochs": 90,
        "new_training_epochs": epochs_after_task1,
        "new_raw_environment_frames": raw_frames_per_epoch * epochs_after_task1,
        "new_world_model_updates": int(config["steps_per_batch"])
        * epochs_after_task1,
        "new_actor_critic_optimizer_updates": ac_steps * epochs_after_task1,
        "shared_actor_distillation_batches": distill_batches,
        "shared_actor_distilled_states": distilled_states,
        "shared_actor_burnin_state_uses": burnin_state_uses,
        "old_world_model_transition_uses": distilled_states + burnin_state_uses,
        "old_real_replay_samples": 0,
        "replay_initial_state": "empty",
        "source_replay_reused": False,
        "current_task_world_model_update_fraction": 1.0,
        "additional_imagination_compute_matched_to_prior_method": False,
    }


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
            "cnn_compact_shared_actor_incremental_"
            f"rep{ADAPTER_SIZES['representation_lora_rank']}_"
            f"trans{ADAPTER_SIZES['transition_lora_rank']}_"
            f"gruout{ADAPTER_SIZES['gru_output_adapter_features']}_"
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

    launch = {
        "schema_version": 1,
        "method": "CNN-Projector-CompactRSSM-SharedActor-ARROW",
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
        "frozen_core": "completed Task-1 CNN encoder and base RSSM route",
        "persistent_later_task_modules": {
            "projector": "residual 256x4x4 bottleneck-64 spatial projector",
            "rssm_adapters": dict(ADAPTER_SIZES),
            "actor_growth": "none; one shared actor is updated sequentially",
        },
        "training_only_state": {
            "critic": "one current shared critic; not used for evaluation actions",
            "old_policy_teacher": "one transient frozen previous shared actor",
            "task_heads": "decoder/reward/continue heads used by world-model training",
        },
        "old_policy_protection": {
            "state_source": "zero-initialized frozen old RSSM route imagination",
            "teacher": "previous shared actor",
            "loss": "KL(teacher || shared actor)",
            "old_task_selection": "round-robin at Task 3",
            "real_old_samples": 0,
            "limitation": "zero-state imagination may not cover replay-reachable states",
        },
        "task_identity_exposed_to_agent": True,
        "task_agnostic_routing_claimed": False,
        "training_data_flow": "current-task ARROW replay plus synthetic old-route states",
        "evaluation_transitions_enter_replay": False,
        "budgets": _budgets(config, epochs_after_task1=args.epochs_after_task1),
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
        "final_evaluation.json",
        "model_parameter_accounting.json",
        "actor_critic_parameter_accounting.json",
        "shared_actor_distillation_accounting.json",
        "task1_seed_initialization.json",
    )
    missing_outputs = [
        name for name in required_outputs if not (output_dir / name).is_file()
    ]
    unexpected_outputs = [
        name for name in ("save_ac_bank.pt",) if (output_dir / name).exists()
    ]
    status = {
        "complete": (
            return_code == 0 and not missing_outputs and not unexpected_outputs
        ),
        "return_code": return_code,
        "missing_required_outputs": missing_outputs,
        "unexpected_outputs": unexpected_outputs,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output_dir / "run_status.json", status)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    if missing_outputs or unexpected_outputs:
        raise RuntimeError(
            "Training artifact contract failed: "
            f"missing={missing_outputs} unexpected={unexpected_outputs}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
