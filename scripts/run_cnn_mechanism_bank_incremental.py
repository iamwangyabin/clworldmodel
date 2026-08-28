#!/usr/bin/env python3
"""Launch Task-2/3 MB-RSSM acquisition from a frozen Task-1 snapshot."""

from __future__ import annotations

import argparse
import copy
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
    CLASSIFICATIONS,
    EXPECTED_TASKS,
    _prepare_replay_symlink,
    _sha256,
    _source_config_path,
)


PROTOCOL = (
    "CNN-MechanismBank-RSSM-ARROW-v1-"
    "Task1SnapshotSeeded-Atari-TaskAware"
)
REC_PROTOCOL = (
    "REC-RSSM-ARROW-v1-Task1SnapshotSeeded-Atari-TaskAware"
)
MECHANISM_WIDTHS = (512, 512, 256)
MECHANISM_RESIDUAL_SCALE = 0.1
MECHANISM_PARAMETERS_PER_LATER_TASK = 3_816_192
REC_NUM_ATOMS = 4
REC_REUSE_PROBE_EPOCHS = 1
REC_ROUTE_LR_SCALE = 5.0
REC_CONSOLIDATION_BATCHES = 8
REC_MIN_CONTRIBUTION = 0.01
REC_MAX_VALIDATION_DROP = 0.05
REC_ROUTE_PARAMETERS_FOR_TASK3 = 12
REUSE_MODES = ("reuse", "no-reuse")
METHOD_PROFILES = ("mechanism-bank", "rec-rssm")
COMPILE_ENVIRONMENT_KEYS = ("TRITON_LIBCUDA_PATH",)


def _compile_environment_override(environment: dict[str, str]) -> dict[str, str]:
    """Return the explicit compiler-discovery environment recorded by a run."""
    return {
        key: environment[key]
        for key in COMPILE_ENVIRONMENT_KEYS
        if environment.get(key)
    }


def _parser(
    default_method_profile: str = "mechanism-bank",
) -> argparse.ArgumentParser:
    if default_method_profile not in METHOD_PROFILES:
        raise ValueError(f"Unknown default method profile: {default_method_profile}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task1-boundary-snapshot", type=Path, required=True)
    parser.add_argument("--source-config", type=Path)
    parser.add_argument("--epochs-after-task1", type=int, default=180)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cpu-threads", type=int, default=12)
    parser.add_argument("--replay-mmap-root", type=Path)
    parser.add_argument(
        "--classification", choices=CLASSIFICATIONS, default="pilot"
    )
    parser.add_argument(
        "--reuse-mode",
        choices=REUSE_MODES,
        default="reuse",
        help="Use old frozen mechanisms or run the capacity-matched NoReuse ablation.",
    )
    parser.add_argument(
        "--method-profile",
        choices=METHOD_PROFILES,
        default=default_method_profile,
    )
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _incremental_config(
    source: dict,
    *,
    epochs_after_task1: int,
    reuse_enabled: bool = True,
    method_profile: str = "mechanism-bank",
) -> dict:
    if epochs_after_task1 < 1:
        raise ValueError("--epochs-after-task1 must be positive")
    if method_profile not in METHOD_PROFILES:
        raise ValueError(f"Unknown method profile: {method_profile}")
    if method_profile == "rec-rssm" and not reuse_enabled:
        raise ValueError("REC-RSSM requires atom reuse")
    is_rec_rssm = method_profile == "rec-rssm"
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
            "continual_method": (
                "rec_rssm_arrow" if is_rec_rssm else "cnn_mechanism_bank_arrow"
            ),
            "rssm_num_experts": 3,
            "dino_fullbank_current_task_fraction": 1.0,
            "observation_objective": "reconstruction",
            "observation_encoder": "cnn",
            "task_banked_image_encoder": False,
            "task_projected_image_encoder": True,
            "task_projector_bottleneck_features": 64,
            "task_lora_recurrent_rank": 0,
            "task_lora_representation_rank": 0,
            "task_lora_transition_rank": 0,
            "task_recurrent_output_adapter_features": 0,
            "task_mechanism_bank": True,
            "task_mechanism_reuse": bool(reuse_enabled),
            "task_mechanism_recurrent_width": MECHANISM_WIDTHS[0],
            "task_mechanism_representation_width": MECHANISM_WIDTHS[1],
            "task_mechanism_transition_width": MECHANISM_WIDTHS[2],
            "task_mechanism_residual_scale": MECHANISM_RESIDUAL_SCALE,
            "task_mechanism_num_atoms": REC_NUM_ATOMS if is_rec_rssm else 1,
            "task_mechanism_reuse_probe_epochs": (
                REC_REUSE_PROBE_EPOCHS if is_rec_rssm else 0
            ),
            "task_mechanism_route_lr_scale": (
                REC_ROUTE_LR_SCALE if is_rec_rssm else 1.0
            ),
            "task_mechanism_consolidation_batches": REC_CONSOLIDATION_BATCHES,
            "task_mechanism_min_contribution": REC_MIN_CONTRIBUTION,
            "task_mechanism_max_validation_drop": REC_MAX_VALIDATION_DROP,
            "data_parallel_world_size": 1,
            "compute_dtype": "bfloat16",
            "replay_observation_dtype": "uint8",
            "random_policy": "new",
            "actor_network": "mlp",
            "fresh_ac": False,
            "shared_actor_imagination_distillation": False,
            "shared_actor_distill_scale": 0.0,
            "shared_actor_distill_interval": 1,
            "shared_actor_distill_n_sync": 1,
            "shared_actor_distill_burnin_steps": 0,
            "shared_actor_distill_steps": 1,
            "residual_correction": "none",
            "residual_consolidation": "none",
            "shared_core_mode": "task1_frozen_mechanism_bank",
            "independent_expert_original_task_index": None,
        }
    )
    for replay_config in config["replay_buffers"]:
        replay_config["rb_device"] = "cpu"
    return config


def _default_output_dir(
    snapshot: Path,
    *,
    epochs_after_task1: int,
    reuse_enabled: bool,
    method_profile: str = "mechanism-bank",
) -> Path:
    reuse_label = (
        "rec_rssm"
        if method_profile == "rec-rssm"
        else "reuse"
        if reuse_enabled
        else "no_reuse"
    )
    return (
        snapshot.parent.parent.parent
        / f"cnn_mechanism_bank_rssm_{reuse_label}_posttask1_e{epochs_after_task1}"
    )


def main(default_method_profile: str = "mechanism-bank") -> int:
    args = _parser(default_method_profile).parse_args()
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

    reuse_enabled = args.reuse_mode == "reuse"
    if args.method_profile == "rec-rssm" and not reuse_enabled:
        raise ValueError("REC-RSSM does not define a no-reuse launch mode")
    is_rec_rssm = args.method_profile == "rec-rssm"
    source_config_path = _source_config_path(snapshot, args.source_config)
    source = json.loads(source_config_path.read_text(encoding="utf-8"))
    config = _incremental_config(
        source,
        epochs_after_task1=args.epochs_after_task1,
        reuse_enabled=reuse_enabled,
        method_profile=args.method_profile,
    )
    python = args.python.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else _default_output_dir(
            snapshot,
            epochs_after_task1=args.epochs_after_task1,
            reuse_enabled=reuse_enabled,
            method_profile=args.method_profile,
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
    compile_environment_override = _compile_environment_override(env)

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
    if not args.no_compile and not is_rec_rssm:
        command.append("--compile-world-model")

    post_task_epochs = args.epochs_after_task1
    decisions_per_epoch = int(config["n_sync"]) * int(config["gen_seq_len"])
    raw_frames_per_epoch = decisions_per_epoch * int(config["env_repeat"])
    later_task_parameters = {
        "task_1": MECHANISM_PARAMETERS_PER_LATER_TASK,
        "task_2": MECHANISM_PARAMETERS_PER_LATER_TASK
        + (
            REC_ROUTE_PARAMETERS_FOR_TASK3
            if is_rec_rssm
            else 3
        ),
    }
    launch = {
        "schema_version": 1,
        "method": (
            "REC-RSSM" if is_rec_rssm else "CNN-MechanismBank-RSSM-ARROW"
        ),
        "protocol": REC_PROTOCOL if is_rec_rssm else PROTOCOL,
        "ablation": (
            "reuse_expand_consolidate"
            if is_rec_rssm
            else "reuse"
            if reuse_enabled
            else "no_reuse"
        ),
        "hypothesis": (
            "Lossless mechanism atoms permit reuse-first probing while a full "
            "new mechanism preserves the established plasticity floor."
            if is_rec_rssm
            else "A full new residual mechanism preserves later-task plasticity "
            "while zero-initialized tanh gates can reuse frozen older mechanisms."
        ),
        "classification": (
            "snapshot_seeded_incremental_smoke"
            if args.classification == "smoke"
            else "single_seed_snapshot_seeded_incremental_pilot"
        ),
        "status": "dry_run" if args.dry_run else "launching",
        "project_git": project_git,
        "upstream_arrow_commit": UPSTREAM_COMMIT,
        "source_config": str(source_config_path),
        "source_task1_boundary_snapshot": str(snapshot),
        "source_task1_boundary_sha256": snapshot_sha,
        "source_snapshot_resumable": False,
        "continuation_equivalence_claimed": False,
        "frozen_core": "completed Task-1 shared CNN and base RSSM",
        "later_task_modules": {
            "per_task_projector": "residual 256x4x4 bottleneck-64 spatial projector",
            "mechanism_widths_recurrent_posterior_prior": list(MECHANISM_WIDTHS),
            "mechanism_residual_scale": MECHANISM_RESIDUAL_SCALE,
            "reuse_gates": (
                "independent per-old-task per-atom tanh gates initialized to zero"
                if is_rec_rssm
                else "independent tanh scalars initialized to zero"
            ),
            "mechanism_atoms": (
                {
                    "count": REC_NUM_ATOMS,
                    "widths_recurrent_posterior_prior": [128, 128, 64],
                    "lossless_sum": True,
                }
                if is_rec_rssm
                else None
            ),
            "reuse_enabled": reuse_enabled,
            "no_reuse_gate_handling": (
                None if reuse_enabled else "allocated for capacity matching but frozen at zero"
            ),
            "rssm_parameters_by_later_task": later_task_parameters,
            "rssm_parameter_bytes_by_later_task": {
                key: 4 * value for key, value in later_task_parameters.items()
            },
            "per_task_actor_critic": "fresh deterministic weights and optimizer",
            "per_task_decoder_reward_continue_heads": (
                "copied from the previous task once, then trained"
            ),
        },
        "extra_world_model_losses": [],
        "actor_distillation": False,
        "task_identity_exposed_to_agent": True,
        "task_agnostic_claimed": False,
        "training_data_flow": "current-task ARROW replay only",
        "evaluation_transitions_enter_replay": False,
        "budgets": {
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
        },
        "resolved_training_config": str(config_path),
        "output_dir": str(output_dir),
        "replay_mmap_root": (
            str(args.replay_mmap_root.expanduser().resolve())
            if args.replay_mmap_root is not None
            else None
        ),
        "cpu_threads": args.cpu_threads,
        "environment": thread_env,
        "compile_environment_override": compile_environment_override,
        "world_model_compile": (
            "disabled_for_dynamic_reuse_probe_trainability"
            if is_rec_rssm
            else not args.no_compile
        ),
        "rec_rssm_phases": (
            {
                "reuse_probe_epochs": REC_REUSE_PROBE_EPOCHS,
                "reuse_probe_first_task": 2,
                "expand": "full current mechanism plus frozen old atom gates",
                "route_lr_scale": REC_ROUTE_LR_SCALE,
                "boundary_consolidation": {
                    "replay_batches": REC_CONSOLIDATION_BATCHES,
                    "minimum_contribution": REC_MIN_CONTRIBUTION,
                    "maximum_validation_drop": REC_MAX_VALIDATION_DROP,
                    "validation_rollouts_per_condition": 16,
                },
            }
            if is_rec_rssm
            else None
        ),
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
    if compile_environment_override:
        _write_json(
            output_dir / "runtime_compile_environment_override.json",
            {
                "schema_version": 1,
                "reason": "Triton libcuda discovery on the target node",
                "environment": compile_environment_override,
            },
        )

    return_code = _run_and_tee(
        command,
        cwd=ARROW_ROOT,
        env=env,
        log_path=output_dir / "train.log",
    )
    required_outputs = [
        "save_wm.pt",
        "save_ac.pt",
        "save_ac_bank.pt",
        "final_evaluation.json",
        "model_parameter_accounting.json",
        "actor_critic_parameter_accounting.json",
        "task1_seed_initialization.json",
    ]
    if is_rec_rssm and args.epochs_after_task1 >= 90:
        required_outputs.append(
            "rec_rssm_consolidation/task_01_boundary.json"
        )
    if is_rec_rssm and args.epochs_after_task1 >= 180:
        required_outputs.append(
            "rec_rssm_consolidation/task_02_boundary.json"
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
