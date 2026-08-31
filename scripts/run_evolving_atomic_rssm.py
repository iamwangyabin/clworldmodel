#!/usr/bin/env python3
"""Launch Evolving-Core Atomic RSSM from scratch on a three-task Atari order."""

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
from launcher_support import (
    run_and_tee as _run_and_tee,
    runtime_info as _runtime_info,
    write_json as _write_json,
)
from run_arrow_ar50_atari import (
    ARROW_ROOT,
    FASTKAN_AC_STABLE_CONFIG_OVERRIDES,
    ROOT,
    SEEDS,
    THREAD_ENV_KEYS,
    UPSTREAM_COMMIT,
    _arrow_replay_storage_budget,
    _config_path,
    _verify_primary_config,
)
from run_cnn_projector_lora_incremental import _prepare_replay_symlink
from summarize_continual_metrics import build_run_report


FORMAL_TASK0_PROFILE = "fixed_v2"
PRIVATE_MLP_BEHAVIOR = "private_mlp"
SHARED_FASTKAN_STABLE_BEHAVIOR = "shared_fastkan_stable"
BEHAVIOR_PROFILES = (
    PRIVATE_MLP_BEHAVIOR,
    SHARED_FASTKAN_STABLE_BEHAVIOR,
)
FIXED_TASK0_PROFILE_LRS = {
    "fixed_v1": 2e-4,
    "fixed_v2": 3e-4,
}
PROTOCOLS = {
    "fixed_v1": "Evolving-Core-Atomic-RSSM-ARROW-v1-Atari-TaskAware",
    "fixed_v2": "Evolving-Core-Atomic-RSSM-ARROW-v2-Atari-TaskAware",
}
SHARED_FASTKAN_PROTOCOL = (
    "Evolving-Core-SharedFrozenDown-SharedFastKANAC-StableTargets-ARROW-v1-"
    "ThreeTask-Atari-TaskAware-Pilot"
)
TASK_ORDERS = {
    "mspacman-boxing-crazyclimber": (
        "ALE/MsPacman-v5",
        "ALE/Boxing-v5",
        "ALE/CrazyClimber-v5",
    ),
    "boxing-mspacman-crazyclimber": (
        "ALE/Boxing-v5",
        "ALE/MsPacman-v5",
        "ALE/CrazyClimber-v5",
    ),
    "crazyclimber-boxing-mspacman": (
        "ALE/CrazyClimber-v5",
        "ALE/Boxing-v5",
        "ALE/MsPacman-v5",
    ),
}
TASK_DURATION_EPOCHS = 90
TASK_COUNT = 3
MECHANISM_WIDTHS = (512, 512, 256)

# Exact FP32 online parameter counts for the fixed 64x64 Atari topology.  The
# launcher records these prospective counts and the trainer independently
# writes runtime tensor accounting for every actual run.
ARROW_WORLD_MODEL_PARAMETERS = 19_498_853
TASK_PROJECTOR_PARAMETERS = 34_240
TASK_MECHANISM_PARAMETERS = 3_816_192
SHARED_FROZEN_DOWN_PARAMETERS = 2_753_792
TASK_SHARED_DOWN_PRIVATE_MECHANISM_PARAMETERS = 1_064_960
TASK_PRIVATE_HEAD_ADDITION_PARAMETERS = 8_562_629
MLP_ACTOR_PARAMETERS = 797_202
MLP_CRITIC_PARAMETERS = 917_759
FASTKAN_ACTOR_PARAMETERS = 793_692
FASTKAN_CRITIC_PARAMETERS = 906_978


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=range(len(SEEDS)), default=0)
    parser.add_argument(
        "--task-order",
        choices=tuple(TASK_ORDERS),
        default="mspacman-boxing-crazyclimber",
    )
    parser.add_argument(
        "--classification", choices=("pilot", "official"), default="pilot"
    )
    parser.add_argument(
        "--task0-profile",
        choices=tuple(FIXED_TASK0_PROFILE_LRS),
        default=FORMAL_TASK0_PROFILE,
        help=(
            "Named full-curriculum optimizer profile. fixed_v2 is the formal "
            "default; fixed_v1 remains available for exact reproduction."
        ),
    )
    parser.add_argument(
        "--behavior-profile",
        choices=BEHAVIOR_PROFILES,
        default=PRIVATE_MLP_BEHAVIOR,
        help=(
            "private_mlp exactly reproduces Evolving-Core v1/v2 behavior banks; "
            "shared_fastkan_stable selects the separately named shared-frozen-"
            "down world model plus shared FastKAN Actor/Critic with replay rehearsal."
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--replay-mmap-root", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cpu-threads", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _resolved_config(
    source: dict,
    *,
    task_order: str,
    task0_profile: str = FORMAL_TASK0_PROFILE,
    behavior_profile: str = PRIVATE_MLP_BEHAVIOR,
) -> dict:
    """Compose the fixed named protocol without changing existing baselines."""

    if task_order not in TASK_ORDERS:
        raise ValueError(f"Unknown Evolving-Core task order: {task_order!r}")
    if task0_profile not in FIXED_TASK0_PROFILE_LRS:
        raise ValueError(
            f"Unknown full-curriculum Task-0 profile: {task0_profile!r}"
        )
    if behavior_profile not in BEHAVIOR_PROFILES:
        raise ValueError(f"Unknown behavior profile: {behavior_profile!r}")
    if (
        behavior_profile == SHARED_FASTKAN_STABLE_BEHAVIOR
        and task0_profile != "fixed_v2"
    ):
        raise ValueError(
            "Shared-Frozen-Down + Shared FastKAN inherits the fixed_v2 Task-0 "
            "world-model optimizer profile"
        )
    config = copy.deepcopy(source)
    by_name = {
        task["name"]: task for task in config["esc"]["env_configs"]
    }
    missing = [name for name in TASK_ORDERS[task_order] if name not in by_name]
    if missing:
        raise ValueError(f"Source config is missing required Atari tasks: {missing}")
    if config["esc"]["kwargs"].get("swap_sched") != TASK_DURATION_EPOCHS:
        raise ValueError("Evolving-Core v1 fixes every task at 90 epochs")
    config["esc"]["env_configs"] = [
        copy.deepcopy(by_name[name]) for name in TASK_ORDERS[task_order]
    ]
    config["epochs"] = TASK_COUNT * TASK_DURATION_EPOCHS
    config.update(
        {
            "continual_method": "evolving_atomic_rssm_arrow",
            "rssm_num_experts": TASK_COUNT,
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
            "task_mechanism_reuse": True,
            "task_mechanism_capacity_profile": "matched_512",
            "task_mechanism_parameterization": "dense_private",
            "task_mechanism_recurrent_width": MECHANISM_WIDTHS[0],
            "task_mechanism_representation_width": MECHANISM_WIDTHS[1],
            "task_mechanism_transition_width": MECHANISM_WIDTHS[2],
            "task_mechanism_residual_scale": 0.1,
            "task_mechanism_num_atoms": 4,
            "task_mechanism_reuse_probe_epochs": 0,
            "task_mechanism_route_lr_scale": 1.0,
            "task_mechanism_consolidation_batches": 8,
            "task_mechanism_min_contribution": 0.01,
            "task_mechanism_max_validation_drop": 0.05,
            "data_parallel_world_size": 1,
            "compute_dtype": "bfloat16",
            "replay_observation_dtype": "uint8",
            "random_policy": "new",
            "actor_network": "mlp",
            "ac_lr": 1e-4,
            "fresh_ac": False,
            "evaluation_seed_protocol": "fixed_validation_heldout_final",
            "evaluation_task_seed_offset": 0,
            "residual_correction": "none",
            "residual_consolidation": "none",
            "shared_core_mode": "evolving_replay_protected",
            "independent_expert_original_task_index": None,
            "evolving_task0_profile": task0_profile,
            "evolving_shared_core": True,
            "first_task_shared_core_lr": FIXED_TASK0_PROFILE_LRS[task0_profile],
            "shared_core_lr": 1e-4,
            "task_private_lr": 2e-4,
            "task_route_lr": 1e-3,
            "current_batch_n": 12,
            "memory_batch_n": 4,
            "memory_loss_scale": 1.0,
            "interface_q_scale": 0.1,
            "interface_h_scale": 0.05,
            "interface_actor_scale": 0.05,
            "component_gradient_projection": True,
            "task_atom_output_regularization": 1e-4,
            "boundary_consolidation_steps": 1000,
            "boundary_consolidation_lr": 2e-5,
            "boundary_max_return_drop": 0.05,
            "evolving_shared_behavior_current_task_fraction": 1.0,
            "task_private_heads": True,
            "task_private_actor_critic": True,
            "task_atomic_routes": True,
            "full_task_rssm_experts": False,
        }
    )
    if behavior_profile == SHARED_FASTKAN_STABLE_BEHAVIOR:
        config.update(FASTKAN_AC_STABLE_CONFIG_OVERRIDES)
        config.update(
            {
                "continual_method": "evolving_atomic_rssm_shared_fastkan_arrow",
                "task_mechanism_parameterization": "shared_frozen_down_film",
                "fresh_ac": False,
                "task_private_actor_critic": False,
                "evolving_shared_behavior_current_task_fraction": 0.75,
                "shared_actor_imagination_distillation": False,
                "shared_actor_distill_scale": 0.0,
                "shared_actor_distill_interval": 1,
                "shared_actor_distill_n_sync": 1,
                "shared_actor_distill_burnin_steps": 0,
                "shared_actor_distill_steps": 1,
            }
        )
    for replay_config in config["replay_buffers"]:
        replay_config["rb_device"] = "cpu"
    return config


def _training_command(
    *,
    python: Path,
    config_path: Path,
    output_dir: Path,
    task_snapshot_dir: Path,
    project_commit: str,
) -> list[str]:
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
        "--evaluate-final",
    ]


def _behavior_update_budget(config: dict) -> dict[str, int]:
    """Return exact routed Actor-Critic update counts without adding steps."""

    updates_per_epoch = int(config["ac_train_steps"])
    current_fraction = float(
        config.get("evolving_shared_behavior_current_task_fraction", 1.0)
    )
    totals = {task_id: 0 for task_id in range(TASK_COUNT)}
    for current_task_id in range(TASK_COUNT):
        if current_task_id == 0 or current_fraction == 1.0:
            allocation = {current_task_id: updates_per_epoch}
        else:
            current_updates = int(updates_per_epoch * current_fraction + 0.5)
            old_total = updates_per_epoch - current_updates
            quotient, remainder = divmod(old_total, current_task_id)
            allocation = {
                task_id: quotient + int(task_id < remainder)
                for task_id in range(current_task_id)
            }
            allocation[current_task_id] = current_updates
        for task_id, updates in allocation.items():
            totals[task_id] += TASK_DURATION_EPOCHS * updates
    return {str(task_id): updates for task_id, updates in totals.items()}


def _parameter_manifest(config: dict) -> dict:
    """Analytic ledger for the fixed three-task online inference topology."""

    task_count = len(config["esc"]["env_configs"])
    if task_count != TASK_COUNT:
        raise ValueError(
            f"Expected {TASK_COUNT} Evolving-Core tasks, got {task_count}"
        )
    mechanism_parameterization = config["task_mechanism_parameterization"]
    if mechanism_parameterization == "shared_frozen_down_film":
        shared_mechanism_parameters = SHARED_FROZEN_DOWN_PARAMETERS
        private_mechanism_parameters = (
            TASK_SHARED_DOWN_PRIVATE_MECHANISM_PARAMETERS
        )
    elif mechanism_parameterization == "dense_private":
        shared_mechanism_parameters = 0
        private_mechanism_parameters = TASK_MECHANISM_PARAMETERS
    else:
        raise ValueError(
            "Unknown mechanism parameterization in parameter ledger: "
            f"{mechanism_parameterization!r}"
        )
    mechanism_parameters = shared_mechanism_parameters + sum(
        private_mechanism_parameters + 12 * task_id
        for task_id in range(task_count)
    )
    world_model_parameters = (
        ARROW_WORLD_MODEL_PARAMETERS
        + task_count * TASK_PROJECTOR_PARAMETERS
        + mechanism_parameters
        + (task_count - 1) * TASK_PRIVATE_HEAD_ADDITION_PARAMETERS
    )
    shared_fastkan = (
        config["continual_method"]
        == "evolving_atomic_rssm_shared_fastkan_arrow"
    )
    mlp_pair = MLP_ACTOR_PARAMETERS + MLP_CRITIC_PARAMETERS
    fastkan_pair = FASTKAN_ACTOR_PARAMETERS + FASTKAN_CRITIC_PARAMETERS
    behavior_parameters = fastkan_pair if shared_fastkan else task_count * mlp_pair
    online_parameters = world_model_parameters + behavior_parameters
    matched_world_model_private_mlp_parameters = (
        world_model_parameters + task_count * mlp_pair
    )
    dense_v2_world_model_parameters = (
        ARROW_WORLD_MODEL_PARAMETERS
        + task_count * TASK_PROJECTOR_PARAMETERS
        + sum(
            TASK_MECHANISM_PARAMETERS + 12 * task_id
            for task_id in range(task_count)
        )
        + (task_count - 1) * TASK_PRIVATE_HEAD_ADDITION_PARAMETERS
    )
    dense_v2_online_parameters = (
        dense_v2_world_model_parameters + task_count * mlp_pair
    )
    arrow_online_parameters = ARROW_WORLD_MODEL_PARAMETERS + mlp_pair
    per_task_world_model_additions = {
        str(task_id): (
            TASK_PROJECTOR_PARAMETERS
            + private_mechanism_parameters
            + 12 * task_id
            + (
                TASK_PRIVATE_HEAD_ADDITION_PARAMETERS if task_id > 0 else 0
            )
        )
        for task_id in range(task_count)
    }
    runtime_verification_artifacts = [
        "model_parameter_accounting.json",
        "actor_critic_parameter_accounting.json",
    ]
    if shared_fastkan:
        runtime_verification_artifacts.append(
            "shared_behavior_replay_accounting.json"
        )
    result = {
        "schema_version": 1,
        "scope": (
            "online inference parameters; FP32 master weights; excludes optimizer "
            "state, gradients, activations, Replay, and boundary world-model teachers"
        ),
        "world_model_parameters": world_model_parameters,
        "mechanism_parameterization": mechanism_parameterization,
        "shared_frozen_down_parameters": shared_mechanism_parameters,
        "behavior_parameters": behavior_parameters,
        "online_parameters": online_parameters,
        "fp32_parameter_bytes": online_parameters * 4,
        "per_task_world_model_additions": per_task_world_model_additions,
        "per_later_task_behavior_growth": 0 if shared_fastkan else mlp_pair,
        "runtime_verification_artifacts": runtime_verification_artifacts,
        "comparison_to_matched_world_model_private_mlp": {
            "reference_parameters": matched_world_model_private_mlp_parameters,
            "difference": (
                online_parameters - matched_world_model_private_mlp_parameters
            ),
            "relative_difference": (
                online_parameters / matched_world_model_private_mlp_parameters
                - 1.0
            ),
        },
        "comparison_to_dense_evolving_v2_private_mlp": {
            "reference_parameters": dense_v2_online_parameters,
            "difference": online_parameters - dense_v2_online_parameters,
            "relative_difference": (
                online_parameters / dense_v2_online_parameters - 1.0
            ),
        },
        "comparison_to_arrow_50": {
            "reference_parameters": arrow_online_parameters,
            "difference": online_parameters - arrow_online_parameters,
            "relative_difference": online_parameters / arrow_online_parameters - 1.0,
        },
    }
    if shared_fastkan:
        result["training_only_behavior_copies"] = {
            "ema_slow_critic_parameters": FASTKAN_CRITIC_PARAMETERS,
            "transient_previous_boundary_actor_parameters": (
                FASTKAN_ACTOR_PARAMETERS
            ),
            "peak_behavior_parameters_excluding_optimizer": (
                fastkan_pair + FASTKAN_CRITIC_PARAMETERS + FASTKAN_ACTOR_PARAMETERS
            ),
            "common_evolving_boundary_world_model_teacher_excluded": True,
        }
    return result


def _budget_manifest(config: dict) -> dict:
    decisions_per_epoch = int(config["n_sync"]) * int(config["gen_seq_len"])
    raw_frames_per_epoch = decisions_per_epoch * int(config["env_repeat"])
    online_updates = int(config["epochs"]) * int(config["steps_per_batch"])
    consolidation_updates = TASK_COUNT * int(
        config["boundary_consolidation_steps"]
    )
    task_updates = TASK_DURATION_EPOCHS * int(config["steps_per_batch"])
    return {
        "task_duration_epochs": [TASK_DURATION_EPOCHS] * TASK_COUNT,
        "raw_environment_frames": raw_frames_per_epoch * int(config["epochs"]),
        "online_world_model_updates": online_updates,
        "boundary_consolidation_world_model_updates": consolidation_updates,
        "total_world_model_optimizer_steps": online_updates
        + consolidation_updates,
        "actor_critic_updates": int(config["epochs"])
        * int(config["ac_train_steps"]),
        "actor_critic_updates_by_task_route": _behavior_update_budget(config),
        "actor_critic_update_budget_fixed": True,
        "shared_behavior_rehearsal_adds_optimizer_steps": False,
        "online_current_sequences": task_updates * 16
        + (TASK_COUNT - 1) * task_updates * 12,
        "online_memory_sequences": (TASK_COUNT - 1) * task_updates * 4,
        "consolidation_sequences": consolidation_updates
        * int(config["mb_n_size"]),
        "online_sequence_batch_total": int(config["mb_n_size"]),
        "later_task_current_memory_split": [12, 4],
        "memory_task_selection": "uniform over completed tasks",
        "memory_source": "LTDM task-homogeneous sequences",
        "replay": _arrow_replay_storage_budget(config),
        "evaluation_transitions_enter_replay": False,
        "consolidation_is_extra_compute": True,
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
    config = _resolved_config(
        source,
        task_order=args.task_order,
        task0_profile=args.task0_profile,
        behavior_profile=args.behavior_profile,
    )
    python = args.python.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else ROOT
        / "runs"
        / (
            f"evolving_atomic_rssm_{args.task0_profile}_"
            f"{args.task_order}_s{args.seed}_{args.classification}"
            if args.behavior_profile == PRIVATE_MLP_BEHAVIOR
            else f"evolving_atomic_rssm_{args.task0_profile}_"
            f"{args.behavior_profile}_{args.task_order}_"
            f"s{args.seed}_{args.classification}"
        )
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
        "method": (
            "Evolving-Core Shared Frozen Down + Shared FastKAN Actor-Critic"
            if args.behavior_profile == SHARED_FASTKAN_STABLE_BEHAVIOR
            else "Evolving-Core Atomic RSSM"
        ),
        "protocol": (
            SHARED_FASTKAN_PROTOCOL
            if args.behavior_profile == SHARED_FASTKAN_STABLE_BEHAVIOR
            else PROTOCOLS[args.task0_profile]
        ),
        "classification": args.classification,
        "status": "dry_run" if args.dry_run else "launching",
        "project_git": project_git,
        "upstream_arrow_commit": UPSTREAM_COMMIT,
        "source_config": str(source_path),
        "seed_index": args.seed,
        "seed": SEEDS[args.seed],
        "task_order": list(TASK_ORDERS[args.task_order]),
        "task_identity_exposed_to_agent": True,
        "task_agnostic_claimed": False,
        "from_scratch": True,
        "behavior_profile": args.behavior_profile,
        "source_task1_snapshot": None,
        "shared_core": (
            "CNN plus posterior/recurrent/prior RSSM stays plastic; each Q/F/P "
            "bank also owns one frozen full-width down basis"
            if args.behavior_profile == SHARED_FASTKAN_STABLE_BEHAVIOR
            else "CNN plus posterior/recurrent/prior RSSM; always plastic"
        ),
        "private_state": (
            "per-task projector, Q/F/P LayerNorm-FiLM-up modules, and heads; "
            "no task-private behavior"
            if args.behavior_profile == SHARED_FASTKAN_STABLE_BEHAVIOR
            else "per-task projector, Q/F/P atoms, heads, actor-critic"
        ),
        "behavior_topology": (
            {
                "actor_critic": "one cross-task width-53 FastKAN pair",
                "stable_targets": True,
                "current_old_update_split": [0.75, 0.25],
                "old_task_selection": "uniform over completed task routes",
                "extra_optimizer_updates": 0,
            }
            if args.behavior_profile == SHARED_FASTKAN_STABLE_BEHAVIOR
            else {
                "actor_critic": "one independent MLP pair per task",
                "stable_targets": False,
                "current_old_update_split": [1.0, 0.0],
                "extra_optimizer_updates": 0,
            }
        ),
        "gradient_rule": "per-component conflicting-current-direction projection",
        "interface_distillation": {
            "posterior_kl": config["interface_q_scale"],
            "layer_normalized_hidden_mse": config["interface_h_scale"],
            "frozen_old_actor_kl": config["interface_actor_scale"],
        },
        "budgets": _budget_manifest(config),
        "parameter_budget": _parameter_manifest(config),
        "resolved_training_config": str(config_path),
        "output_dir": str(output_dir),
        "replay_mmap_root": (
            None
            if args.replay_mmap_root is None
            else str(args.replay_mmap_root.expanduser().resolve())
        ),
        "project_pythonpath_prepend": project_pythonpath,
        "world_model_compile": False,
        "metric_reporting": {
            "schema": "arrow-paper-v1",
            "automatic_after_training": True,
            "required_output": str(output_dir / "continual_metrics.json"),
            "raw_checkpoint_matrix_preserved": True,
            "partial_curriculum_metric_suffix": "3",
            "published_arrow_direct_comparison": False,
        },
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
        "final_evaluation.json",
        "model_parameter_accounting.json",
        "actor_critic_parameter_accounting.json",
    ]
    if args.behavior_profile == SHARED_FASTKAN_STABLE_BEHAVIOR:
        required.append("shared_behavior_replay_accounting.json")
    else:
        required.append("save_ac_bank.pt")
    for task_id in range(TASK_COUNT):
        required.extend(
            [
                f"evolving_core_checkpoints/task_{task_id:02d}_pre_consolidation.pt",
                f"evolving_core_checkpoints/task_{task_id:02d}_post_consolidation.pt",
            ]
        )
    missing = [name for name in required if not (output_dir / name).is_file()]
    missing_consolidation_records = []
    for task_id in range(TASK_COUNT):
        success = (
            output_dir
            / "evolving_core_consolidation"
            / f"task_{task_id:02d}_boundary.json"
        )
        failure = (
            output_dir
            / "evolving_core_checkpoints"
            / f"task_{task_id:02d}_consolidation_failure.json"
        )
        if not success.is_file() and not failure.is_file():
            missing_consolidation_records.append(task_id)
    metric_report_error = None
    metric_report_path = output_dir / "continual_metrics.json"
    if return_code == 0 and not missing and not missing_consolidation_records:
        try:
            _write_json(metric_report_path, build_run_report(output_dir))
        except (FileNotFoundError, KeyError, ValueError) as exc:
            metric_report_error = f"{type(exc).__name__}: {exc}"

    status = {
        "complete": return_code == 0
        and not missing
        and not missing_consolidation_records
        and metric_report_error is None,
        "return_code": return_code,
        "missing_required_outputs": missing,
        "missing_consolidation_records": missing_consolidation_records,
        "continual_metric_report": (
            str(metric_report_path) if metric_report_path.is_file() else None
        ),
        "continual_metric_report_error": metric_report_error,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output_dir / "run_status.json", status)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    if missing or missing_consolidation_records:
        raise RuntimeError(f"Training omitted required outputs: {status}")
    if metric_report_error is not None:
        raise RuntimeError(
            f"Training completed but metric reporting failed: {metric_report_error}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
