#!/usr/bin/env python3
"""Launch Evolving-Core Atomic RSSM from scratch on a declared Atari order."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import shutil
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
FIXED_TASK0_PROFILE_LRS = {
    "fixed_v1": 2e-4,
    "fixed_v2": 3e-4,
}
PROTOCOLS = {
    "fixed_v1": "Evolving-Core-Atomic-RSSM-ARROW-v1-Atari-TaskAware",
    "fixed_v2": "Evolving-Core-Atomic-RSSM-ARROW-v2-Atari-TaskAware",
}
PROTOCOL = PROTOCOLS[FORMAL_TASK0_PROFILE]
ORIGINAL_SIX_TASK_PROTOCOL = (
    "Evolving-Core-Atomic-RSSM-ARROW-v2-OriginalSix-Atari-TaskAware-Pilot"
)
COMPACT_MECHANISM_ORIGINAL_SIX_PROTOCOL = (
    "Evolving-Core-Atomic-RSSM-CompactMechanism-128-128-64-ARROW-v1-"
    "OriginalSix-Atari-TaskAware-Pilot"
)
SHARED_DOWN_ORIGINAL_SIX_PROTOCOL = (
    "Evolving-Core-Atomic-RSSM-SharedFrozenDown-FiLM-ARROW-v1-"
    "OriginalSix-Atari-TaskAware-Pilot"
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
    "arrow-original-six": (
        "ALE/MsPacman-v5",
        "ALE/Boxing-v5",
        "ALE/CrazyClimber-v5",
        "ALE/Frostbite-v5",
        "ALE/Seaquest-v5",
        "ALE/Enduro-v5",
    ),
}
TASK_DURATION_EPOCHS = 90
DEFAULT_MECHANISM_PROFILE = "matched_512"
COMPACT_MECHANISM_PROFILE = "compact_128_128_64"
DENSE_PRIVATE_PARAMETERIZATION = "dense_private"
SHARED_DOWN_PARAMETERIZATION = "shared_frozen_down_film"
MECHANISM_PARAMETERIZATIONS = (
    DENSE_PRIVATE_PARAMETERIZATION,
    SHARED_DOWN_PARAMETERIZATION,
)
MECHANISM_PROFILE_WIDTHS = {
    DEFAULT_MECHANISM_PROFILE: (512, 512, 256),
    COMPACT_MECHANISM_PROFILE: (128, 128, 64),
}
ORIGINAL_SIX_MINIMUM_FREE_BYTES = 48 * 1024**3


def _task0_profile_for_order(
    task_order: str, task0_profile: str | None = None
) -> str:
    if task_order not in TASK_ORDERS:
        raise ValueError(f"Unknown Evolving-Core task order: {task_order!r}")
    resolved = task0_profile
    if resolved is None:
        resolved = (
            "fixed_v1"
            if task_order == "arrow-original-six"
            else FORMAL_TASK0_PROFILE
        )
    if resolved not in FIXED_TASK0_PROFILE_LRS:
        raise ValueError(
            f"Unknown full-curriculum Task-0 profile: {resolved!r}"
        )
    if task_order == "arrow-original-six" and resolved != "fixed_v1":
        raise ValueError(
            "The named original-six pilot preserves the fixed_v1 Task-0 profile"
        )
    return resolved


def _validate_mechanism_profile(
    task_order: str,
    mechanism_profile: str,
    mechanism_parameterization: str = DENSE_PRIVATE_PARAMETERIZATION,
) -> None:
    if task_order not in TASK_ORDERS:
        raise ValueError(f"Unknown Evolving-Core task order: {task_order!r}")
    if mechanism_profile not in MECHANISM_PROFILE_WIDTHS:
        raise ValueError(
            f"Unknown Evolving-Core mechanism profile: {mechanism_profile!r}"
        )
    if mechanism_parameterization not in MECHANISM_PARAMETERIZATIONS:
        raise ValueError(
            "Unknown Evolving-Core mechanism parameterization: "
            f"{mechanism_parameterization!r}"
        )
    if (
        mechanism_profile == COMPACT_MECHANISM_PROFILE
        and task_order != "arrow-original-six"
    ):
        raise ValueError(
            "The compact 128/128/64 mechanism capacity ablation is fixed to "
            "the complete ARROW original-six order"
        )
    if mechanism_parameterization == SHARED_DOWN_PARAMETERIZATION:
        if mechanism_profile != DEFAULT_MECHANISM_PROFILE:
            raise ValueError(
                "The shared-frozen-down parameterization preserves matched_512 widths"
            )
        if task_order != "arrow-original-six":
            raise ValueError(
                "The shared-frozen-down pilot is fixed to the complete ARROW "
                "original-six route allocation"
            )


def _protocol_for_task_order(
    task_order: str,
    mechanism_profile: str = DEFAULT_MECHANISM_PROFILE,
    mechanism_parameterization: str = DENSE_PRIVATE_PARAMETERIZATION,
    task0_profile: str | None = None,
) -> str:
    _validate_mechanism_profile(
        task_order, mechanism_profile, mechanism_parameterization
    )
    resolved_task0_profile = _task0_profile_for_order(task_order, task0_profile)
    if mechanism_parameterization == SHARED_DOWN_PARAMETERIZATION:
        return SHARED_DOWN_ORIGINAL_SIX_PROTOCOL
    if mechanism_profile == COMPACT_MECHANISM_PROFILE:
        return COMPACT_MECHANISM_ORIGINAL_SIX_PROTOCOL
    if task_order == "arrow-original-six":
        return ORIGINAL_SIX_TASK_PROTOCOL
    return PROTOCOLS[resolved_task0_profile]


def _residual_mechanism_parameters(
    *, in_features: int, out_features: int, hidden_features: int
) -> int:
    """Return LayerNorm/down/up parameters for one residual mechanism."""

    return (
        2 * in_features
        + in_features * hidden_features
        + hidden_features
        + hidden_features * out_features
        + out_features
    )


def _shared_down_private_parameters(
    *, in_features: int, out_features: int, hidden_features: int
) -> int:
    """Return private LayerNorm/FiLM/up parameters for one task."""

    return (
        2 * in_features
        + 2 * hidden_features
        + hidden_features * out_features
        + out_features
    )


def _shared_down_parameters(*, in_features: int, hidden_features: int) -> int:
    return in_features * hidden_features + hidden_features


def _mechanism_capacity_manifest(
    *,
    task_count: int,
    mechanism_profile: str,
    mechanism_parameterization: str = DENSE_PRIVATE_PARAMETERIZATION,
) -> dict[str, object]:
    """Record fixed Atari RSSM capacity before allocating the full model."""

    if task_count < 1:
        raise ValueError("task_count must be positive")
    if mechanism_profile not in MECHANISM_PROFILE_WIDTHS:
        raise ValueError(
            f"Unknown Evolving-Core mechanism profile: {mechanism_profile!r}"
        )
    if mechanism_parameterization not in MECHANISM_PARAMETERIZATIONS:
        raise ValueError(
            f"Unknown mechanism parameterization: {mechanism_parameterization!r}"
        )
    if (
        mechanism_parameterization == SHARED_DOWN_PARAMETERIZATION
        and mechanism_profile != DEFAULT_MECHANISM_PROFILE
    ):
        raise ValueError(
            "Shared-frozen-down accounting requires matched_512 hidden widths"
        )
    recurrent_width, representation_width, transition_width = (
        MECHANISM_PROFILE_WIDTHS[mechanism_profile]
    )
    parameter_counter = (
        _shared_down_private_parameters
        if mechanism_parameterization == SHARED_DOWN_PARAMETERIZATION
        else _residual_mechanism_parameters
    )
    per_task = {
        "recurrent": parameter_counter(
            in_features=512,
            out_features=512,
            hidden_features=recurrent_width,
        ),
        "representation_posterior": parameter_counter(
            in_features=4096 + 512,
            out_features=32 * 32,
            hidden_features=representation_width,
        ),
        "transition_prior": parameter_counter(
            in_features=512,
            out_features=32 * 32,
            hidden_features=transition_width,
        ),
    }
    per_task_total = sum(per_task.values())
    shared = {
        "recurrent": 0,
        "representation_posterior": 0,
        "transition_prior": 0,
    }
    if mechanism_parameterization == SHARED_DOWN_PARAMETERIZATION:
        shared = {
            "recurrent": _shared_down_parameters(
                in_features=512, hidden_features=recurrent_width
            ),
            "representation_posterior": _shared_down_parameters(
                in_features=4096 + 512, hidden_features=representation_width
            ),
            "transition_prior": _shared_down_parameters(
                in_features=512, hidden_features=transition_width
            ),
        }
    shared_total = sum(shared.values())
    route_parameters = 3 * 4 * sum(range(task_count))
    return {
        "profile": mechanism_profile,
        "parameterization": mechanism_parameterization,
        "widths": {
            "recurrent": recurrent_width,
            "representation_posterior": representation_width,
            "transition_prior": transition_width,
        },
        "fixed_interfaces": {
            "recurrent": [512, 512],
            "representation_posterior": [4608, 1024],
            "transition_prior": [512, 1024],
        },
        "atoms_per_mechanism": 4,
        "parameters_per_task": {**per_task, "total": per_task_total},
        "shared_frozen_down_parameters": {**shared, "total": shared_total},
        "private_mechanism_parameters": task_count * per_task_total,
        "reuse_route_parameters": route_parameters,
        "mechanism_and_route_parameters": (
            shared_total + task_count * per_task_total + route_parameters
        ),
    }


def _existing_ancestor(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists():
        if candidate.parent == candidate:
            raise FileNotFoundError(f"No existing ancestor for storage path: {path}")
        candidate = candidate.parent
    return candidate


def _storage_preflight(
    *, output_dir: Path, replay_mmap_root: Path | None, task_order: str
) -> dict[str, object]:
    """Reject a six-task launch without room for rolling atomic checkpoints."""

    output_ancestor = _existing_ancestor(output_dir.parent)
    replay_target = output_dir if replay_mmap_root is None else replay_mmap_root
    replay_ancestor = _existing_ancestor(replay_target)
    output_usage = shutil.disk_usage(output_ancestor)
    replay_usage = shutil.disk_usage(replay_ancestor)
    same_filesystem = output_ancestor.stat().st_dev == replay_ancestor.stat().st_dev
    required_output_bytes = (
        ORIGINAL_SIX_MINIMUM_FREE_BYTES
        if task_order == "arrow-original-six"
        else 0
    )
    if output_usage.free < required_output_bytes:
        raise RuntimeError(
            "Original-six Evolving-Core requires at least "
            f"{required_output_bytes / 1024**3:.0f} GiB free for live Replay, "
            "rolling boundary checkpoints, atomic-save temporaries, and logs; "
            f"found {output_usage.free / 1024**3:.1f} GiB at {output_ancestor}"
        )
    return {
        "output_existing_ancestor": str(output_ancestor),
        "replay_existing_ancestor": str(replay_ancestor),
        "same_filesystem": same_filesystem,
        "output_free_bytes": output_usage.free,
        "replay_free_bytes": replay_usage.free,
        "required_output_free_bytes": required_output_bytes,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=range(len(SEEDS)), default=0)
    parser.add_argument(
        "--task-order",
        choices=tuple(TASK_ORDERS),
        default="mspacman-boxing-crazyclimber",
    )
    parser.add_argument(
        "--mechanism-parameterization",
        choices=MECHANISM_PARAMETERIZATIONS,
        default=DENSE_PRIVATE_PARAMETERIZATION,
        help="Dense private mechanisms or a shared frozen full-width down basis.",
    )
    parser.add_argument(
        "--classification", choices=("pilot", "official"), default="pilot"
    )
    parser.add_argument(
        "--task0-profile",
        choices=tuple(FIXED_TASK0_PROFILE_LRS),
        default=None,
        help=(
            "Named optimizer profile. The three-task formal default is fixed_v2; "
            "the separately named original-six pilot preserves fixed_v1."
        ),
    )
    parser.add_argument(
        "--mechanism-profile",
        choices=tuple(MECHANISM_PROFILE_WIDTHS),
        default=DEFAULT_MECHANISM_PROFILE,
        help="Explicit mechanism-capacity preset; the default preserves v1/v2.",
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
    task0_profile: str | None = None,
    mechanism_profile: str = DEFAULT_MECHANISM_PROFILE,
    mechanism_parameterization: str = DENSE_PRIVATE_PARAMETERIZATION,
) -> dict:
    """Compose the fixed named protocol without changing existing baselines."""

    _validate_mechanism_profile(
        task_order, mechanism_profile, mechanism_parameterization
    )
    resolved_task0_profile = _task0_profile_for_order(task_order, task0_profile)
    mechanism_widths = MECHANISM_PROFILE_WIDTHS[mechanism_profile]
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
    task_count = len(TASK_ORDERS[task_order])
    config["epochs"] = task_count * TASK_DURATION_EPOCHS
    config.update(
        {
            "continual_method": "evolving_atomic_rssm_arrow",
            "rssm_num_experts": task_count,
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
            "task_mechanism_capacity_profile": mechanism_profile,
            "task_mechanism_parameterization": mechanism_parameterization,
            "task_mechanism_recurrent_width": mechanism_widths[0],
            "task_mechanism_representation_width": mechanism_widths[1],
            "task_mechanism_transition_width": mechanism_widths[2],
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
            "evolving_task0_profile": resolved_task0_profile,
            "evolving_shared_core": True,
            "evolving_checkpoint_retention": (
                "latest_boundary"
                if task_order == "arrow-original-six"
                else "all_boundaries"
            ),
            "first_task_shared_core_lr": FIXED_TASK0_PROFILE_LRS[
                resolved_task0_profile
            ],
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
            "task_private_heads": True,
            "task_private_actor_critic": True,
            "task_atomic_routes": True,
            "full_task_rssm_experts": False,
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


def _budget_manifest(config: dict) -> dict:
    task_count = len(config["esc"]["env_configs"])
    decisions_per_epoch = int(config["n_sync"]) * int(config["gen_seq_len"])
    raw_frames_per_epoch = decisions_per_epoch * int(config["env_repeat"])
    online_updates = int(config["epochs"]) * int(config["steps_per_batch"])
    consolidation_updates = task_count * int(
        config["boundary_consolidation_steps"]
    )
    task_updates = TASK_DURATION_EPOCHS * int(config["steps_per_batch"])
    replay_budget = _arrow_replay_storage_budget(config)
    checkpoint_retention = config.get(
        "evolving_checkpoint_retention", "all_boundaries"
    )
    retained_replay_boundaries = (
        1 if checkpoint_retention == "latest_boundary" else task_count
    )
    peak_replay_boundaries = (
        min(task_count, 2)
        if checkpoint_retention == "latest_boundary"
        else task_count
    )
    return {
        "task_count": task_count,
        "task_duration_epochs": [TASK_DURATION_EPOCHS] * task_count,
        "raw_environment_frames": raw_frames_per_epoch * int(config["epochs"]),
        "online_world_model_updates": online_updates,
        "boundary_consolidation_world_model_updates": consolidation_updates,
        "total_world_model_optimizer_steps": online_updates
        + consolidation_updates,
        "actor_critic_updates": int(config["epochs"])
        * int(config["ac_train_steps"]),
        "online_current_sequences": task_updates * int(config["mb_n_size"])
        + (task_count - 1) * task_updates * int(config["current_batch_n"]),
        "online_memory_sequences": (task_count - 1)
        * task_updates
        * int(config["memory_batch_n"]),
        "consolidation_sequences": consolidation_updates
        * int(config["mb_n_size"]),
        "online_sequence_batch_total": int(config["mb_n_size"]),
        "later_task_current_memory_split": [
            int(config["current_batch_n"]),
            int(config["memory_batch_n"]),
        ],
        "memory_task_selection": "uniform over completed tasks",
        "memory_source": "LTDM task-homogeneous sequences",
        "replay": replay_budget,
        "checkpoint_retention": checkpoint_retention,
        "retained_boundary_replay_asset_bytes": retained_replay_boundaries
        * int(replay_budget["observation_bytes"]),
        "peak_boundary_replay_asset_bytes": peak_replay_boundaries
        * int(replay_budget["observation_bytes"]),
        "minimum_live_plus_peak_replay_observation_bytes": (
            1 + peak_replay_boundaries
        )
        * int(replay_budget["observation_bytes"]),
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
        mechanism_profile=args.mechanism_profile,
        mechanism_parameterization=args.mechanism_parameterization,
    )
    task_count = len(TASK_ORDERS[args.task_order])
    resolved_task0_profile = config["evolving_task0_profile"]
    protocol = _protocol_for_task_order(
        args.task_order,
        mechanism_profile=args.mechanism_profile,
        mechanism_parameterization=args.mechanism_parameterization,
        task0_profile=resolved_task0_profile,
    )
    if args.task_order == "arrow-original-six" and args.classification != "pilot":
        raise ValueError("The original-six Evolving-Core campaign is pilot-only")
    python = args.python.expanduser().resolve()
    mechanism_output_suffix = (
        ""
        if args.mechanism_profile == DEFAULT_MECHANISM_PROFILE
        else f"_{args.mechanism_profile}"
    )
    if args.mechanism_parameterization != DENSE_PRIVATE_PARAMETERIZATION:
        mechanism_output_suffix += f"_{args.mechanism_parameterization}"
    task0_output_suffix = (
        ""
        if args.task_order == "arrow-original-six"
        else f"_{resolved_task0_profile}"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else ROOT
        / "runs"
        / (
            f"evolving_atomic_rssm{task0_output_suffix}_{args.task_order}"
            f"{mechanism_output_suffix}_"
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
            "Evolving-Core Atomic RSSM Shared Frozen Down + Private FiLM/Up"
            if args.mechanism_parameterization == SHARED_DOWN_PARAMETERIZATION
            else "Evolving-Core Atomic RSSM"
            if args.mechanism_profile == DEFAULT_MECHANISM_PROFILE
            else "Evolving-Core Atomic RSSM Compact Mechanism 128/128/64"
        ),
        "protocol": protocol,
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
        "source_task1_snapshot": None,
        "shared_core": "CNN plus posterior/recurrent/prior RSSM; always plastic",
        "private_state": "per-task projector, Q/F/P atoms, heads, actor-critic",
        "mechanism_capacity": _mechanism_capacity_manifest(
            task_count=task_count,
            mechanism_profile=args.mechanism_profile,
            mechanism_parameterization=args.mechanism_parameterization,
        ),
        "capacity_control_profile": DEFAULT_MECHANISM_PROFILE,
        "capacity_ablation_only": (
            args.mechanism_profile != DEFAULT_MECHANISM_PROFILE
            or args.mechanism_parameterization != DENSE_PRIVATE_PARAMETERIZATION
        ),
        "gradient_rule": "per-component conflicting-current-direction projection",
        "interface_distillation": {
            "posterior_kl": config["interface_q_scale"],
            "layer_normalized_hidden_mse": config["interface_h_scale"],
            "frozen_old_actor_kl": config["interface_actor_scale"],
        },
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
        "metric_reporting": {
            "schema": "arrow-paper-v1",
            "automatic_after_training": True,
            "required_output": str(output_dir / "continual_metrics.json"),
            "raw_checkpoint_matrix_preserved": True,
            "partial_curriculum_metric_suffix": "3",
            "published_arrow_direct_comparison": False,
        },
        "checkpoint_retention": config["evolving_checkpoint_retention"],
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
    launch["storage_preflight"] = _storage_preflight(
        output_dir=output_dir,
        replay_mmap_root=args.replay_mmap_root,
        task_order=args.task_order,
    )
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
        "final_evaluation.json",
        "model_parameter_accounting.json",
        "actor_critic_parameter_accounting.json",
    ]
    required_checkpoint_task_ids = (
        [task_count - 1]
        if config["evolving_checkpoint_retention"] == "latest_boundary"
        else range(task_count)
    )
    for task_id in required_checkpoint_task_ids:
        required.extend(
            [
                f"evolving_core_checkpoints/task_{task_id:02d}_pre_consolidation.pt",
                f"evolving_core_checkpoints/task_{task_id:02d}_post_consolidation.pt",
            ]
        )
    missing = [name for name in required if not (output_dir / name).is_file()]
    missing_consolidation_records = []
    for task_id in range(task_count):
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
