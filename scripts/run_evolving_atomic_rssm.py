#!/usr/bin/env python3
"""Launch AWM (Accumulative World Modeling) or AWM-AutoRoute on Atari."""

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
from launcher_support import prepare_replay_symlink as _prepare_replay_symlink
from summarize_continual_metrics import build_run_report


FORMAL_TASK0_PROFILE = 'fixed_v1'
PRIVATE_MLP_BEHAVIOR = "private_mlp"
PRIVATE_MLP_AUTOROUTE_BEHAVIOR = "private_mlp_autoroute"


ADAPTIVE_SHARED_RESIDUAL_MLP_BEHAVIOR = "shared_adaptive_residual_mlp"
BEHAVIOR_PROFILES = ('private_mlp', 'private_mlp_autoroute')
PRIVATE_PREDICTION_HEADS_PROFILE = "private"
SHARED_DISTILLED_HEADS_PROFILE = "shared_distilled"
PREDICTION_HEAD_PROFILES = ('shared_distilled',)
FIXED_TASK0_PROFILE_LRS = {'fixed_v1': 0.0002}
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

SHARED_DISTILLED_HEADS_THREE_TASK_PROTOCOL = (
    "Evolving-Core-DenseQFP-SharedDistilledHeads-PrivateMLPAC-ARROW-v1-"
    "ThreeTask-Atari-TaskAware-Pilot"
)
SHARED_DISTILLED_HEADS_ORIGINAL_SIX_PROTOCOL = (
    "Evolving-Core-DenseQFP-SharedDistilledHeads-PrivateMLPAC-ARROW-v1-"
    "OriginalSix-Atari-TaskAware-Pilot"
)
ADAPTIVE_QFP_COMPRESSION_PROTOCOL = (
    "Evolving-Core-DenseAcquire-ReturnGatedAdaptiveQFP-SharedDistilledHeads-"
    "PrivateMLPAC-ARROW-v1-OriginalSix-Atari-TaskAware-Pilot"
)
ADAPTIVE_QFP_COMPRESSION_METHOD = (
    "evolving_atomic_rssm_adaptive_compression_shared_heads_arrow"
)
ADAPTIVE_QFP_AC_COMPRESSION_PROTOCOL = (
    "Evolving-Core-DenseAcquire-ReturnGatedAdaptiveQFPAC-"
    "SharedDistilledHeads-SharedResidualMLPAC-ARROW-v1-"
    "OriginalSix-Atari-TaskAware-Pilot"
)
ADAPTIVE_QFP_AC_COMPRESSION_METHOD = (
    "evolving_atomic_rssm_adaptive_qfp_ac_compression_shared_heads_arrow"
)


D_AUTOROUTE_METHOD = (
    "evolving_atomic_rssm_adaptive_compression_shared_heads_autoroute_arrow"
)
D_AUTOROUTE_PROTOCOL = (
    "Evolving-Core-DenseAcquire-AdaptiveQFP-SharedHeads-PrivateMLPAC-"
    "FirstFrameRouter-ARROWParity-v2-OriginalSix-Atari-"
    "TaskAwareTraining-TaskIDFreeInference-Pilot"
)
AUTOROUTE_METHODS = (D_AUTOROUTE_METHOD,)
AUTOROUTE_BEHAVIORS = (PRIVATE_MLP_AUTOROUTE_BEHAVIOR,)
ADAPTIVE_QFP_WIDTH_FRACTIONS = (0.75, 0.5, 0.25, 0.125)
ADAPTIVE_QFP_STEPS_PER_CANDIDATE = 250
ADAPTIVE_QFP_LEARNING_RATE = 2e-4
ADAPTIVE_QFP_VALIDATION_ROLLOUTS = 16
ADAPTIVE_QFP_MAXIMUM_RETURN_DROP = 0.05
ADAPTIVE_QFP_DISTILL_SCALE = 1.0
TASK_ORDERS = {'arrow-original-six': ('ALE/MsPacman-v5',
                        'ALE/Boxing-v5',
                        'ALE/CrazyClimber-v5',
                        'ALE/Frostbite-v5',
                        'ALE/Seaquest-v5',
                        'ALE/Enduro-v5')}
TASK_DURATION_EPOCHS = 90
DEFAULT_MECHANISM_PROFILE = "matched_512"
COMPACT_MECHANISM_PROFILE = "compact_128_128_64"
DENSE_PRIVATE_PARAMETERIZATION = "dense_private"
ADAPTIVE_DENSE_WIDTH_PARAMETERIZATION = "adaptive_dense_width"
SHARED_DOWN_PARAMETERIZATION = "shared_frozen_down_film"
MECHANISM_PARAMETERIZATIONS = ('dense_private',)
MECHANISM_PROFILE_WIDTHS = {'matched_512': (512, 512, 256)}
ORIGINAL_SIX_MINIMUM_FREE_BYTES = 48 * 1024**3


def _task0_profile_for_order(task_order: str, task0_profile: str | None = None) -> str:
    if task_order != "arrow-original-six":
        raise ValueError("AWM-family protocols require the original-six task order")
    if task0_profile not in (None, "fixed_v1"):
        raise ValueError("AWM-family protocols preserve the fixed_v1 Task-0 profile")
    return "fixed_v1"


def _validate_mechanism_profile(
    task_order: str, mechanism_profile: str,
    mechanism_parameterization: str = DENSE_PRIVATE_PARAMETERIZATION,
) -> None:
    _task0_profile_for_order(task_order)
    if mechanism_profile != DEFAULT_MECHANISM_PROFILE or mechanism_parameterization != DENSE_PRIVATE_PARAMETERIZATION:
        raise ValueError("AWM-family acquisition requires dense matched_512 mechanisms")


def _protocol_for_task_order(
    task_order: str, mechanism_profile: str = DEFAULT_MECHANISM_PROFILE,
    mechanism_parameterization: str = DENSE_PRIVATE_PARAMETERIZATION,
    task0_profile: str | None = None,
    prediction_head_profile: str = SHARED_DISTILLED_HEADS_PROFILE,
    behavior_profile: str = PRIVATE_MLP_BEHAVIOR,
    adaptive_qfp_compression: bool = True,
) -> str:
    _validate_mechanism_profile(task_order, mechanism_profile, mechanism_parameterization)
    _task0_profile_for_order(task_order, task0_profile)
    if prediction_head_profile != SHARED_DISTILLED_HEADS_PROFILE:
        raise ValueError("AWM-family protocols require shared distilled prediction heads")
    if not adaptive_qfp_compression:
        raise ValueError("AWM-family protocols require adaptive Q/F/P compression")
    protocols = {PRIVATE_MLP_BEHAVIOR: ADAPTIVE_QFP_COMPRESSION_PROTOCOL, PRIVATE_MLP_AUTOROUTE_BEHAVIOR: D_AUTOROUTE_PROTOCOL}
    if behavior_profile not in protocols:
        raise ValueError(f"Unsupported AWM-family behavior profile: {behavior_profile!r}")
    return protocols[behavior_profile]


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
    if mechanism_parameterization not in {
        *MECHANISM_PARAMETERIZATIONS,
        ADAPTIVE_DENSE_WIDTH_PARAMETERIZATION,
    }:
        raise ValueError(
            f"Unknown mechanism parameterization: {mechanism_parameterization!r}"
        )
    recurrent_width, representation_width, transition_width = (
        MECHANISM_PROFILE_WIDTHS[mechanism_profile]
    )
    parameter_counter = (_residual_mechanism_parameters)
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
    shared_total = sum(shared.values())
    route_parameters = 3 * 4 * sum(range(task_count))
    result = {
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
    if mechanism_parameterization == ADAPTIVE_DENSE_WIDTH_PARAMETERIZATION:
        result["adaptive_compression"] = {
            "acquisition_widths": [
                recurrent_width,
                representation_width,
                transition_width,
            ],
            "candidate_width_fractions": list(ADAPTIVE_QFP_WIDTH_FRACTIONS),
            "candidate_widths": [
                [
                    int(round(recurrent_width * fraction)),
                    int(round(representation_width * fraction)),
                    int(round(transition_width * fraction)),
                ]
                for fraction in ADAPTIVE_QFP_WIDTH_FRACTIONS
            ],
            "physical_structured_pruning": True,
            "masked_full_width_storage": False,
        }
    return result


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
MLP_CRITIC_PARAMETERS = 918_783


ADAPTIVE_BEHAVIOR_HIDDEN_FEATURES = 512
ADAPTIVE_BEHAVIOR_NUM_ATOMS = 4
ADAPTIVE_BEHAVIOR_RESIDUAL_SCALE = 0.1
ADAPTIVE_BEHAVIOR_STEPS_PER_CANDIDATE = 250
ADAPTIVE_BEHAVIOR_LEARNING_RATE = 2e-4
ADAPTIVE_BEHAVIOR_VALIDATION_ROLLOUTS = 16
ADAPTIVE_BEHAVIOR_MAXIMUM_RETURN_DROP = 0.05
ADAPTIVE_BEHAVIOR_ACTOR_DISTILL_SCALE = 1.0
ADAPTIVE_BEHAVIOR_CRITIC_DISTILL_SCALE = 1.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=range(len(SEEDS)), default=0)
    parser.add_argument(
        "--task-order",
        choices=tuple(TASK_ORDERS),
        default="arrow-original-six",
    )
    parser.add_argument(
        "--prediction-head-profile",
        choices=PREDICTION_HEAD_PROFILES,
        default=SHARED_DISTILLED_HEADS_PROFILE,
        help=(
            "private reproduces Dense Evolving-Core; shared_distilled keeps one "
            "replay-protected decoder/reward/continue set while retaining private "
            "Dense Q/F/P and private MLP Actor-Critics."
        ),
    )
    parser.add_argument(
        "--adaptive-qfp-compression",
        action="store_true",
        default=True,
        help=(
            "Use the separately named original-six protocol that acquires each "
            "task at Dense 512/512/256 width, evaluates all fixed structured-"
            "pruning candidates on a dedicated validation cohort, and retains "
            "the smallest candidate within the raw-return gate."
        ),
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
    parser.add_argument(
        "--behavior-profile",
        choices=BEHAVIOR_PROFILES,
        default=PRIVATE_MLP_BEHAVIOR,
        help=(
            "private_mlp selects AWM (Accumulative World Modeling, formerly D); "
            "private_mlp_autoroute selects AWM-AutoRoute, preserving AWM's private "
            "MLPs and adding first-frame reconstruction routing."
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--replay-mmap-root", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cpu-threads", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _resolved_config(
    source: dict, *, task_order: str = "arrow-original-six",
    task0_profile: str | None = None,
    mechanism_profile: str = DEFAULT_MECHANISM_PROFILE,
    mechanism_parameterization: str = DENSE_PRIVATE_PARAMETERIZATION,
    behavior_profile: str = PRIVATE_MLP_BEHAVIOR,
    prediction_head_profile: str = SHARED_DISTILLED_HEADS_PROFILE,
    adaptive_qfp_compression: bool = True,
) -> dict:
    """Compose AWM or AWM-AutoRoute without inheriting any retired method preset."""
    _protocol_for_task_order(
        task_order, mechanism_profile, mechanism_parameterization,
        task0_profile, prediction_head_profile, behavior_profile,
        adaptive_qfp_compression,
    )
    config = copy.deepcopy(source)
    by_name = {task["name"]: task for task in config["esc"]["env_configs"]}
    missing = [name for name in TASK_ORDERS[task_order] if name not in by_name]
    if missing:
        raise ValueError(f"Source config is missing required Atari tasks: {missing}")
    if config["esc"]["kwargs"].get("swap_sched") != TASK_DURATION_EPOCHS:
        raise ValueError("AWM-family protocols fix every task at 90 epochs")
    config["esc"]["env_configs"] = [copy.deepcopy(by_name[name]) for name in TASK_ORDERS[task_order]]
    config["epochs"] = len(TASK_ORDERS[task_order]) * TASK_DURATION_EPOCHS
    config.update({'random_policy': 'new',
 'continual_method': 'evolving_atomic_rssm_adaptive_compression_shared_heads_arrow',
 'rssm_num_experts': 6,
 'observation_objective': 'reconstruction',
 'observation_encoder': 'cnn',
 'task_projected_image_encoder': True,
 'task_projector_bottleneck_features': 64,
 'task_mechanism_bank': True,
 'task_mechanism_reuse': True,
 'task_mechanism_capacity_profile': 'matched_512',
 'task_mechanism_parameterization': 'adaptive_dense_width',
 'task_mechanism_recurrent_width': 512,
 'task_mechanism_representation_width': 512,
 'task_mechanism_transition_width': 256,
 'task_mechanism_residual_scale': 0.1,
 'task_mechanism_num_atoms': 4,
 'data_parallel_world_size': 1,
 'compute_dtype': 'bfloat16',
 'replay_observation_dtype': 'uint8',
 'actor_network': 'mlp',
 'ac_lr': 0.0001,
 'evaluation_seed_protocol': 'fixed_validation_heldout_final',
 'evaluation_task_seed_offset': 0,
 'shared_core_mode': 'evolving_replay_protected',
 'evolving_task0_profile': 'fixed_v1',
 'evolving_shared_core': True,
 'evolving_checkpoint_retention': 'latest_boundary',
 'first_task_shared_core_lr': 0.0002,
 'shared_core_lr': 0.0001,
 'task_private_lr': 0.0002,
 'task_route_lr': 0.001,
 'current_batch_n': 12,
 'memory_batch_n': 4,
 'memory_loss_scale': 1.0,
 'interface_q_scale': 0.1,
 'interface_h_scale': 0.05,
 'interface_actor_scale': 0.05,
 'component_gradient_projection': True,
 'task_atom_output_regularization': 0.0001,
 'boundary_consolidation_steps': 1000,
 'boundary_consolidation_lr': 2e-05,
 'boundary_max_return_drop': 0.05,
 'evolving_shared_behavior_current_task_fraction': 1.0,
 'task_shared_prediction_heads': True,
 'shared_prediction_distill_scale': 0.1,
 'task_private_actor_critic': True,
 'task_atomic_routes': True,
 'adaptive_compression_width_fractions': [0.75, 0.5, 0.25, 0.125],
 'adaptive_compression_steps_per_candidate': 250,
 'adaptive_compression_lr': 0.0002,
 'adaptive_compression_rollouts': 16,
 'adaptive_compression_max_return_drop': 0.05,
 'adaptive_compression_qfp_distill_scale': 1.0})
    if behavior_profile in AUTOROUTE_BEHAVIORS:
        if behavior_profile == PRIVATE_MLP_AUTOROUTE_BEHAVIOR:
            config["continual_method"] = D_AUTOROUTE_METHOD
        config["task_route_inference"] = "first_frame_reconstruction"
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

    task_count = len(config["esc"]["env_configs"])
    updates_per_epoch = int(config["ac_train_steps"])
    current_fraction = float(
        config.get("evolving_shared_behavior_current_task_fraction", 1.0)
    )
    totals = {task_id: 0 for task_id in range(task_count)}
    for current_task_id in range(task_count):
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
    """Analytic ledger for a declared Evolving-Core online topology."""

    task_count = len(config["esc"]["env_configs"])
    if task_count < 1:
        raise ValueError("Evolving-Core parameter accounting requires tasks")
    mechanism_parameterization = config["task_mechanism_parameterization"]
    if mechanism_parameterization in {
        "dense_private",
        ADAPTIVE_DENSE_WIDTH_PARAMETERIZATION,
    }:
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
    shared_prediction_heads = bool(
        config.get("task_shared_prediction_heads", False)
    )
    world_model_parameters = (
        ARROW_WORLD_MODEL_PARAMETERS
        + task_count * TASK_PROJECTOR_PARAMETERS
        + mechanism_parameters
        + (
            0
            if shared_prediction_heads
            else (task_count - 1) * TASK_PRIVATE_HEAD_ADDITION_PARAMETERS
        )
    )

    mlp_pair = MLP_ACTOR_PARAMETERS + MLP_CRITIC_PARAMETERS

    adaptive_hidden = 512
    adaptive_actor_residual = _residual_mechanism_parameters(
        in_features=1536,
        out_features=18,
        hidden_features=adaptive_hidden,
    )
    adaptive_critic_residual = _residual_mechanism_parameters(
        in_features=1536,
        out_features=255,
        hidden_features=adaptive_hidden,
    )
    adaptive_residual_pair = adaptive_actor_residual + adaptive_critic_residual
    adaptive_route_parameters = (
        2 * 4
        * sum(range(task_count))
    )
    behavior_parameters = ((task_count * mlp_pair))
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
                TASK_PRIVATE_HEAD_ADDITION_PARAMETERS
                if task_id > 0 and not shared_prediction_heads
                else 0
            )
        )
        for task_id in range(task_count)
    }
    runtime_verification_artifacts = [
        "model_parameter_accounting.json",
        "actor_critic_parameter_accounting.json",
    ]
    result = {
        "schema_version": 1,
        "scope": (
            "online inference parameters; FP32 master weights; excludes optimizer "
            "state, gradients, activations, Replay, and boundary world-model teachers"
        ),
        "world_model_parameters": world_model_parameters,
        "prediction_head_topology": (
            "single_shared" if shared_prediction_heads else "per_task_private"
        ),
        "shared_prediction_head_parameters": (
            TASK_PRIVATE_HEAD_ADDITION_PARAMETERS
            if shared_prediction_heads
            else 0
        ),
        "mechanism_parameterization": mechanism_parameterization,
        "shared_frozen_down_parameters": shared_mechanism_parameters,
        "behavior_topology": (("per_task_private_mlp")),
        "behavior_parameters": behavior_parameters,
        "learned_router_parameters": 0,
        "online_parameters": online_parameters,
        "fp32_parameter_bytes": online_parameters * 4,
        "per_task_world_model_additions": per_task_world_model_additions,
        "per_later_task_behavior_growth": ((mlp_pair)),
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
    if shared_prediction_heads:
        result["training_only_prediction_head_teacher"] = {
            "parameters": TASK_PRIVATE_HEAD_ADDITION_PARAMETERS,
            "additional_teacher_copy": False,
            "contained_in_common_evolving_boundary_world_model_teacher": True,
            "growth_with_task_count": 0,
        }
    if mechanism_parameterization == ADAPTIVE_DENSE_WIDTH_PARAMETERIZATION:
        dense_widths = [
            int(config["task_mechanism_recurrent_width"]),
            int(config["task_mechanism_representation_width"]),
            int(config["task_mechanism_transition_width"]),
        ]
        fractions = [
            float(value)
            for value in config["adaptive_compression_width_fractions"]
        ]
        minimum_widths = [
            int(round(width * fractions[-1])) for width in dense_widths
        ]
        minimum_per_task_mechanism_parameters = sum(
            (
                _residual_mechanism_parameters(
                    in_features=512,
                    out_features=512,
                    hidden_features=minimum_widths[0],
                ),
                _residual_mechanism_parameters(
                    in_features=4096 + 512,
                    out_features=32 * 32,
                    hidden_features=minimum_widths[1],
                ),
                _residual_mechanism_parameters(
                    in_features=512,
                    out_features=32 * 32,
                    hidden_features=minimum_widths[2],
                ),
            )
        )
        minimum_mechanism_parameters = sum(
            minimum_per_task_mechanism_parameters + 12 * task_id
            for task_id in range(task_count)
        )
        non_mechanism_world_model_parameters = (
            world_model_parameters - mechanism_parameters
        )
        minimum_final_world_model_parameters = (
            non_mechanism_world_model_parameters + minimum_mechanism_parameters
        )
        minimum_final_online_parameters = (
            minimum_final_world_model_parameters + behavior_parameters
        )
        result["adaptive_compression"] = {
            "outcome_dependent": True,
            "dense_acquisition_widths": dense_widths,
            "candidate_width_fractions": fractions,
            "candidate_widths": [
                [int(round(width * fraction)) for width in dense_widths]
                for fraction in fractions
            ],
            "minimum_candidate_widths": minimum_widths,
            "minimum_final_world_model_parameters": (
                minimum_final_world_model_parameters
            ),
            "minimum_final_online_parameters": minimum_final_online_parameters,
            "maximum_final_online_parameters": online_parameters,
            "maximum_acquisition_online_parameters": online_parameters,
            "dense_fallback_retained_when_no_candidate_passes": True,
            "selection_metric": "current-task oracle raw episodic return",
            "final_heldout_cohort_used_for_selection": False,
            "full_dense_teacher_persistent": False,
        }
    return result


def _budget_manifest(config: dict) -> dict:
    task_count = len(config["esc"]["env_configs"])
    decisions_per_epoch = int(config["n_sync"]) * int(config["gen_seq_len"])
    raw_frames_per_epoch = decisions_per_epoch * int(config["env_repeat"])
    online_updates = int(config["epochs"]) * int(config["steps_per_batch"])
    consolidation_updates = task_count * int(
        config["boundary_consolidation_steps"]
    )
    adaptive_compression = config.get("continual_method") in {ADAPTIVE_QFP_COMPRESSION_METHOD, D_AUTOROUTE_METHOD}
    adaptive_behavior_compression = (
        config.get("continual_method") == 'evolving_atomic_rssm_adaptive_qfp_ac_compression_shared_heads_arrow'
    )
    adaptive_compression_updates = (
        task_count
        * len(config.get("adaptive_compression_width_fractions", ()))
        * int(config.get("adaptive_compression_steps_per_candidate", 0))
        if adaptive_compression
        else 0
    )
    adaptive_compression_sequences = adaptive_compression_updates * int(
        config["mb_n_size"]
    )
    adaptive_behavior_compression_updates = (
        task_count
        * 0
        * 0
        if adaptive_behavior_compression
        else 0
    )
    adaptive_behavior_compression_imagined_states = (
        0
        * int(config["mb_n_size"])
        * int(config.get("ac_dream_steps", 16))
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
        "adaptive_compression_world_model_updates": adaptive_compression_updates,
        "adaptive_compression_sequences": adaptive_compression_sequences,
        "adaptive_compression_validation_rollouts": (
            task_count
            * (1 + len(config.get("adaptive_compression_width_fractions", ())))
            * int(config.get("adaptive_compression_rollouts", 0))
            if adaptive_compression
            else 0
        ),
        "adaptive_compression_validation_scope": (
            "current_task_oracle" if adaptive_compression else None
        ),
        "inference_router_adds_compute": config.get("continual_method") in AUTOROUTE_METHODS,
        "evaluation_episode_count_mode": config.get("evaluation_episode_count_mode", "legacy"),
        "adaptive_behavior_compression_updates": 0,
        "adaptive_behavior_compression_imagined_states": (
            adaptive_behavior_compression_imagined_states
        ),
        "adaptive_behavior_compression_validation_rollouts": (
            task_count
            * (
                1
                + 0
            )
            * 0
            if adaptive_behavior_compression
            else 0
        ),
        "total_world_model_optimizer_steps": online_updates
        + consolidation_updates
        + adaptive_compression_updates,
        "actor_critic_updates": int(config["epochs"])
        * int(config["ac_train_steps"]),
        "total_actor_critic_optimizer_steps": (
            int(config["epochs"]) * int(config["ac_train_steps"])
            + 0
        ),
        "online_current_sequences": task_updates * int(config["mb_n_size"])
        + (task_count - 1) * task_updates * int(config["current_batch_n"]),
        "online_memory_sequences": (task_count - 1)
        * task_updates
        * int(config["memory_batch_n"]),
        "actor_critic_updates_by_task_route": _behavior_update_budget(config),
        "actor_critic_update_budget_fixed": True,
        "shared_behavior_rehearsal_adds_optimizer_steps": False,
        "shared_prediction_distillation_adds_optimizer_steps": False,
        "shared_prediction_distillation_reuses_memory_teacher_forward": bool(
            config.get("task_shared_prediction_heads", False)
        ),
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
        "adaptive_compression_is_extra_compute": adaptive_compression,
        "adaptive_compression_candidate_compute_is_fixed": adaptive_compression,
        "adaptive_compression_replay_source": (
            "completed-task LTDM" if adaptive_compression else None
        ),
        "adaptive_behavior_compression_is_extra_compute": (
            adaptive_behavior_compression
        ),
        "adaptive_behavior_compression_candidate_compute_is_fixed": (
            adaptive_behavior_compression
        ),
        "adaptive_behavior_compression_replay_source": (
            "completed-task LTDM imagined states"
            if adaptive_behavior_compression
            else None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
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
        behavior_profile=args.behavior_profile,
        prediction_head_profile=args.prediction_head_profile,
        adaptive_qfp_compression=args.adaptive_qfp_compression,
    )
    task_count = len(TASK_ORDERS[args.task_order])
    resolved_task0_profile = config["evolving_task0_profile"]
    protocol = _protocol_for_task_order(
        args.task_order,
        mechanism_profile=args.mechanism_profile,
        mechanism_parameterization=args.mechanism_parameterization,
        task0_profile=resolved_task0_profile,
        prediction_head_profile=args.prediction_head_profile,
        behavior_profile=args.behavior_profile,
        adaptive_qfp_compression=args.adaptive_qfp_compression,
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
    behavior_output_suffix = (
        ""
        if args.behavior_profile == PRIVATE_MLP_BEHAVIOR
        else f"_{args.behavior_profile}"
    )
    prediction_head_output_suffix = (
        ""
        if args.prediction_head_profile == PRIVATE_PREDICTION_HEADS_PROFILE
        else f"_{args.prediction_head_profile}_heads"
    )
    adaptive_compression_output_suffix = (
        "_adaptive_qfp_compression" if args.adaptive_qfp_compression else ""
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else ROOT
        / "runs"
        / (
            f"evolving_atomic_rssm{task0_output_suffix}{behavior_output_suffix}"
            f"{prediction_head_output_suffix}{adaptive_compression_output_suffix}_"
            f"{args.task_order}"
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
            "AWM-AutoRoute (Accumulative World Modeling with "
            "First-Frame Reconstruction Routing)"
            if args.behavior_profile == PRIVATE_MLP_AUTOROUTE_BEHAVIOR
            else
            ("AWM (Accumulative World Modeling)"
            if args.adaptive_qfp_compression
            else ("Evolving-Core Dense Q/F/P + Shared Distilled Prediction Heads "
            "+ Private MLP Actor-Critic"
            if args.prediction_head_profile == SHARED_DISTILLED_HEADS_PROFILE
            else
            ("Evolving-Core Atomic RSSM"
            if args.mechanism_profile == DEFAULT_MECHANISM_PROFILE
            else "Evolving-Core Atomic RSSM Compact Mechanism 128/128/64")))
        ),
        "protocol": (protocol),
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
        "task_identity_exposed_during_training": True,
        "task_identity_exposed_during_action_selection": (
            args.behavior_profile not in AUTOROUTE_BEHAVIORS
        ),
        "inference_routing": {
            "mode": config.get("task_route_inference", "oracle"),
            "eligible_routes": "acquired slots plus currently acquiring slot; never future slots",
            "episode_lock": args.behavior_profile in AUTOROUTE_BEHAVIORS,
            "learned_router_parameters": 0,
            "evaluation_episode_count_mode": config.get("evaluation_episode_count_mode", "legacy"),
            "extra_inference_compute": "one RSSM posterior plus decoder per eligible route at episode start",
        },
        "from_scratch": True,
        "behavior_profile": args.behavior_profile,
        "prediction_head_profile": args.prediction_head_profile,
        "adaptive_qfp_compression": args.adaptive_qfp_compression,
        "source_task1_snapshot": None,
        "shared_core": ("CNN, posterior/recurrent/prior RSSM, and one shared "
            "decoder/reward/continue set; always plastic and replay protected"
            if args.prediction_head_profile == SHARED_DISTILLED_HEADS_PROFILE
            else "CNN plus posterior/recurrent/prior RSSM; always plastic"),
        "private_state": (
            ("per-task projector, physically width-adaptive Dense Q/F/P atoms, "
            "routes, and independent MLP actor-critic; decoder/reward/continue "
            "are shared"
            if args.adaptive_qfp_compression
            else ("per-task projector, dense Q/F/P atoms, routes, and independent "
            "MLP actor-critic; decoder/reward/continue are shared"
            if args.prediction_head_profile == SHARED_DISTILLED_HEADS_PROFILE
            else "per-task projector, Q/F/P atoms, heads, actor-critic"))
        ),
        "mechanism_capacity": _mechanism_capacity_manifest(
            task_count=task_count,
            mechanism_profile=config["task_mechanism_capacity_profile"],
            mechanism_parameterization=config["task_mechanism_parameterization"],
        ),
        "capacity_control_profile": DEFAULT_MECHANISM_PROFILE,
        "capacity_ablation_only": (
            not args.adaptive_qfp_compression
            and (
                config["task_mechanism_capacity_profile"]
                != DEFAULT_MECHANISM_PROFILE
                or config["task_mechanism_parameterization"]
                != DENSE_PRIVATE_PARAMETERIZATION
            )
        ),
        "prediction_head_topology": (
            {
                "ownership": "one shared plastic decoder/reward/continue set",
                "old_task_supervision": "real LTDM Dreamer loss plus boundary-teacher distillation",
                "distillation_scale": config[
                    "shared_prediction_distill_scale"
                ],
                "component_gradient_projection": True,
                "boundary_consolidation_and_rollback": True,
                "extra_teacher_forward": False,
                "extra_optimizer_updates": 0,
            }
            if args.prediction_head_profile == SHARED_DISTILLED_HEADS_PROFILE
            else {
                "ownership": "one frozen private decoder/reward/continue set per task",
                "distillation_scale": 0.0,
            }
        ),
        "behavior_topology": (({
                "actor_critic": "one independent MLP pair per task",
                "stable_targets": False,
                "current_old_update_split": [1.0, 0.0],
                "extra_optimizer_updates": 0,
            })),
        "gradient_rule": "per-component conflicting-current-direction projection",
        "interface_distillation": {
            "posterior_kl": config["interface_q_scale"],
            "layer_normalized_hidden_mse": config["interface_h_scale"],
            "frozen_old_actor_kl": config["interface_actor_scale"],
            "shared_prediction_outputs": config[
                "shared_prediction_distill_scale"
            ],
            "adaptive_qfp_outputs": config.get(
                "adaptive_compression_qfp_distill_scale", 0.0
            ),
        },
        "adaptive_compression_protocol": (
            {
                "acquisition": "full Dense Q/F/P 512/512/256 per task",
                "structured_pruning": "per-atom channel removal with physical compact modules",
                "candidate_width_fractions": config[
                    "adaptive_compression_width_fractions"
                ],
                "optimizer_steps_per_candidate": config[
                    "adaptive_compression_steps_per_candidate"
                ],
                "learning_rate": config["adaptive_compression_lr"],
                "validation_rollouts": config[
                    "adaptive_compression_rollouts"
                ],
                "maximum_relative_raw_return_drop": config[
                    "adaptive_compression_max_return_drop"
                ],
                "selection": "smallest passing candidate after evaluating all candidates",
                "validation_scope": (
                    "every seen task under automatic routing"
                    if args.behavior_profile in AUTOROUTE_BEHAVIORS
                    else "completed task with oracle routing"
                ),
                "fallback": "retain full Dense Q/F/P when no candidate passes",
                "candidate_replay": "completed-task LTDM only",
                "selection_cohort": "dedicated fixed pruning validation",
                "final_heldout_cohort_used_for_selection": False,
                "actor_critic_compression": False,
            }
            if args.adaptive_qfp_compression
            else None
        ),
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
        "final_evaluation.json",
        "model_parameter_accounting.json",
        "actor_critic_parameter_accounting.json",
    ]
    required.append("save_ac_bank.pt")
    if args.adaptive_qfp_compression:
        required.extend(
            f"adaptive_qfp_compression/task_{task_id:02d}_boundary.json"
            for task_id in range(task_count)
        )
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
