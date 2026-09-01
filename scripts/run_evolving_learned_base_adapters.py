#!/usr/bin/env python3
"""Launch the fixed six-task learned-base low-rank Evolving-Core pilot."""

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
    _config_path,
    _verify_primary_config,
)
from run_cnn_projector_lora_incremental import _prepare_replay_symlink
from run_evolving_atomic_rssm import (
    ARROW_WORLD_MODEL_PARAMETERS,
    DEFAULT_MECHANISM_PROFILE,
    MLP_ACTOR_PARAMETERS,
    MLP_CRITIC_PARAMETERS,
    TASK_DURATION_EPOCHS,
    TASK_MECHANISM_PARAMETERS,
    TASK_ORDERS,
    TASK_PRIVATE_HEAD_ADDITION_PARAMETERS,
    TASK_PROJECTOR_PARAMETERS,
    _budget_manifest,
    _residual_mechanism_parameters,
    _storage_preflight,
    _training_command,
    _resolved_config as _dense_resolved_config,
)
from summarize_continual_metrics import build_run_report


METHOD_KEY = "evolving_atomic_rssm_learned_base_adapters_arrow"
METHOD_NAME = (
    "Evolving-Core Learned Task-0 Base + Rank-32 Private Q/F/P Residuals + "
    "Private Prediction Feature Adapters + Private MLP Actor-Critic"
)
PROTOCOL = (
    "Evolving-Core-LearnedTask0Base-LowRank32QFP-PrivatePredictionAdapters-"
    "PrivateMLPAC-ARROW-v1-OriginalSix-Atari-TaskAware-Pilot"
)
TASK_ORDER = "arrow-original-six"
MECHANISM_PARAMETERIZATION = "learned_task0_low_rank"
MECHANISM_LOW_RANK = 32
PREDICTION_ADAPTER_RANK = 32
PREDICTION_ADAPTER_RESIDUAL_SCALE = 0.1
MODEL_STATE_FEATURES = 32 * 32 + 512
PREDICTION_HEAD_COUNT = 3
NUM_ATOMS = 4


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=range(len(SEEDS)), default=0)
    parser.add_argument(
        "--classification", choices=("pilot",), default="pilot"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--replay-mmap-root", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cpu-threads", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _resolved_config(source: dict) -> dict:
    """Compose the independently named fixed-rank v1 pilot."""

    config = _dense_resolved_config(
        source,
        task_order=TASK_ORDER,
        task0_profile="fixed_v1",
    )
    config.update(
        {
            "continual_method": METHOD_KEY,
            "task_mechanism_reuse": False,
            "task_mechanism_parameterization": MECHANISM_PARAMETERIZATION,
            "task_mechanism_low_rank": MECHANISM_LOW_RANK,
            "task_private_heads": False,
            "task_shared_prediction_heads": True,
            "task_private_prediction_adapters": True,
            "prediction_adapter_rank": PREDICTION_ADAPTER_RANK,
            "prediction_adapter_residual_scale": (
                PREDICTION_ADAPTER_RESIDUAL_SCALE
            ),
            "freeze_shared_prediction_heads_after_task0": True,
            "shared_prediction_distill_scale": 0.1,
        }
    )
    return config


def _low_rank_mechanism_parameters(
    *, in_features: int, out_features: int, hidden_features: int, rank: int
) -> int:
    """LayerNorm plus in-rank-hidden-rank-out private residual parameters."""

    if min(in_features, out_features, hidden_features, rank) < 1:
        raise ValueError("Low-rank mechanism dimensions must be positive")
    return (
        2 * in_features
        + in_features * rank
        + rank * hidden_features
        + hidden_features
        + hidden_features * rank
        + rank * out_features
        + out_features
    )


def _prediction_feature_adapter_parameters(*, in_features: int, rank: int) -> int:
    """LayerNorm plus zero-effect in-rank-in residual parameters."""

    if min(in_features, rank) < 1:
        raise ValueError("Prediction-adapter dimensions must be positive")
    return 2 * in_features + in_features * rank + rank * in_features + in_features


def _mechanism_capacity_manifest(task_count: int) -> dict[str, object]:
    if task_count < 2:
        raise ValueError("The learned-base method requires multiple tasks")
    interfaces = {
        "recurrent": (512, 512, 512),
        "representation_posterior": (4096 + 512, 32 * 32, 512),
        "transition_prior": (512, 32 * 32, 256),
    }
    task0 = {
        name: _residual_mechanism_parameters(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
        )
        for name, (in_features, out_features, hidden_features) in interfaces.items()
    }
    later = {
        name: _low_rank_mechanism_parameters(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            rank=MECHANISM_LOW_RANK,
        )
        for name, (in_features, out_features, hidden_features) in interfaces.items()
    }
    task0_total = sum(task0.values())
    later_total = sum(later.values())
    dormant_route_parameters = PREDICTION_HEAD_COUNT * NUM_ATOMS * sum(
        range(task_count)
    )
    return {
        "profile": DEFAULT_MECHANISM_PROFILE,
        "parameterization": MECHANISM_PARAMETERIZATION,
        "rank": MECHANISM_LOW_RANK,
        "atoms": NUM_ATOMS,
        "task0_learned_dense_base": {**task0, "total": task0_total},
        "per_later_task_low_rank_delta": {**later, "total": later_total},
        "task0_base_parameters": task0_total,
        "later_task_delta_parameters": (task_count - 1) * later_total,
        "old_atom_reuse_enabled": False,
        "registered_dormant_route_parameters": dormant_route_parameters,
        "mechanism_and_route_parameters": (
            task0_total
            + (task_count - 1) * later_total
            + dormant_route_parameters
        ),
    }


def _parameter_manifest(config: dict) -> dict[str, object]:
    task_count = len(config["esc"]["env_configs"])
    mechanisms = _mechanism_capacity_manifest(task_count)
    low_rank_per_later = int(
        mechanisms["per_later_task_low_rank_delta"]["total"]
    )
    route_parameters = int(mechanisms["registered_dormant_route_parameters"])
    one_feature_adapter = _prediction_feature_adapter_parameters(
        in_features=MODEL_STATE_FEATURES,
        rank=PREDICTION_ADAPTER_RANK,
    )
    prediction_adapters_per_later = PREDICTION_HEAD_COUNT * one_feature_adapter
    prediction_adapter_parameters = (
        task_count - 1
    ) * prediction_adapters_per_later
    world_model_parameters = (
        ARROW_WORLD_MODEL_PARAMETERS
        + task_count * TASK_PROJECTOR_PARAMETERS
        + int(mechanisms["task0_base_parameters"])
        + int(mechanisms["later_task_delta_parameters"])
        + route_parameters
        + prediction_adapter_parameters
    )
    mlp_pair = MLP_ACTOR_PARAMETERS + MLP_CRITIC_PARAMETERS
    behavior_parameters = task_count * mlp_pair
    online_parameters = world_model_parameters + behavior_parameters
    dense_world_model = (
        ARROW_WORLD_MODEL_PARAMETERS
        + task_count * TASK_PROJECTOR_PARAMETERS
        + sum(TASK_MECHANISM_PARAMETERS + 12 * task_id for task_id in range(task_count))
        + (task_count - 1) * TASK_PRIVATE_HEAD_ADDITION_PARAMETERS
    )
    dense_online = dense_world_model + behavior_parameters
    shared_distilled_heads_world_model = (
        ARROW_WORLD_MODEL_PARAMETERS
        + task_count * TASK_PROJECTOR_PARAMETERS
        + sum(TASK_MECHANISM_PARAMETERS + 12 * task_id for task_id in range(task_count))
    )
    shared_distilled_heads_online = (
        shared_distilled_heads_world_model + behavior_parameters
    )
    arrow_online = (
        ARROW_WORLD_MODEL_PARAMETERS + MLP_ACTOR_PARAMETERS + MLP_CRITIC_PARAMETERS
    )
    per_task_world_model_additions = {
        str(task_id): (
            TASK_PROJECTOR_PARAMETERS
            + (
                int(mechanisms["task0_base_parameters"])
                if task_id == 0
                else low_rank_per_later + prediction_adapters_per_later
            )
            + PREDICTION_HEAD_COUNT * NUM_ATOMS * task_id
        )
        for task_id in range(task_count)
    }
    return {
        "schema_version": 1,
        "scope": (
            "online inference parameters; FP32 master weights; excludes optimizer "
            "state, gradients, activations, Replay, and boundary teachers"
        ),
        "world_model_parameters": world_model_parameters,
        "behavior_parameters": behavior_parameters,
        "online_parameters": online_parameters,
        "fp32_parameter_bytes": online_parameters * 4,
        "mechanism_parameterization": MECHANISM_PARAMETERIZATION,
        "mechanism_low_rank": MECHANISM_LOW_RANK,
        "task0_dense_mechanism_parameters": int(
            mechanisms["task0_base_parameters"]
        ),
        "low_rank_mechanism_parameters_per_later_task": low_rank_per_later,
        "prediction_head_topology": (
            "one frozen learned Task-0 decoder/reward/continue base plus three "
            "private rank-32 feature adapters per later task"
        ),
        "prediction_adapter_rank": PREDICTION_ADAPTER_RANK,
        "one_prediction_feature_adapter_parameters": one_feature_adapter,
        "prediction_adapter_parameters_per_later_task": (
            prediction_adapters_per_later
        ),
        "prediction_adapter_parameters": prediction_adapter_parameters,
        "registered_dormant_route_parameters": route_parameters,
        "per_task_world_model_additions": per_task_world_model_additions,
        "per_later_task_behavior_growth": mlp_pair,
        "runtime_verification_artifacts": [
            "model_parameter_accounting.json",
            "actor_critic_parameter_accounting.json",
        ],
        "training_only_prediction_head_teacher": {
            "parameters": TASK_PRIVATE_HEAD_ADDITION_PARAMETERS,
            "additional_teacher_copy": False,
            "contained_in_common_evolving_boundary_world_model_teacher": True,
            "growth_with_task_count": 0,
        },
        "comparison_to_dense_evolving_v2_private_mlp": {
            "reference_parameters": dense_online,
            "difference": online_parameters - dense_online,
            "relative_difference": online_parameters / dense_online - 1.0,
        },
        "comparison_to_shared_distilled_heads_private_mlp": {
            "reference_parameters": shared_distilled_heads_online,
            "difference": online_parameters - shared_distilled_heads_online,
            "relative_difference": (
                online_parameters / shared_distilled_heads_online - 1.0
            ),
        },
        "comparison_to_arrow_50": {
            "reference_parameters": arrow_online,
            "difference": online_parameters - arrow_online,
            "relative_difference": online_parameters / arrow_online - 1.0,
        },
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
    config = _resolved_config(source)
    task_count = len(TASK_ORDERS[TASK_ORDER])
    python = args.python.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else ROOT
        / "runs"
        / f"evolving_learned_base_lora32_head_adapters_original_six_s{args.seed}_pilot"
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
        "method": METHOD_NAME,
        "protocol": PROTOCOL,
        "classification": args.classification,
        "status": "dry_run" if args.dry_run else "launching",
        "project_git": project_git,
        "upstream_arrow_commit": UPSTREAM_COMMIT,
        "source_config": str(source_path),
        "seed_index": args.seed,
        "seed": SEEDS[args.seed],
        "task_order": list(TASK_ORDERS[TASK_ORDER]),
        "task_identity_exposed_to_agent": True,
        "task_agnostic_claimed": False,
        "from_scratch": True,
        "fixed_rank_not_adaptive": True,
        "shared_core": (
            "CNN and base posterior/recurrent/prior RSSM remain replay-protected "
            "and plastic; Task-0 full Q/F/P mechanisms become the frozen learned "
            "mechanism base after Task 0"
        ),
        "private_state": (
            "per-task projector; Rank-32 Q/F/P delta for Tasks 1-5; three "
            "Rank-32 prediction-feature adapters for Tasks 1-5; independent MLP "
            "Actor-Critic per task"
        ),
        "mechanism_capacity": _mechanism_capacity_manifest(task_count),
        "prediction_head_topology": {
            "base": "Task-0 decoder/reward/continue frozen after Task 0",
            "later_task_adaptation": (
                "independent zero-effect rank-32 input-feature adapter per head"
            ),
            "old_task_supervision": (
                "real LTDM Dreamer loss plus boundary-teacher output distillation"
            ),
            "distillation_scale": config["shared_prediction_distill_scale"],
            "extra_optimizer_updates": 0,
        },
        "behavior_topology": {
            "actor_critic": "one independent MLP pair per task",
            "current_old_update_split": [1.0, 0.0],
            "extra_optimizer_updates": 0,
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
        task_order=TASK_ORDER,
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
    for task_id in [task_count - 1]:
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
