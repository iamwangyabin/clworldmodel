#!/usr/bin/env python3
"""Launch method C from the exact post-Task-0 learned-base boundary."""

from __future__ import annotations

import argparse
import hashlib
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
    _config_path,
    _verify_primary_config,
)
from run_cnn_projector_lora_incremental import _prepare_replay_symlink
from run_evolving_atomic_rssm import (
    ARROW_WORLD_MODEL_PARAMETERS,
    DEFAULT_MECHANISM_PROFILE,
    MLP_ACTOR_PARAMETERS,
    MLP_CRITIC_PARAMETERS,
    TASK_ORDERS,
    TASK_PRIVATE_HEAD_ADDITION_PARAMETERS,
    TASK_PROJECTOR_PARAMETERS,
    _budget_manifest,
    _residual_mechanism_parameters,
    _storage_preflight,
    _training_command,
    _resolved_config as _dense_resolved_config,
)
from run_evolving_learned_base_adapters import _low_rank_mechanism_parameters
from summarize_continual_metrics import build_run_report


METHOD_KEY = "evolving_atomic_rssm_atomic_lora_shared_heads_arrow"
METHOD_NAME = (
    "Evolving-Core Task-0 Dense Base + Rank-128 Atomic Q/F/P Residuals + "
    "Shared Distilled Prediction Heads + Private MLP Actor-Critic"
)
PROTOCOL = (
    "Evolving-Core-Task0BoundaryBootstrap-AtomicRank128QFP-"
    "SharedDistilledHeads-PrivateMLPAC-ARROW-v1-OriginalSix-Atari-"
    "TaskAware-Pilot"
)
TASK_ORDER = "arrow-original-six"
MECHANISM_PARAMETERIZATION = "dense_task0_low_rank_atoms"
MECHANISM_LOW_RANK = 128
NUM_ATOMS = 4
PREDICTION_HEAD_COUNT = 3
INHERITED_TASK0_EPOCHS = 90


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=range(len(SEEDS)), default=0)
    parser.add_argument("--classification", choices=("pilot",), default="pilot")
    parser.add_argument("--task0-checkpoint", type=Path, required=True)
    parser.add_argument("--task0-boundary-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--replay-mmap-root", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cpu-threads", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_artifact(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if not resolved.is_file() or not sidecar.is_file():
        raise FileNotFoundError(
            f"Required bootstrap artifact/checksum is missing: {resolved}"
        )
    fields = sidecar.read_text(encoding="ascii").split()
    digest = _sha256(resolved)
    if not fields or fields[0] != digest:
        raise ValueError(f"Bootstrap artifact checksum mismatch: {resolved}")
    return {
        "path": str(resolved),
        "sha256": digest,
        "bytes": resolved.stat().st_size,
        "checksum_sidecar": str(sidecar),
    }


def _resolved_config(source: dict) -> dict:
    """Compose A's plastic shared-head topology with rank-128 later Q/F/P."""

    config = _dense_resolved_config(
        source,
        task_order=TASK_ORDER,
        task0_profile="fixed_v1",
    )
    config.update(
        {
            "continual_method": METHOD_KEY,
            "task_mechanism_reuse": True,
            "task_mechanism_parameterization": MECHANISM_PARAMETERIZATION,
            "task_mechanism_low_rank": MECHANISM_LOW_RANK,
            "task_private_heads": False,
            "task_shared_prediction_heads": True,
            "task_private_prediction_adapters": False,
            "prediction_adapter_rank": 0,
            "freeze_shared_prediction_heads_after_task0": False,
            "shared_prediction_distill_scale": 0.1,
        }
    )
    return config


def _mechanism_capacity_manifest(task_count: int) -> dict[str, object]:
    if task_count < 2:
        raise ValueError("Method C requires a multi-task schedule")
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
    route_parameters = PREDICTION_HEAD_COUNT * NUM_ATOMS * sum(range(task_count))
    return {
        "profile": DEFAULT_MECHANISM_PROFILE,
        "parameterization": MECHANISM_PARAMETERIZATION,
        "rank": MECHANISM_LOW_RANK,
        "atoms": NUM_ATOMS,
        "rank_per_atom": MECHANISM_LOW_RANK // NUM_ATOMS,
        "task0_dense": {**task0, "total": task0_total},
        "per_later_task_atomic_low_rank": {**later, "total": later_total},
        "task0_dense_parameters": task0_total,
        "later_task_delta_parameters": (task_count - 1) * later_total,
        "old_atom_reuse_enabled": True,
        "route_parameters": route_parameters,
        "mechanism_and_route_parameters": (
            task0_total + (task_count - 1) * later_total + route_parameters
        ),
    }


def _parameter_manifest(config: dict) -> dict[str, object]:
    task_count = len(config["esc"]["env_configs"])
    mechanisms = _mechanism_capacity_manifest(task_count)
    dense_task0 = int(mechanisms["task0_dense_parameters"])
    low_rank_later = int(
        mechanisms["per_later_task_atomic_low_rank"]["total"]
    )
    routes = int(mechanisms["route_parameters"])
    world_model_parameters = (
        ARROW_WORLD_MODEL_PARAMETERS
        + task_count * TASK_PROJECTOR_PARAMETERS
        + dense_task0
        + (task_count - 1) * low_rank_later
        + routes
    )
    mlp_pair = MLP_ACTOR_PARAMETERS + MLP_CRITIC_PARAMETERS
    behavior_parameters = task_count * mlp_pair
    online_parameters = world_model_parameters + behavior_parameters
    dense_shared_heads_online = (
        ARROW_WORLD_MODEL_PARAMETERS
        + task_count * TASK_PROJECTOR_PARAMETERS
        + sum(dense_task0 + PREDICTION_HEAD_COUNT * NUM_ATOMS * task_id for task_id in range(task_count))
        + behavior_parameters
    )
    learned_base_rank32_online = 37_156_095
    arrow_online = ARROW_WORLD_MODEL_PARAMETERS + mlp_pair
    per_task = {
        str(task_id): (
            TASK_PROJECTOR_PARAMETERS
            + (dense_task0 if task_id == 0 else low_rank_later)
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
        "task0_dense_mechanism_parameters": dense_task0,
        "low_rank_mechanism_parameters_per_later_task": low_rank_later,
        "route_parameters": routes,
        "prediction_head_topology": "single_shared_plastic",
        "prediction_adapter_parameters": 0,
        "per_task_world_model_additions": per_task,
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
        "comparison_to_dense_shared_heads_private_mlp": {
            "reference_parameters": dense_shared_heads_online,
            "difference": online_parameters - dense_shared_heads_online,
            "relative_difference": online_parameters / dense_shared_heads_online - 1.0,
        },
        "comparison_to_failed_learned_base_rank32_pilot": {
            "reference_parameters": learned_base_rank32_online,
            "difference": online_parameters - learned_base_rank32_online,
            "relative_difference": online_parameters / learned_base_rank32_online - 1.0,
        },
        "comparison_to_arrow_50": {
            "reference_parameters": arrow_online,
            "difference": online_parameters - arrow_online,
            "relative_difference": online_parameters / arrow_online - 1.0,
        },
    }


def _bootstrap_provenance(
    checkpoint: Path, boundary_snapshot: Path
) -> dict[str, object]:
    checkpoint_record = _verified_artifact(checkpoint)
    snapshot_record = _verified_artifact(boundary_snapshot)
    checkpoint_path = Path(str(checkpoint_record["path"]))
    source_run = checkpoint_path.parent.parent
    expected_snapshot_dir = source_run / "task_boundary_snapshots"
    if Path(str(snapshot_record["path"])).parent != expected_snapshot_dir:
        raise ValueError("Task-0 checkpoint and boundary snapshot come from different runs")
    launch_path = source_run / "launch.json"
    if not launch_path.is_file():
        raise FileNotFoundError(f"Source Task-0 launch manifest is missing: {launch_path}")
    source_launch = json.loads(launch_path.read_text(encoding="utf-8"))
    if source_launch.get("project_git", {}).get("commit") != (
        "6fef9bdde01b77110606d50b3fa7f9449aae60ac"
    ):
        raise ValueError("Task-0 source commit is not the predeclared learned-base pilot")
    return {
        "source_run": str(source_run),
        "source_project_git": source_launch["project_git"],
        "source_method": source_launch.get("method"),
        "checkpoint": checkpoint_record,
        "boundary_inference_snapshot": snapshot_record,
        "source_launch": str(launch_path),
        "source_launch_sha256": _sha256(launch_path),
        "future_task_data_in_source": False,
        "task0_replay_restored": True,
        "equivalent_resume": False,
        "reason_not_equivalent": "future-task module and prediction-head ownership changes",
    }


def _materialize_task0_boundary_snapshot(
    source: Path, target_directory: Path
) -> dict[str, object]:
    """Copy the immutable Task-0 boundary into C's self-contained run record."""

    source_record = _verified_artifact(source)
    resolved_source = Path(str(source_record["path"]))
    if "boundary_01_task_00_completed_0090" not in resolved_source.stem:
        raise ValueError(
            "Method C requires the exact Task-0 / 90-epoch boundary snapshot"
        )
    target_directory.mkdir(parents=True, exist_ok=False)
    target = target_directory / resolved_source.name
    target_sidecar = target.with_suffix(target.suffix + ".sha256")
    shutil.copy2(resolved_source, target)
    shutil.copy2(
        resolved_source.with_suffix(resolved_source.suffix + ".sha256"),
        target_sidecar,
    )
    target_record = _verified_artifact(target)
    if target_record["sha256"] != source_record["sha256"]:
        raise RuntimeError("Materialized Task-0 boundary snapshot changed content")
    return target_record


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
    checkpoint = args.task0_checkpoint.expanduser().resolve()
    boundary_snapshot = args.task0_boundary_snapshot.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else ROOT
        / "runs"
        / f"evolving_atomic_lora128_shared_heads_original_six_s{args.seed}_task0bootstrap_pilot"
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
    command.extend(
        ["--init-evolving-task0-transition-checkpoint", str(checkpoint)]
    )
    env = os.environ.copy()
    thread_env = {key: str(args.cpu_threads) for key in THREAD_ENV_KEYS}
    env.update(thread_env)
    project_pythonpath = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (project_pythonpath, env.get("PYTHONPATH")) if value
    )
    full_budget = _budget_manifest(config)
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
        "from_scratch": False,
        "task0_boundary_bootstrap": True,
        "bootstrap_provenance": (
            None
            if args.dry_run
            else _bootstrap_provenance(checkpoint, boundary_snapshot)
        ),
        "shared_core": (
            "A-style plastic CNN/base RSSM/decoder/reward/continue protected by "
            "LTDM, interface distillation, component projection, and boundary rollback"
        ),
        "private_state": (
            "Task-0 dense Q/F/P; Tasks 1-5 independent rank-128 nonlinear Q/F/P "
            "residual atoms plus routes; projector and MLP Actor-Critic per task"
        ),
        "mechanism_capacity": _mechanism_capacity_manifest(task_count),
        "prediction_head_topology": {
            "ownership": "one shared plastic decoder/reward/continue set",
            "old_task_supervision": "real LTDM Dreamer loss plus boundary-teacher distillation",
            "distillation_scale": config["shared_prediction_distill_scale"],
            "private_prediction_adapters": False,
            "extra_optimizer_updates": 0,
        },
        "behavior_topology": {
            "actor_critic": "one independent MLP pair per task",
            "current_old_update_split": [1.0, 0.0],
            "extra_optimizer_updates": 0,
        },
        "budgets": full_budget,
        "execution_budget": {
            "inherited_completed_epochs": INHERITED_TASK0_EPOCHS,
            "newly_executed_epochs": config["epochs"] - INHERITED_TASK0_EPOCHS,
            "combined_protocol_epochs": config["epochs"],
            "combined_environment_and_update_budget_matches_full_pilot": True,
            "world_model_optimizer_reset_at_transition": True,
        },
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
    launch["materialized_task0_boundary_snapshot"] = (
        _materialize_task0_boundary_snapshot(boundary_snapshot, task_snapshot_dir)
    )
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
        "task0_transition_initialization.json",
        f"evolving_core_checkpoints/task_{task_count - 1:02d}_pre_consolidation.pt",
        f"evolving_core_checkpoints/task_{task_count - 1:02d}_post_consolidation.pt",
    ]
    missing = [name for name in required if not (output_dir / name).is_file()]
    missing_consolidation_records = []
    for task_id in range(1, task_count):
        success = output_dir / "evolving_core_consolidation" / f"task_{task_id:02d}_boundary.json"
        failure = output_dir / "evolving_core_checkpoints" / f"task_{task_id:02d}_consolidation_failure.json"
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
        "complete": (
            return_code == 0
            and not missing
            and not missing_consolidation_records
            and metric_report_error is None
        ),
        "return_code": return_code,
        "combined_completed_epochs": config["epochs"] if return_code == 0 else None,
        "inherited_task0_epochs": INHERITED_TASK0_EPOCHS,
        "newly_executed_epochs": config["epochs"] - INHERITED_TASK0_EPOCHS,
        "boundary_count": 6 if return_code == 0 else None,
        "boundary_sources": {
            "task0": str(
                launch["materialized_task0_boundary_snapshot"]["path"]
            ),
            "tasks1_to_5": str(task_snapshot_dir),
        },
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
