#!/usr/bin/env python3
"""Launch task-aware MoE-ARROW with fixed ARROW-50 training budgets."""

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
    CURRICULUM_DIRS,
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
from run_karrow_ar50_atari import (
    DINOV3_CACHE_DTYPE,
    DINOV3_DEPENDENCIES,
    DINOV3_FEATURE_LOSS_SCALE,
    DINOV3_FEATURE_STD_FLOOR,
    DINOV3_INPUT_SIZE,
    DINOV3_MAX_BATCH_SIZE,
    DINOV3_MODEL_ID,
    DINOV3_PATCH_FEATURE_DIM,
    DINOV3_PATCH_POOL_SIZE,
    _dinov3_dependency_versions,
    _feature_cache_budget,
    _model_artifact_manifest,
)


METHOD = "MoE-ARROW-50"
CODE_ID = "moe_arrow"
PROTOCOL = "MoE-ARROW-v1-Atari-TaskAware"
CURRENT_TASK_FRACTION = 0.5
PATCH_PROJECTION = "fixed_orthogonal"
PATCH_PROJECTION_SEED = 0
FEATURE_DIM = DINOV3_PATCH_FEATURE_DIM * DINOV3_PATCH_POOL_SIZE**2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch task-aware MoE-ARROW with one expert and actor per game"
    )
    parser.add_argument("--seed", type=int, choices=range(5), default=0)
    parser.add_argument("--curriculum", choices=CURRICULUM_DIRS, default="original")
    parser.add_argument(
        "--dinov3-model-path",
        type=Path,
        default=(
            Path(os.environ["DINOV3_MODEL_PATH"])
            if "DINOV3_MODEL_PATH" in os.environ
            else None
        ),
        help="Absolute local DINOv3 ViT-S/16 directory; online loading is disabled",
    )
    parser.add_argument(
        "--task-prefix-length",
        type=int,
        choices=[1, 2, 3],
        help="Run a task-prefix pilot without changing per-task duration",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--swanlab-project")
    parser.add_argument("--swanlab-experiment-name")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _resolved_config(source: dict, *, model_path: Path, epochs: int) -> dict:
    config = json.loads(json.dumps(source))
    config.update(
        {
            "epochs": epochs,
            "continual_method": CODE_ID,
            "rssm_num_experts": len(config["esc"]["env_configs"]),
            "moe_arrow_current_task_fraction": CURRENT_TASK_FRACTION,
            "observation_objective": "dinov3_next_feature",
            "observation_encoder": "dinov3_vits16",
            "dinov3_model_path": str(model_path),
            "dinov3_input_size": DINOV3_INPUT_SIZE,
            "dinov3_max_batch_size": DINOV3_MAX_BATCH_SIZE,
            "dinov3_feature_cache_dtype": DINOV3_CACHE_DTYPE,
            "dinov3_feature_loss_scale": DINOV3_FEATURE_LOSS_SCALE,
            "dinov3_feature_mode": "patch_grid",
            "dinov3_patch_pool_size": DINOV3_PATCH_POOL_SIZE,
            "dinov3_patch_feature_dim": DINOV3_PATCH_FEATURE_DIM,
            "dinov3_patch_projection": PATCH_PROJECTION,
            "dinov3_patch_projection_frames": 0,
            "dinov3_patch_projection_seed": PATCH_PROJECTION_SEED,
            "dinov3_feature_loss_kind": "cosine",
            "dinov3_feature_std_floor": DINOV3_FEATURE_STD_FLOOR,
            "actor_network": "mlp",
            "fresh_ac": False,
            "residual_correction": "none",
            "residual_consolidation": "none",
            "shared_core_mode": "trainable",
        }
    )
    for replay_config in config["replay_buffers"]:
        replay_config["rb_device"] = "cpu"
    return config


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.dinov3_model_path is None:
        parser.error("--dinov3-model-path or DINOV3_MODEL_PATH is required")
    if args.cpu_threads is not None and args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if args.swanlab_experiment_name and not args.swanlab_project:
        parser.error("--swanlab-experiment-name requires --swanlab-project")

    project_git = (
        git_state(ROOT) if args.dry_run else require_synced_training_git_state(ROOT)
    )
    python = args.python.resolve()
    model_path = args.dinov3_model_path.expanduser().resolve()
    model_artifact = _model_artifact_manifest(model_path)
    source_config_path = _config_path(args.curriculum, args.seed)
    source_config = _verify_primary_config(
        source_config_path, args.curriculum, args.seed
    )
    swap_sched = int(source_config["esc"]["kwargs"]["swap_sched"])
    training_epochs = (
        int(source_config["epochs"])
        if args.task_prefix_length is None
        else swap_sched * args.task_prefix_length
    )
    config = _resolved_config(
        source_config,
        model_path=model_path,
        epochs=training_epochs,
    )
    allocated_experts = int(config["rssm_num_experts"])

    method = METHOD
    role = "primary-task-aware-continual-method"
    output_prefix = "moe_arrow_ar50"
    if args.task_prefix_length is not None:
        method += f"-T{args.task_prefix_length}Pilot"
        role += "-pilot"
        output_prefix += f"_t{args.task_prefix_length}_pilot"
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else ROOT / "runs" / f"{output_prefix}_{args.curriculum}_s{args.seed}"
    )
    config_path = output_dir / "resolved_training_config.json"

    env = os.environ.copy()
    recorded_env: dict[str, str] = {}
    if args.cpu_threads is not None:
        recorded_env = {key: str(args.cpu_threads) for key in THREAD_ENV_KEYS}
        env.update(recorded_env)
    triton_libcuda_path = env.get("TRITON_LIBCUDA_PATH")
    if triton_libcuda_path:
        resolved = Path(triton_libcuda_path).expanduser().resolve()
        if not (resolved / "libcuda.so").exists():
            raise FileNotFoundError(
                f"TRITON_LIBCUDA_PATH must contain libcuda.so: {resolved}"
            )
        env["TRITON_LIBCUDA_PATH"] = str(resolved)
        recorded_env["TRITON_LIBCUDA_PATH"] = str(resolved)
    project_pythonpath = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
    inherited_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (project_pythonpath, inherited_pythonpath) if value
    )
    dependency_versions = _dinov3_dependency_versions(python, env)

    command = [
        str(python),
        "Code/ARROW_and_DV3/Atari/train.py",
        "--config",
        str(config_path),
        "--arrow-replay-ratio",
        "50-50",
        "--log-dir",
        str(output_dir),
        "--fused-adam",
        "--tf32",
        "--evaluate-final",
    ]
    if args.profile_stages:
        command.append("--profile-stages")
    if args.swanlab_project:
        command.extend(("--swanlab-project", args.swanlab_project))
    if args.swanlab_experiment_name:
        command.extend(("--swanlab-experiment-name", args.swanlab_experiment_name))

    decisions_per_epoch = source_config["n_sync"] * source_config["gen_seq_len"]
    collection_epoch_equivalents = training_epochs
    if source_config.get("pretrain_enabled", True):
        collection_epoch_equivalents += (
            source_config.get("pretrain_data_multiplier", 4) - 1
        )
    agent_decisions = decisions_per_epoch * collection_epoch_equivalents
    raw_environment_frames = agent_decisions * source_config["env_repeat"]
    tasks = source_config["esc"]["env_configs"]
    visited_tasks = (
        tasks
        if args.task_prefix_length is None
        else tasks[: args.task_prefix_length]
    )
    feature_cache = _feature_cache_budget(
        config,
        dtype=DINOV3_CACHE_DTYPE,
        feature_dim=FEATURE_DIM,
    )
    task_id_storage_bytes = 2 * source_config["data_n_max"] * 8

    launch = {
        "method": method,
        "code_id": CODE_ID,
        "role": role,
        "protocol": PROTOCOL,
        "started_at_utc": None,
        "project_git": project_git,
        "upstream_arrow_commit": UPSTREAM_COMMIT,
        "source": str(ARROW_ROOT),
        "source_config": str(source_config_path),
        "resolved_training_config": str(config_path),
        "output_dir": str(output_dir),
        "curriculum": args.curriculum,
        "seed_id": args.seed,
        "seed": SEEDS[args.seed],
        "task_identity": {
            "exposed_to_agent": True,
            "source": "sequential scheduler",
            "uses": [
                "RSSM/head expert routing",
                "task-filtered replay sampling",
                "actor-critic bank selection",
            ],
            "not_concatenated_to_latent": True,
            "comparison_class": "task-aware upper-bound method",
        },
        "training_scope": {
            "task_prefix_length": args.task_prefix_length,
            "epochs": training_epochs,
            "task_duration_epochs": swap_sched,
            "tasks": [task["name"] for task in visited_tasks],
            "agent_decisions": agent_decisions,
            "raw_environment_frames": raw_environment_frames,
            "world_model_updates": training_epochs
            * source_config["steps_per_batch"],
            "actor_critic_updates": training_epochs
            * source_config["ac_train_steps"],
        },
        "world_model": {
            "router": "hard_task_id",
            "routing_granularity": "one homogeneous-task minibatch",
            "allocated_experts": allocated_experts,
            "expert_modules": [
                "recurrent_dynamics",
                "latent_prior",
                "reward_head",
                "continue_head",
            ],
            "shared_modules": [
                "frozen DINOv3 encoder",
                "posterior representation",
                "feature predictor",
            ],
            "new_task_initialization": "copy previous task expert once",
            "pixel_decoder": False,
        },
        "actor_critic": {
            "topology": "per_task_bank",
            "network": "DreamerV3 MLP actor and critic",
            "optimizer_state_shared": False,
            "new_task_initialization": "copy previous actor-critic weights; fresh optimizer",
            "current_task_update_fraction": CURRENT_TASK_FRACTION,
            "old_task_allocation": "uniform across replay-available old tasks",
            "total_updates_unchanged": True,
        },
        "observation": {
            "encoder": "frozen DINOv3 ViT-S/16",
            "model_id": DINOV3_MODEL_ID,
            "model_artifact": model_artifact,
            "input_size": DINOV3_INPUT_SIZE,
            "feature_mode": "patch_grid",
            "patch_pool_size": DINOV3_PATCH_POOL_SIZE,
            "patch_feature_dim": DINOV3_PATCH_FEATURE_DIM,
            "patch_projection": PATCH_PROJECTION,
            "patch_projection_seed": PATCH_PROJECTION_SEED,
            "patch_projection_frames": 0,
            "feature_dim": FEATURE_DIM,
            "objective": "one-step prior prediction of stopped spatial features",
            "feature_loss": "cosine",
            "pixel_decoder": False,
            "task1_fitted_visual_projection": False,
        },
        "replay": {
            "capacity_and_sampling": "ARROW-50 base allocation unchanged",
            "storage_device": "cpu",
            "base_storage": _arrow_replay_storage_budget(config),
            "feature_cache": feature_cache,
            "task_id_storage_bytes": task_id_storage_bytes,
            "task_sampling": (
                "fixed current/old update allocation, then conditional uniform "
                "sequence sampling inside the selected task"
            ),
            "subbuffer_selection": (
                "ARROW-50 weights renormalized only if a subbuffer lacks the selected task"
            ),
        },
        "residual_correction": "none",
        "extra_gradient_updates": 0,
        "extra_environment_interactions": 0,
        "evaluation": {
            "policy": "deterministic_argmax_and_latent_mode",
            "all_configured_tasks_at_periodic_checkpoints": True,
            "evaluation_data_enters_replay": False,
        },
        "checkpointing": {
            "final_world_model_and_actor_bank": True,
            "resumable": False,
            "reason": "replay and optimizer states are not serialized by vendored ARROW",
        },
        "determinism": {
            "python_numpy_torch_environment_and_replay_seeded": True,
            "task_update_scheduler_rng": "owned NumPy Generator",
            "actor_construction_preserves_training_rng": True,
            "torch_deterministic_algorithms": False,
            "tf32_enabled": True,
            "known_nondeterminism": [
                "CUDA kernels are not forced into deterministic-only mode"
            ],
        },
        "runtime_dependencies": dependency_versions,
        "cpu_threads": args.cpu_threads,
        "environment": recorded_env,
        "project_pythonpath_prepend": project_pythonpath,
        "command": command,
    }
    print(json.dumps(launch, indent=2))
    rendered_env = [f"{key}={value}" for key, value in recorded_env.items()]
    rendered_env.append(f"PYTHONPATH={env['PYTHONPATH']}")
    print(f"command: {shlex.join([*rendered_env, *command])}")
    if args.dry_run:
        return 0

    if dependency_versions != DINOV3_DEPENDENCIES:
        raise RuntimeError(
            "MoE-ARROW requires pinned DINOv3 dependencies: "
            f"expected={DINOV3_DEPENDENCIES} observed={dependency_versions}"
        )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {output_dir}")
    runtime_environment = _runtime_info(python, env)
    runtime_environment["packages"].update(dependency_versions)
    output_dir.mkdir(parents=True)
    _write_json(config_path, config)
    launch["started_at_utc"] = datetime.now(timezone.utc).isoformat()
    launch["runtime_environment"] = runtime_environment
    _write_json(output_dir / "launch.json", launch)

    return_code = _run_and_tee(
        command,
        cwd=ARROW_ROOT,
        env=env,
        log_path=output_dir / "train.log",
    )
    _write_json(
        output_dir / "run_status.json",
        {
            "complete": return_code == 0,
            "return_code": return_code,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
