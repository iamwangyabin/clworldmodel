#!/usr/bin/env python3
"""Launch the fixed-capacity KARROW Frozen-Core Atari protocol and controls."""

from __future__ import annotations

import argparse
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


DINOV3_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
DINOV3_TOKEN_FEATURES = 384
DINOV3_INPUT_SIZE = 256
DINOV3_MAX_BATCH_SIZE = 128
DINOV3_CACHE_DTYPE = "float16"
DINOV3_FEATURE_LOSS_SCALE = 1.0
DINOV3_PATCH_POOL_SIZE = 4
DINOV3_PATCH_FEATURE_DIM = 64
DINOV3_PATCH_PROJECTION_FRAMES = 512
DINOV3_FEATURE_STD_FLOOR = 0.05
DINOV3_DEPENDENCIES = {"transformers": "4.57.1", "safetensors": "0.6.2"}
RESIDUAL_BOTTLENECK_FEATURES = 64
RESIDUAL_GRID_SIZE = 8
RESIDUAL_INPUT_RANGE = (-2.0, 2.0)
RESIDUAL_RMS_NORM_EPSILON = 1e-4
RESIDUAL_ALPHA = 0.1
RESIDUAL_MODULE_COUNT = 8
RESIDUAL_CORE_PARAMETERS = 32_768
RESIDUAL_CONSOLIDATION_BATCHES = 16
RESIDUAL_CONSOLIDATION_IMAGINATION_HORIZON = 8
RESIDUAL_CONSOLIDATION_GRADIENT_POWER = 2.0
RESIDUAL_CONSOLIDATION_MIN_PLASTICITY = 0.01
RESIDUAL_CONSOLIDATION_ANCHOR_LOSS_SCALE = 1.0
PROTOCOLS = {
    "v1": {
        "protocol": "KARROW-FrozenCore-v1-Atari",
        "residual_input_mode": "base_output",
        "observation_objective": "dinov3_next_feature",
        "feature_mode": "cls",
        "patch_pool_size": DINOV3_PATCH_POOL_SIZE,
        "patch_feature_dim": DINOV3_TOKEN_FEATURES,
        "patch_projection": "none",
        "patch_projection_frames": 0,
        "feature_dim": DINOV3_TOKEN_FEATURES,
        "feature_loss_kind": "cosine",
        "objective_description": "one-step prior-state cosine feature prediction",
        "prediction_state": "one-step prior",
        "first_and_reset_steps_masked": True,
        "variants": {
            "dino": ("ARROW-DINO-50", "arrow_dino_ar50"),
            "mlp": (
                "ARROW-DINO-FrozenCore-MLPRes-50",
                "arrow_dino_frozen_core_mlp_residual_ar50",
            ),
            "kan": ("KARROW-FrozenCore-50", "karrow_frozen_core_ar50"),
        },
    },
    "v2": {
        "protocol": "KARROW-SpatialFrozenCore-v2-Atari",
        "residual_input_mode": "base_output",
        "observation_objective": "dinov3_posterior_feature",
        "feature_mode": "patch_grid",
        "patch_pool_size": DINOV3_PATCH_POOL_SIZE,
        "patch_feature_dim": DINOV3_PATCH_FEATURE_DIM,
        "patch_projection": "task1_pca",
        "patch_projection_frames": DINOV3_PATCH_PROJECTION_FRAMES,
        "feature_dim": (
            DINOV3_PATCH_FEATURE_DIM * DINOV3_PATCH_POOL_SIZE**2
        ),
        "feature_loss_kind": "batch_standardized_smooth_l1",
        "objective_description": (
            "posterior-state batch-standardized projected spatial feature reconstruction"
        ),
        "prediction_state": "posterior",
        "first_and_reset_steps_masked": False,
        "variants": {
            "dino": ("ARROW-DINOSpatial-50", "arrow_dino_spatial_ar50"),
            "mlp": (
                "ARROW-DINOSpatial-FrozenCore-MLPRes-50",
                "arrow_dino_spatial_frozen_core_mlp_residual_ar50",
            ),
            "kan": (
                "KARROW-SpatialFrozenCore-50",
                "karrow_spatial_frozen_core_ar50",
            ),
        },
    },
    "v3": {
        "protocol": "KARROW-ReplayConsolidated-v3-Atari",
        "residual_input_mode": "base_output",
        "observation_objective": "dinov3_posterior_feature",
        "feature_mode": "patch_grid",
        "patch_pool_size": DINOV3_PATCH_POOL_SIZE,
        "patch_feature_dim": DINOV3_PATCH_FEATURE_DIM,
        "patch_projection": "task1_pca",
        "patch_projection_frames": DINOV3_PATCH_PROJECTION_FRAMES,
        "feature_dim": (
            DINOV3_PATCH_FEATURE_DIM * DINOV3_PATCH_POOL_SIZE**2
        ),
        "feature_loss_kind": "batch_standardized_smooth_l1",
        "objective_description": (
            "posterior-state batch-standardized projected spatial feature reconstruction"
        ),
        "prediction_state": "posterior",
        "first_and_reset_steps_masked": False,
        "replay_functional_consolidation": True,
        "variants": {
            "dino": (
                "ARROW-DINOSpatial-v3Control-50",
                "arrow_dino_spatial_v3_control_ar50",
            ),
            "mlp": (
                "ARROW-DINOSpatial-FrozenCore-MLPRes-v3Control-50",
                "arrow_dino_spatial_frozen_core_mlp_v3_control_ar50",
            ),
            "kan": (
                "KARROW-ReplayConsolidated-50",
                "karrow_replay_consolidated_ar50",
            ),
        },
    },
    "v4": {
        "protocol": "KARROW-InputAligned-v4-Atari",
        "residual_input_mode": "module_input",
        "observation_objective": "dinov3_posterior_feature",
        "feature_mode": "patch_grid",
        "patch_pool_size": DINOV3_PATCH_POOL_SIZE,
        "patch_feature_dim": DINOV3_PATCH_FEATURE_DIM,
        "patch_projection": "task1_pca",
        "patch_projection_frames": DINOV3_PATCH_PROJECTION_FRAMES,
        "feature_dim": (
            DINOV3_PATCH_FEATURE_DIM * DINOV3_PATCH_POOL_SIZE**2
        ),
        "feature_loss_kind": "batch_standardized_smooth_l1",
        "objective_description": (
            "posterior-state batch-standardized projected spatial feature reconstruction"
        ),
        "prediction_state": "posterior",
        "first_and_reset_steps_masked": False,
        "variants": {
            "dino": (
                "ARROW-DINOSpatial-v4Control-50",
                "arrow_dino_spatial_v4_control_ar50",
            ),
            "mlp": (
                "ARROW-DINOSpatial-InputAligned-MLPRes-50",
                "arrow_dino_spatial_input_aligned_mlp_residual_ar50",
            ),
            "kan": (
                "KARROW-InputAligned-50",
                "karrow_input_aligned_ar50",
            ),
        },
    },
}
VARIANT_ROLES = {
    "dino": ("frozen-representation-control", "none"),
    "mlp": ("parameter-matched-frozen-core-control", "mlp"),
    "kan": ("primary-frozen-core-method", "kan"),
}


def _parser(*, default_visual_version: str = "v1") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch KARROW Frozen-Core or a matched frozen-DINO control"
    )
    parser.add_argument("--variant", choices=VARIANT_ROLES, default="kan")
    parser.add_argument(
        "--visual-version",
        choices=PROTOCOLS,
        default=default_visual_version,
        help=(
            "v1 preserves the CLS pilot; v2 uses spatial patch features; "
            "v3 adds replay-guided incremental KAN consolidation; v4 feeds "
            "each residual from its base module input"
        ),
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
        help=(
            "Absolute local Hugging Face model directory. The launcher never "
            "downloads weights during a run."
        ),
    )
    parser.add_argument(
        "--task-prefix-length",
        type=int,
        choices=[1, 2, 3],
        help="Run a one-, two-, or three-task pilot without changing task duration",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--swanlab-project")
    parser.add_argument("--swanlab-experiment-name")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_artifact_manifest(model_path: Path) -> dict[str, object]:
    if not model_path.is_absolute():
        raise ValueError("--dinov3-model-path must resolve to an absolute path")
    if not model_path.is_dir():
        raise FileNotFoundError(f"DINOv3 model directory does not exist: {model_path}")
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"DINOv3 config is missing: {config_path}")
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    if model_config.get("hidden_size") != DINOV3_TOKEN_FEATURES:
        raise ValueError(
            "KARROW Frozen-Core requires the 384-dimensional DINOv3 ViT-S model"
        )
    if model_config.get("patch_size") != 16:
        raise ValueError("KARROW Frozen-Core requires a DINOv3 ViT-S/16 model")

    files = []
    for path in sorted(model_path.rglob("*")):
        if not path.is_file() or ".git" in path.relative_to(model_path).parts:
            continue
        files.append(
            {
                "path": str(path.relative_to(model_path)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if not any(
        item["path"].endswith((".safetensors", ".bin")) for item in files
    ):
        raise FileNotFoundError("DINOv3 model directory contains no local weight file")
    return {
        "model_id": DINOV3_MODEL_ID,
        "local_path": str(model_path),
        "feature_dim": DINOV3_TOKEN_FEATURES,
        "token_features": DINOV3_TOKEN_FEATURES,
        "files": files,
        "total_bytes": sum(int(item["size_bytes"]) for item in files),
        "license": "DINOv3 License",
        "distributed_with_project": False,
    }


def _feature_cache_budget(
    config: dict,
    *,
    dtype: str,
    feature_dim: int,
) -> dict[str, object]:
    bytes_per_element = {"float16": 2, "float32": 4}[dtype]
    transitions_per_buffer = config["data_t"] * config["data_n_max"]
    bytes_per_buffer = (
        transitions_per_buffer * feature_dim * bytes_per_element
    )
    replay_devices = {
        item["rb_type"]: item["rb_device"] for item in config["replay_buffers"]
    }
    return {
        "dtype": dtype,
        "feature_dim": feature_dim,
        "bytes_per_element": bytes_per_element,
        "storage_bytes": 2 * bytes_per_buffer,
        "buffers": {
            "fifo": {
                "device": replay_devices["FifoReplay"],
                "storage_bytes": bytes_per_buffer,
            },
            "ltdm": {
                "device": replay_devices["LongTermReplay"],
                "storage_bytes": bytes_per_buffer,
            },
        },
        "retention_and_sampling": "aligned sidecar; ARROW decisions unchanged",
    }


def _dinov3_dependency_versions(
    python: Path, env: dict[str, str]
) -> dict[str, str | None]:
    probe_code = """
import json
from importlib import metadata

versions = {}
for name in ("transformers", "safetensors"):
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None
print(json.dumps(versions))
"""
    probe = subprocess.run(
        [
            str(python),
            "-c",
            probe_code,
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    return json.loads(probe.stdout)


def _resolved_config(
    source: dict,
    *,
    model_path: Path,
    visual_protocol: dict[str, object],
    residual_correction: str,
    training_epochs: int,
) -> dict:
    config = json.loads(json.dumps(source))
    uses_replay_consolidation = bool(
        visual_protocol.get("replay_functional_consolidation", False)
        and residual_correction == "kan"
    )
    config.update(
        {
            "epochs": training_epochs,
            "observation_objective": visual_protocol["observation_objective"],
            "observation_encoder": "dinov3_vits16",
            "dinov3_model_path": str(model_path),
            "dinov3_input_size": DINOV3_INPUT_SIZE,
            "dinov3_max_batch_size": DINOV3_MAX_BATCH_SIZE,
            "dinov3_feature_cache_dtype": DINOV3_CACHE_DTYPE,
            "dinov3_feature_loss_scale": DINOV3_FEATURE_LOSS_SCALE,
            "dinov3_feature_mode": visual_protocol["feature_mode"],
            "dinov3_patch_pool_size": visual_protocol["patch_pool_size"],
            "dinov3_patch_feature_dim": visual_protocol["patch_feature_dim"],
            "dinov3_patch_projection": visual_protocol["patch_projection"],
            "dinov3_patch_projection_frames": visual_protocol[
                "patch_projection_frames"
            ],
            "dinov3_feature_loss_kind": visual_protocol["feature_loss_kind"],
            "dinov3_feature_std_floor": DINOV3_FEATURE_STD_FLOOR,
            "actor_network": "mlp",
            "fresh_ac": False,
            "residual_correction": residual_correction,
            "residual_bottleneck_features": RESIDUAL_BOTTLENECK_FEATURES,
            "residual_grid_size": RESIDUAL_GRID_SIZE,
            "residual_input_min": RESIDUAL_INPUT_RANGE[0],
            "residual_input_max": RESIDUAL_INPUT_RANGE[1],
            "residual_rms_norm_epsilon": RESIDUAL_RMS_NORM_EPSILON,
            "residual_alpha": RESIDUAL_ALPHA,
            "residual_input_mode": visual_protocol["residual_input_mode"],
            "residual_consolidation": (
                "replay_functional" if uses_replay_consolidation else "none"
            ),
            "residual_consolidation_batches": RESIDUAL_CONSOLIDATION_BATCHES,
            "residual_consolidation_imagination_horizon": (
                RESIDUAL_CONSOLIDATION_IMAGINATION_HORIZON
            ),
            "residual_consolidation_gradient_power": (
                RESIDUAL_CONSOLIDATION_GRADIENT_POWER
            ),
            "residual_consolidation_min_plasticity": (
                RESIDUAL_CONSOLIDATION_MIN_PLASTICITY
            ),
            "residual_consolidation_anchor_loss_scale": (
                RESIDUAL_CONSOLIDATION_ANCHOR_LOSS_SCALE
            ),
            "shared_core_mode": (
                "freeze_after_first_task"
                if residual_correction != "none"
                else "trainable"
            ),
        }
    )
    return config


def main(*, default_visual_version: str = "v1") -> int:
    parser = _parser(default_visual_version=default_visual_version)
    args = parser.parse_args()
    if args.dinov3_model_path is None:
        parser.error(
            "--dinov3-model-path or DINOV3_MODEL_PATH is required; online loading "
            "is intentionally disabled"
        )
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
    visual_protocol = PROTOCOLS[args.visual_version]
    role, residual_correction = VARIANT_ROLES[args.variant]
    method, output_prefix = visual_protocol["variants"][args.variant]
    swap_sched = source_config["esc"]["kwargs"]["swap_sched"]
    training_epochs = (
        source_config["epochs"]
        if args.task_prefix_length is None
        else swap_sched * args.task_prefix_length
    )
    config = _resolved_config(
        source_config,
        model_path=model_path,
        visual_protocol=visual_protocol,
        residual_correction=residual_correction,
        training_epochs=training_epochs,
    )

    output_prefix = str(output_prefix)
    method = str(method)
    role = str(role)
    if config["residual_consolidation"] == "replay_functional":
        role = "primary-replay-consolidated-method"
    if args.task_prefix_length is not None:
        output_prefix += f"_t{args.task_prefix_length}_pilot"
        method += f"-T{args.task_prefix_length}Pilot"
        role += "-pilot"
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else ROOT
        / "runs"
        / f"{output_prefix}_{args.curriculum}_s{args.seed}_analysis"
    )
    config_path = output_dir / "resolved_training_config.json"
    snapshot_dir = output_dir / "analysis_snapshots"

    env = os.environ.copy()
    thread_env: dict[str, str] = {}
    if args.cpu_threads is not None:
        thread_env = {key: str(args.cpu_threads) for key in THREAD_ENV_KEYS}
        env.update(thread_env)
    # The ARROW subprocess runs from the vendored tree, so both the clean
    # package and the project-root namespace for vendored adapters are needed.
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
        "--analysis-snapshot-dir",
        str(snapshot_dir),
        "--compile-world-model",
        "--fused-adam",
        "--tf32",
    ]
    if args.task_prefix_length is not None:
        command.extend(("--epochs", str(training_epochs), "--evaluate-final"))
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
    boundary_epochs = list(range(swap_sched - 1, training_epochs, swap_sched))
    tasks = source_config["esc"]["env_configs"]
    if args.task_prefix_length is not None:
        tasks = tasks[: args.task_prefix_length]

    launch = {
        "method": method,
        "role": role,
        "protocol": visual_protocol["protocol"],
        "started_at_utc": None,
        "project_git": project_git,
        "upstream_arrow_commit": UPSTREAM_COMMIT,
        "source": str(ARROW_ROOT),
        "source_config": str(source_config_path),
        "resolved_training_config": str(config_path),
        "output_dir": str(output_dir),
        "analysis_snapshot_dir": str(snapshot_dir),
        "variant": args.variant,
        "visual_version": args.visual_version,
        "curriculum": args.curriculum,
        "seed_id": args.seed,
        "seed": SEEDS[args.seed],
        "training_scope": {
            "task_prefix_length": args.task_prefix_length,
            "epochs": training_epochs,
            "task_duration_epochs": swap_sched,
            "tasks": [task["name"] for task in tasks],
            "agent_decisions": agent_decisions,
            "raw_environment_frames": raw_environment_frames,
            "world_model_updates": training_epochs
            * source_config["steps_per_batch"],
            "actor_critic_updates": training_epochs
            * source_config["ac_train_steps"],
            "task_boundary_epochs": boundary_epochs,
        },
        "observation": {
            "encoder": "frozen DINOv3 ViT-S/16",
            "model_artifact": model_artifact,
            "input_size": DINOV3_INPUT_SIZE,
            "feature_mode": visual_protocol["feature_mode"],
            "patch_pool_size": visual_protocol["patch_pool_size"],
            "patch_feature_dim": visual_protocol["patch_feature_dim"],
            "patch_projection": visual_protocol["patch_projection"],
            "patch_projection_frames": visual_protocol[
                "patch_projection_frames"
            ],
            "patch_projection_fit": (
                "closed-form PCA before the first world-model update"
                if visual_protocol["patch_projection"] == "task1_pca"
                else None
            ),
            "feature_dim": visual_protocol["feature_dim"],
            "preprocessing": {
                "resize": [DINOV3_INPUT_SIZE, DINOV3_INPUT_SIZE],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "pixel_decoder": False,
            "objective": visual_protocol["objective_description"],
            "prediction_state": visual_protocol["prediction_state"],
            "feature_loss_kind": visual_protocol["feature_loss_kind"],
            "feature_loss_scale": DINOV3_FEATURE_LOSS_SCALE,
            "feature_std_floor": DINOV3_FEATURE_STD_FLOOR,
            "target_gradient": "stopped",
            "first_and_reset_steps_masked": visual_protocol[
                "first_and_reset_steps_masked"
            ],
        },
        "residuals": {
            "kind": residual_correction,
            "input_mode": config["residual_input_mode"],
            "placements": (
                []
                if args.variant == "dino"
                else (
                    [
                        "dynamics [z,a,h] -> hidden residual",
                        "posterior [embedding,h] -> logits residual",
                        "latent prior h -> logits residual",
                        "reward [z,h] -> reward residual",
                        "continue [z,h] -> continuation-logit residual",
                        "feature predictor [z,h] -> feature residual",
                        "actor [z,h] -> action-logit residual",
                        "critic [z,h] -> value-logit residual",
                    ]
                    if config["residual_input_mode"] == "module_input"
                    else [
                        "post-GRU hidden",
                        "posterior logits",
                        "latent prior logits",
                        "reward",
                        "continue",
                        "feature predictor",
                        "actor logits",
                        "critic logits",
                    ]
                )
            ),
            "independent_modules": True,
            "bottleneck_features": RESIDUAL_BOTTLENECK_FEATURES,
            "grid_size": (
                RESIDUAL_GRID_SIZE if args.variant == "kan" else None
            ),
            "grid_range": (
                list(RESIDUAL_INPUT_RANGE) if args.variant == "kan" else None
            ),
            "grid_trainable": False if args.variant == "kan" else None,
            "alpha": RESIDUAL_ALPHA,
            "zero_initialized_output": True,
            "trained_from_task_1": args.variant != "dino",
            "task_1_optimization": (
                "joint base-and-residual optimization"
                if args.variant != "dino"
                else None
            ),
            "task_1_base_priority": (
                "zero residual output at initialization and alpha=0.1"
                if args.variant != "dino"
                else None
            ),
            "matched_core_parameters": RESIDUAL_CORE_PARAMETERS,
            "task_specific_parameters": False,
            "task_id_or_router": False,
            "coordinate_map_frozen_after_task_1": (
                config["residual_consolidation"] == "replay_functional"
            ),
            "consolidation_state_storage_bytes": (
                RESIDUAL_MODULE_COUNT
                * (3 * RESIDUAL_CORE_PARAMETERS * 4 + 4 + 8 + 1)
                if args.variant == "kan"
                and config["residual_consolidation"] == "replay_functional"
                else 0
            ),
        },
        "residual_consolidation": {
            "mode": config["residual_consolidation"],
            "boundary_timing": (
                "before collecting the next task"
                if config["residual_consolidation"] == "replay_functional"
                else None
            ),
            "importance": (
                "squared local output Jacobian per Gaussian RBF coefficient"
                if config["residual_consolidation"] == "replay_functional"
                else None
            ),
            "replay_batches": (
                RESIDUAL_CONSOLIDATION_BATCHES
                if config["residual_consolidation"] == "replay_functional"
                else 0
            ),
            "deterministic_imagination_horizon": (
                RESIDUAL_CONSOLIDATION_IMAGINATION_HORIZON
                if config["residual_consolidation"] == "replay_functional"
                else 0
            ),
            "gradient_power": (
                RESIDUAL_CONSOLIDATION_GRADIENT_POWER
                if config["residual_consolidation"] == "replay_functional"
                else None
            ),
            "minimum_plasticity": (
                RESIDUAL_CONSOLIDATION_MIN_PLASTICITY
                if config["residual_consolidation"] == "replay_functional"
                else None
            ),
            "post_adam_parameter_delta_scaling": (
                config["residual_consolidation"] == "replay_functional"
            ),
            "anchor_loss_scale": (
                RESIDUAL_CONSOLIDATION_ANCHOR_LOSS_SCALE
                if config["residual_consolidation"] == "replay_functional"
                else None
            ),
            "boundary_importance_accumulator_peak_bytes": (
                RESIDUAL_MODULE_COUNT * RESIDUAL_CORE_PARAMETERS * 4
                if config["residual_consolidation"] == "replay_functional"
                else 0
            ),
            "post_adam_delta_snapshot_peak_bytes": (
                6 * RESIDUAL_CORE_PARAMETERS * 4
                if config["residual_consolidation"] == "replay_functional"
                else 0
            ),
            "extra_environment_interactions": 0,
            "extra_gradient_updates": 0,
            "training_rng_restored_after_estimation": True,
        },
        "shared_core": {
            "mode": config["shared_core_mode"],
            "freeze_after_completed_task": 1 if args.variant != "dino" else None,
            "task_1_base_trainable": args.variant != "dino",
            "task_1_residual_trainable": args.variant != "dino",
            "frozen_modules": (
                [
                    "frozen DINOv3 encoder (frozen from initialization)",
                    "RSSM input MLP",
                    "RSSM GRUCell",
                    "posterior representation MLP",
                    "latent transition MLP",
                    "feature predictor base head",
                    "reward base head",
                    "continue base head",
                    "actor MLP trunk and logits",
                    "critic MLP trunk and logits",
                ]
                if args.variant != "dino"
                else []
            ),
            "trainable_after_freeze": (
                (
                    ["Gaussian RBF coefficients in each residual"]
                    if config["residual_consolidation"] == "replay_functional"
                    else [
                        "dynamics residual",
                        "posterior residual",
                        "prior residual",
                        "reward residual",
                        "continue residual",
                        "feature-prediction residual",
                        "actor residual",
                        "critic residual",
                    ]
                )
                if args.variant != "dino"
                else []
            ),
        },
        "replay": {
            "capacity_and_sampling": "ARROW-50 unchanged",
            "base_storage": _arrow_replay_storage_budget(source_config),
            "feature_cache": _feature_cache_budget(
                source_config,
                dtype=DINOV3_CACHE_DTYPE,
                feature_dim=int(visual_protocol["feature_dim"]),
            ),
        },
        "determinism": {
            "python_numpy_torch_environment_and_replay_seeded": True,
            "torch_deterministic_algorithms": False,
            "tf32_enabled": True,
            "known_nondeterminism": [
                "CUDA kernels are not forced into deterministic-only mode"
            ],
        },
        "runtime_dependencies": dependency_versions,
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

    if dependency_versions != DINOV3_DEPENDENCIES:
        raise RuntimeError(
            "KARROW requires the pinned DINOv3 dependencies before launch: "
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
    status = {
        "complete": return_code == 0,
        "return_code": return_code,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output_dir / "run_status.json", status)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
