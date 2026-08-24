#!/usr/bin/env python3
"""Launch task-aware MoE-ARROW with fixed ARROW-50 training budgets."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
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


PATCH_PROJECTION = "fixed_orthogonal"
PATCH_PROJECTION_SEED = 0
FULL_PATCH_POOL_SIZE = DINOV3_INPUT_SIZE // 16
FULL_PATCH_FEATURE_DIM = 384
CONV_ADAPTER_OUTPUT_CHANNELS = 64
CONV_ADAPTER_OUTPUT_GRID_SIZE = 8
RSSM_LATENT_SHAPE = (32, 32)
CNN_ENCODER_OUTPUT_SIZE = 4_096


@dataclass(frozen=True)
class LaunchVariant:
    method: str
    code_id: str
    protocol: str
    role: str
    output_prefix: str
    current_task_fraction: float
    observation_objective: str
    observation_encoder: str
    task_banked_image_encoder: bool
    replay_observation_dtype: str
    feature_loss: str
    random_policy: str
    shared_core_mode: str
    full_task_experts: bool
    patch_pool_size: int
    patch_feature_dim: int
    patch_projection: str
    patch_projection_seed: int
    replay_feature_mode: str
    patch_adapter: str
    pixel_decoder: bool
    observation_description: str


@dataclass(frozen=True)
class PrecisionProfile:
    compute_dtype: str
    dinov3_max_batch_size: int
    feature_dtype: str
    autocast_enabled: bool
    name_suffix: str
    output_suffix: str


@dataclass(frozen=True)
class BatchProfile:
    scale: int
    protocol_suffix: str
    output_suffix: str
    config_overrides: dict[str, int | float]
    hypothesis: str


PRECISION_PROFILES = {
    "fp32-tf32": PrecisionProfile(
        compute_dtype="float32",
        dinov3_max_batch_size=DINOV3_MAX_BATCH_SIZE,
        feature_dtype=DINOV3_CACHE_DTYPE,
        autocast_enabled=False,
        name_suffix="",
        output_suffix="",
    ),
    "bf16-amp": PrecisionProfile(
        compute_dtype="bfloat16",
        dinov3_max_batch_size=512,
        feature_dtype="bfloat16",
        autocast_enabled=True,
        name_suffix="-BF16AMP",
        output_suffix="_bf16_amp",
    ),
}


BATCH_PROFILES = {
    "x2-linear-lr": BatchProfile(
        scale=2,
        protocol_suffix="LargeBatchX2LinearLR",
        output_suffix="large_batch_x2_linear_lr",
        config_overrides={
            "mb_n_size": 32,
            "pretrain_mb_n_size": 32,
            "steps_per_batch": 500,
            "pretrain_steps": 15_000,
            "ac_train_sync": 256,
            "ac_train_steps": 400,
            "wm_lr": 2e-4,
            "ac_lr": 2e-4,
        },
        hypothesis=(
            "Doubling each global optimization batch while halving optimizer "
            "steps preserves sampled-frame use, reduces DDP synchronizations, "
            "and linear learning-rate scaling preserves update magnitude."
        ),
    ),
    "x4-linear-lr": BatchProfile(
        scale=4,
        protocol_suffix="LargeBatchX4LinearLR",
        output_suffix="large_batch_x4_linear_lr",
        config_overrides={
            "mb_n_size": 64,
            "pretrain_mb_n_size": 64,
            "steps_per_batch": 250,
            "pretrain_steps": 7_500,
            "ac_train_sync": 512,
            "ac_train_steps": 200,
            "wm_lr": 4e-4,
            "ac_lr": 4e-4,
        },
        hypothesis=(
            "Quadrupling each global optimization batch while quartering "
            "optimizer steps preserves sampled-frame use, amortizes DDP "
            "synchronization, and linear learning-rate scaling preserves "
            "update magnitude."
        ),
    ),
}


TASK1_TUNING_PROFILES = {
    "aclr5e5": {
        "protocol_suffix": "Task1AcLR5e5",
        "output_suffix": "task1_aclr5e5",
        "config_overrides": {
            "ac_lr": 5e-5,
            "ac_entropy_scale": 3e-4,
        },
        "hypothesis": (
            "Halving the actor-critic learning rate will reduce the post-peak "
            "policy regression observed on MsPacman while retaining the "
            "published entropy scale."
        ),
    },
    "aclr5e5-ent1e4": {
        "protocol_suffix": "Task1AcLR5e5Ent1e4",
        "output_suffix": "task1_aclr5e5_ent1e4",
        "config_overrides": {
            "ac_lr": 5e-5,
            "ac_entropy_scale": 1e-4,
        },
        "hypothesis": (
            "Conditional on the halved actor-critic learning rate, reducing "
            "entropy regularization will improve final MsPacman exploitation."
        ),
    },
}


MOE_ARROW_VARIANT = LaunchVariant(
    method="MoE-ARROW-50",
    code_id="moe_arrow",
    protocol="MoE-ARROW-v1-Atari-TaskAware",
    role="primary-task-aware-continual-method",
    output_prefix="moe_arrow_ar50",
    current_task_fraction=0.5,
    observation_objective="dinov3_next_feature",
    observation_encoder="dinov3_vits16",
    task_banked_image_encoder=False,
    replay_observation_dtype="float32",
    feature_loss="cosine",
    random_policy="first",
    shared_core_mode="trainable",
    full_task_experts=False,
    patch_pool_size=DINOV3_PATCH_POOL_SIZE,
    patch_feature_dim=DINOV3_PATCH_FEATURE_DIM,
    patch_projection=PATCH_PROJECTION,
    patch_projection_seed=PATCH_PROJECTION_SEED,
    replay_feature_mode="cached",
    patch_adapter="none",
    pixel_decoder=False,
    observation_description="one-step prior prediction of stopped spatial features",
)

CNN_FULLBANK_VARIANT = LaunchVariant(
    method="CNN-FullBank-ARROW-50",
    code_id="cnn_fullbank_arrow",
    protocol="CNN-FullBank-ARROW-v1-Atari-TaskAware",
    role="fully-task-banked-dreamerv3-arrow-method",
    output_prefix="cnn_fullbank_arrow_ar50",
    current_task_fraction=1.0,
    observation_objective="reconstruction",
    observation_encoder="cnn",
    task_banked_image_encoder=True,
    replay_observation_dtype="uint8",
    feature_loss="cosine",
    random_policy="new",
    shared_core_mode="task_isolated",
    full_task_experts=True,
    patch_pool_size=0,
    patch_feature_dim=0,
    patch_projection="none",
    patch_projection_seed=0,
    replay_feature_mode="cached",
    patch_adapter="none",
    pixel_decoder=True,
    observation_description=(
        "DreamerV3 pixel reconstruction from a task-banked CNN encoder"
    ),
)

DINO_FULLBANK_VARIANT = LaunchVariant(
    method="DINO-FullBank-ARROW-50",
    code_id="dino_fullbank_arrow",
    protocol="DINO-FullBank-ARROW-v2-Atari-TaskAware",
    role="corrected-task-aware-upper-bound-method",
    output_prefix="dino_fullbank_arrow_ar50",
    current_task_fraction=1.0,
    observation_objective="dinov3_posterior_feature",
    observation_encoder="dinov3_vits16",
    task_banked_image_encoder=False,
    replay_observation_dtype="float32",
    feature_loss="batch_standardized_smooth_l1",
    random_policy="new",
    shared_core_mode="task_isolated",
    full_task_experts=True,
    patch_pool_size=DINOV3_PATCH_POOL_SIZE,
    patch_feature_dim=DINOV3_PATCH_FEATURE_DIM,
    patch_projection=PATCH_PROJECTION,
    patch_projection_seed=PATCH_PROJECTION_SEED,
    replay_feature_mode="cached",
    patch_adapter="none",
    pixel_decoder=False,
    observation_description="current posterior reconstruction of stopped spatial features",
)

DINO_PATCHBANK_VARIANT = LaunchVariant(
    method="DINO-PatchBank-ARROW-50",
    code_id="dino_patchbank_arrow",
    protocol="DINO-PatchBank-ARROW-v3-Atari-TaskAware",
    role="full-patch-dino-dreamerv3-task-aware-method",
    output_prefix="dino_patchbank_arrow_ar50",
    current_task_fraction=1.0,
    observation_objective="reconstruction",
    observation_encoder="dinov3_vits16",
    task_banked_image_encoder=False,
    replay_observation_dtype="float32",
    feature_loss="cosine",
    random_policy="new",
    shared_core_mode="task_isolated",
    full_task_experts=True,
    patch_pool_size=FULL_PATCH_POOL_SIZE,
    patch_feature_dim=FULL_PATCH_FEATURE_DIM,
    patch_projection="none",
    patch_projection_seed=0,
    replay_feature_mode="on_the_fly",
    patch_adapter="none",
    pixel_decoder=True,
    observation_description="DreamerV3 pixel reconstruction from full DINOv3 patches",
)

DINO_CONVBANK_VARIANT = LaunchVariant(
    method="DINO-ConvBank-ARROW-50",
    code_id="dino_convbank_arrow",
    protocol="DINO-ConvBank-ARROW-v4-Atari-TaskAware",
    role="shared-conv-adapter-dino-dreamerv3-task-aware-method",
    output_prefix="dino_convbank_arrow_ar50",
    current_task_fraction=1.0,
    observation_objective="reconstruction",
    observation_encoder="dinov3_vits16",
    task_banked_image_encoder=False,
    replay_observation_dtype="uint8",
    feature_loss="cosine",
    random_policy="new",
    shared_core_mode="task_banked_shared_adapter",
    full_task_experts=True,
    patch_pool_size=FULL_PATCH_POOL_SIZE,
    patch_feature_dim=FULL_PATCH_FEATURE_DIM,
    patch_projection="none",
    patch_projection_seed=0,
    replay_feature_mode="on_the_fly",
    patch_adapter="conv_3x3_stride2",
    pixel_decoder=True,
    observation_description=(
        "DreamerV3 pixel reconstruction from a shared learned 8x8x64 DINO adapter"
    ),
)

LAUNCH_VARIANTS = {
    "moe": MOE_ARROW_VARIANT,
    "cnn-fullbank": CNN_FULLBANK_VARIANT,
    "dino-fullbank": DINO_FULLBANK_VARIANT,
    "dino-patchbank": DINO_PATCHBANK_VARIANT,
    "dino-convbank": DINO_CONVBANK_VARIANT,
}


def _posterior_parameter_count(
    *,
    embedding_features: int,
    hidden_features: int,
    mlp_features: int,
    mlp_layers: int,
) -> int:
    """Match the vendored RSSM Representation parameterization."""
    if mlp_layers < 1:
        raise ValueError("posterior MLP must contain at least one layer")
    latent_features = RSSM_LATENT_SHAPE[0] * RSSM_LATENT_SHAPE[1]
    sizes = (
        [embedding_features + hidden_features]
        + [mlp_features] * (mlp_layers - 1)
        + [latent_features]
    )
    parameters = sum(
        input_features * output_features + output_features
        for input_features, output_features in zip(sizes[:-1], sizes[1:])
    )
    parameters += sum(2 * output_features for output_features in sizes[1:-1])
    if mlp_layers > 1:
        parameters += embedding_features * latent_features + latent_features
    return parameters


def _cnn_encoder_parameter_count(*, img_channels: int, channels: int) -> int:
    channel_widths = [channels * 2**index for index in range(4)]
    input_widths = [img_channels, *channel_widths[:-1]]
    convolution_parameters = sum(
        input_width * output_width * 4 * 4 + output_width
        for input_width, output_width in zip(input_widths, channel_widths)
    )
    layer_norm_parameters = sum(2 * width for width in channel_widths)
    return convolution_parameters + layer_norm_parameters


def _parser(*, default_method: str = "moe") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch a task-aware DINO/ARROW method with one expert and actor per game"
    )
    parser.add_argument(
        "--method",
        choices=tuple(LAUNCH_VARIANTS),
        default=default_method,
        help=(
            "moe is the original partial-expert route; cnn-fullbank and the "
            "DINO variants select named task-bank protocols"
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
    parser.add_argument(
        "--devices",
        type=int,
        choices=(1, 2, 4),
        default=1,
        help=(
            "Number of CUDA devices. Two- and four-device native DDP are "
            "validated for DINO-ConvBank and CNN-FullBank"
        ),
    )
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument(
        "--precision-profile",
        choices=tuple(PRECISION_PROFILES),
        default=None,
        help=(
            "Explicit execution precision. DINO-ConvBank and CNN-FullBank default "
            "to and require BF16 AMP; other methods default to FP32/TF32"
        ),
    )
    parser.add_argument(
        "--task1-tuning-profile",
        choices=tuple(TASK1_TUNING_PROFILES),
        help=(
            "Named MsPacman acquisition ablation. Requires dino-convbank and "
            "--task-prefix-length 1; data and update budgets remain fixed"
        ),
    )
    parser.add_argument(
        "--batch-profile",
        choices=tuple(BATCH_PROFILES),
        help=(
            "Named sample-matched large-batch ablation. Requires cnn-fullbank "
            "with --devices 4; sampled replay/context frames and environment "
            "interaction remain fixed while optimizer steps decrease"
        ),
    )
    parser.add_argument("--swanlab-project")
    parser.add_argument("--swanlab-experiment-name")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _resolved_config(
    source: dict,
    *,
    model_path: Path | None,
    epochs: int,
    variant: LaunchVariant = MOE_ARROW_VARIANT,
    precision_profile: PrecisionProfile = PRECISION_PROFILES["fp32-tf32"],
    data_parallel_world_size: int = 1,
    config_overrides: dict[str, int | float] | None = None,
) -> dict:
    config = json.loads(json.dumps(source))
    config.update(
        {
            "epochs": epochs,
            "compute_dtype": precision_profile.compute_dtype,
            "data_parallel_world_size": data_parallel_world_size,
            "continual_method": variant.code_id,
            "rssm_num_experts": len(config["esc"]["env_configs"]),
            "moe_arrow_current_task_fraction": 0.5,
            "dino_fullbank_current_task_fraction": 1.0,
            "observation_objective": variant.observation_objective,
            "observation_encoder": variant.observation_encoder,
            "task_banked_image_encoder": variant.task_banked_image_encoder,
            "dinov3_model_path": (
                str(model_path)
                if variant.observation_encoder == "dinov3_vits16"
                else None
            ),
            "dinov3_input_size": DINOV3_INPUT_SIZE,
            "dinov3_max_batch_size": (
                precision_profile.dinov3_max_batch_size
                if variant.observation_encoder == "dinov3_vits16"
                else DINOV3_MAX_BATCH_SIZE
            ),
            "dinov3_feature_cache_dtype": (
                precision_profile.feature_dtype
                if variant.observation_encoder == "dinov3_vits16"
                else DINOV3_CACHE_DTYPE
            ),
            "dinov3_replay_feature_mode": variant.replay_feature_mode,
            "dinov3_feature_loss_scale": DINOV3_FEATURE_LOSS_SCALE,
            "dinov3_feature_mode": (
                "patch_grid"
                if variant.observation_encoder == "dinov3_vits16"
                else "cls"
            ),
            "dinov3_patch_pool_size": (
                variant.patch_pool_size
                if variant.observation_encoder == "dinov3_vits16"
                else DINOV3_PATCH_POOL_SIZE
            ),
            "dinov3_patch_feature_dim": (
                variant.patch_feature_dim
                if variant.observation_encoder == "dinov3_vits16"
                else 384
            ),
            "dinov3_patch_projection": (
                variant.patch_projection
                if variant.observation_encoder == "dinov3_vits16"
                else "none"
            ),
            "dinov3_patch_projection_frames": 0,
            "dinov3_patch_projection_seed": (
                variant.patch_projection_seed
                if variant.observation_encoder == "dinov3_vits16"
                else 0
            ),
            "dinov3_patch_adapter": (
                variant.patch_adapter
                if variant.observation_encoder == "dinov3_vits16"
                else "none"
            ),
            "dinov3_feature_loss_kind": variant.feature_loss,
            "dinov3_feature_std_floor": DINOV3_FEATURE_STD_FLOOR,
            "replay_observation_dtype": variant.replay_observation_dtype,
            "actor_network": "mlp",
            "fresh_ac": False,
            "random_policy": variant.random_policy,
            "residual_correction": "none",
            "residual_consolidation": "none",
            "shared_core_mode": variant.shared_core_mode,
        }
    )
    for replay_config in config["replay_buffers"]:
        replay_config["rb_device"] = "cpu"
    if config_overrides is not None:
        config.update(config_overrides)
    return config


def main(*, default_method: str = "moe") -> int:
    parser = _parser(default_method=default_method)
    args = parser.parse_args()
    variant = LAUNCH_VARIANTS[args.method]
    requires_dinov3 = variant.observation_encoder == "dinov3_vits16"
    bf16_required_methods = {"dino-convbank", "cnn-fullbank"}
    precision_profile_name = args.precision_profile or (
        "bf16-amp" if args.method in bf16_required_methods else "fp32-tf32"
    )
    precision_profile = PRECISION_PROFILES[precision_profile_name]
    if args.method in bf16_required_methods and precision_profile_name != "bf16-amp":
        parser.error(f"{args.method} requires --precision-profile bf16-amp")
    if precision_profile_name == "bf16-amp" and args.method not in bf16_required_methods:
        parser.error(
            "--precision-profile bf16-amp is currently validated only for "
            "dino-convbank and cnn-fullbank"
        )
    if args.devices > 1 and args.method not in bf16_required_methods:
        parser.error(
            "--devices 2/4 is validated only for dino-convbank and cnn-fullbank"
        )
    if args.task1_tuning_profile is not None:
        if args.method != "dino-convbank":
            parser.error(
                "--task1-tuning-profile is validated only for dino-convbank"
            )
        if args.task_prefix_length != 1:
            parser.error(
                "--task1-tuning-profile requires --task-prefix-length 1"
            )
    if args.batch_profile is not None:
        if args.method != "cnn-fullbank":
            parser.error("--batch-profile is validated only for cnn-fullbank")
        if args.devices != 4:
            parser.error("--batch-profile requires --devices 4")
    if requires_dinov3 and args.dinov3_model_path is None:
        parser.error("--dinov3-model-path or DINOV3_MODEL_PATH is required")
    if args.cpu_threads is not None and args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if args.swanlab_experiment_name and not args.swanlab_project:
        parser.error("--swanlab-experiment-name requires --swanlab-project")

    project_git = (
        git_state(ROOT) if args.dry_run else require_synced_training_git_state(ROOT)
    )
    python = args.python.resolve()
    model_path = (
        args.dinov3_model_path.expanduser().resolve()
        if requires_dinov3 and args.dinov3_model_path is not None
        else None
    )
    model_artifact = (
        _model_artifact_manifest(model_path) if model_path is not None else None
    )
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
    tuning_profile = (
        TASK1_TUNING_PROFILES[args.task1_tuning_profile]
        if args.task1_tuning_profile is not None
        else None
    )
    batch_profile = (
        BATCH_PROFILES[args.batch_profile]
        if args.batch_profile is not None
        else None
    )
    config_overrides: dict[str, int | float] = {}
    if tuning_profile is not None:
        config_overrides.update(tuning_profile["config_overrides"])
    if batch_profile is not None:
        config_overrides.update(batch_profile.config_overrides)
    config = _resolved_config(
        source_config,
        model_path=model_path,
        epochs=training_epochs,
        variant=variant,
        precision_profile=precision_profile,
        data_parallel_world_size=args.devices,
        config_overrides=config_overrides or None,
    )
    allocated_experts = int(config["rssm_num_experts"])

    method = variant.method
    role = variant.role
    output_prefix = variant.output_prefix
    protocol = variant.protocol
    if precision_profile.name_suffix:
        method += precision_profile.name_suffix
        role += "-bf16-runtime-profile"
        output_prefix += precision_profile.output_suffix
        protocol = protocol.replace("-Atari-", "-BF16AMP-Atari-")
    if config["replay_observation_dtype"] == "uint8":
        method += "-Uint8Replay"
        role += "-uint8-replay"
        output_prefix += "_uint8_replay"
        protocol = protocol.replace("-Atari-", "-Uint8Replay-Atari-")
    if args.devices > 1:
        method += f"-DP{args.devices}"
        role += (
            f"-sample-matched-large-batch-ddp{args.devices}"
            if batch_profile is not None
            else f"-fixed-global-batch-ddp{args.devices}"
        )
        output_prefix += f"_dp{args.devices}"
        protocol = protocol.replace("-Atari-", f"-DP{args.devices}-Atari-")
    if batch_profile is not None:
        method += f"-{batch_profile.protocol_suffix}"
        role += "-optimization-batch-ablation"
        output_prefix += f"_{batch_profile.output_suffix}"
        protocol = protocol.replace(
            "-Atari-", f"-{batch_profile.protocol_suffix}-Atari-"
        )
    if tuning_profile is not None:
        tuning_suffix = str(tuning_profile["protocol_suffix"])
        method += f"-{tuning_suffix}"
        role += "-task1-acquisition-ablation"
        output_prefix += f"_{tuning_profile['output_suffix']}"
        protocol = protocol.replace("-Atari-", f"-{tuning_suffix}-Atari-")
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
    dependency_versions = (
        _dinov3_dependency_versions(python, env) if requires_dinov3 else {}
    )

    command = [str(python)]
    if args.devices > 1:
        command.extend(
            (
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node",
                str(args.devices),
            )
        )
    command.extend(
        [
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
    )
    if args.profile_stages:
        command.append("--profile-stages")
    if args.swanlab_project:
        command.extend(("--swanlab-project", args.swanlab_project))
    if args.swanlab_experiment_name:
        command.extend(("--swanlab-experiment-name", args.swanlab_experiment_name))

    tasks = source_config["esc"]["env_configs"]
    visited_tasks = (
        tasks
        if args.task_prefix_length is None
        else tasks[: args.task_prefix_length]
    )
    decisions_per_epoch = source_config["n_sync"] * source_config["gen_seq_len"]
    random_collection_epochs = (
        len(visited_tasks) if variant.random_policy == "new" else 1
    )
    extra_collections_per_random_epoch = 0
    if source_config.get("pretrain_enabled", True):
        extra_collections_per_random_epoch = (
            source_config.get("pretrain_data_multiplier", 4) - 1
        )
    collection_epoch_equivalents = (
        training_epochs
        + random_collection_epochs * extra_collections_per_random_epoch
    )
    agent_decisions = decisions_per_epoch * collection_epoch_equivalents
    raw_environment_frames = agent_decisions * source_config["env_repeat"]
    extra_environment_interactions = (
        max(0, random_collection_epochs - 1)
        * extra_collections_per_random_epoch
        * decisions_per_epoch
        * source_config["env_repeat"]
    )
    feature_dim = (
        CNN_ENCODER_OUTPUT_SIZE
        if variant.observation_encoder == "cnn"
        else variant.patch_feature_dim * variant.patch_pool_size**2
    )
    posterior_embedding_dim = (
        CNN_ENCODER_OUTPUT_SIZE
        if variant.observation_encoder == "cnn"
        else CONV_ADAPTER_OUTPUT_CHANNELS * CONV_ADAPTER_OUTPUT_GRID_SIZE**2
        if variant.patch_adapter == "conv_3x3_stride2"
        else feature_dim
    )
    posterior_mlp_layers = (
        1
        if source_config["wall_time_optimisation"]
        else source_config["mlp_layers"]
    )
    posterior_parameters_per_task = _posterior_parameter_count(
        embedding_features=posterior_embedding_dim,
        hidden_features=source_config["gru_units"],
        mlp_features=source_config["mlp_features"],
        mlp_layers=posterior_mlp_layers,
    )
    unadapted_posterior_parameters_per_task = _posterior_parameter_count(
        embedding_features=feature_dim,
        hidden_features=source_config["gru_units"],
        mlp_features=source_config["mlp_features"],
        mlp_layers=posterior_mlp_layers,
    )
    patch_adapter = {
        "kind": variant.patch_adapter,
        "shared_across_tasks": variant.patch_adapter != "none",
        "trainable": variant.patch_adapter != "none",
        "input_layout": (
            [variant.patch_pool_size] * 2 + [variant.patch_feature_dim]
            if variant.patch_adapter != "none"
            else None
        ),
        "output_layout": (
            [CONV_ADAPTER_OUTPUT_GRID_SIZE] * 2
            + [CONV_ADAPTER_OUTPUT_CHANNELS]
            if variant.patch_adapter != "none"
            else None
        ),
        "kernel_size": 3 if variant.patch_adapter != "none" else None,
        "stride": 2 if variant.patch_adapter != "none" else None,
        "padding": 1 if variant.patch_adapter != "none" else None,
        "normalization": (
            "channel_layer_norm_eps_1e-3"
            if variant.patch_adapter != "none"
            else None
        ),
        "activation": "silu" if variant.patch_adapter != "none" else None,
        "output_features": (
            posterior_embedding_dim if variant.patch_adapter != "none" else None
        ),
        "trainable_parameters": (
            FULL_PATCH_FEATURE_DIM
            * CONV_ADAPTER_OUTPUT_CHANNELS
            * 3
            * 3
            + CONV_ADAPTER_OUTPUT_CHANNELS
            + 2 * CONV_ADAPTER_OUTPUT_CHANNELS
            if variant.patch_adapter != "none"
            else 0
        ),
    }
    if variant.observation_encoder == "cnn":
        feature_cache = {
            "dtype": None,
            "feature_dim": 0,
            "storage_bytes": 0,
            "storage_backend": "none",
            "mode": "none_cnn_encodes_sampled_observations",
            "buffers": {},
            "retention_and_sampling": "not_applicable",
        }
    elif variant.replay_feature_mode == "cached":
        feature_cache = _feature_cache_budget(
            config,
            dtype=config["dinov3_feature_cache_dtype"],
            feature_dim=feature_dim,
        )
        feature_cache["storage_backend"] = "anonymous_cpu"
    else:
        feature_dtype = config["dinov3_feature_cache_dtype"]
        consumer_dtype = config["compute_dtype"]
        feature_cache = {
            "dtype": feature_dtype,
            "quantization_dtype": feature_dtype,
            "consumer_dtype": consumer_dtype,
            "feature_dim": feature_dim,
            "storage_bytes": 0,
            "storage_backend": "none",
            "mode": "on_the_fly_from_sampled_observations",
            "quantization_semantics": (
                "encoder output retained without a dtype round trip"
                if feature_dtype == consumer_dtype
                else "round through configured replay dtype before RSSM consumption"
            ),
            "buffers": {},
            "retention_and_sampling": (
                "no feature sidecar; uses the sampled ARROW observation batch"
            ),
        }
    base_replay_storage = _arrow_replay_storage_budget(config)
    if variant.code_id in {
        "cnn_fullbank_arrow",
        "dino_patchbank_arrow",
        "dino_convbank_arrow",
    }:
        base_replay_storage["observation_storage_backend"] = "file_mmap"
        base_replay_storage["anonymous_cpu_tensor_bytes"] = (
            base_replay_storage["allocated_tensor_bytes"]
            - base_replay_storage["observation_bytes"]
        )
        base_replay_storage["mmap_directory"] = "mmap_replay/observations"
        for buffer in base_replay_storage["buffers"].values():
            buffer["observation_storage_backend"] = "file_mmap"
    task_id_storage_bytes = 2 * source_config["data_n_max"] * 8
    local_world_model_sequences = config["mb_n_size"] // args.devices
    local_pretrain_sequences = config["pretrain_mb_n_size"] // args.devices
    local_actor_sequences = config["ac_train_sync"] // args.devices
    world_model_updates = training_epochs * config["steps_per_batch"]
    actor_critic_updates = training_epochs * config["ac_train_steps"]
    world_model_sampled_replay_frame_uses = (
        world_model_updates * config["mb_t_size"] * config["mb_n_size"]
    )
    actor_context_frame_uses = (
        actor_critic_updates * 4 * config["ac_train_sync"]
    )

    launch = {
        "method": method,
        "code_id": variant.code_id,
        "role": role,
        "protocol": protocol,
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
                (
                    "CNN/RSSM/head expert routing"
                    if variant.task_banked_image_encoder
                    else "RSSM/head expert routing"
                ),
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
            "world_model_updates": world_model_updates,
            "actor_critic_updates": actor_critic_updates,
            "world_model_sampled_replay_frame_uses": (
                world_model_sampled_replay_frame_uses
            ),
            "actor_context_frame_uses": actor_context_frame_uses,
        },
        "hyperparameter_tuning": (
            {
                "profile": args.task1_tuning_profile,
                "classification": "task1_acquisition_pilot",
                "hypothesis": tuning_profile["hypothesis"],
                "config_overrides": dict(tuning_profile["config_overrides"]),
                "world_model_learning_rate": config["wm_lr"],
                "fixed_data_and_update_budgets": True,
                "acquisition_gate": {
                    "task": "ALE/MsPacman-v5",
                    "after_completed_epochs": 90,
                    "rollouts": 16,
                    "metric": "raw_return_mean",
                    "minimum": 2000.0,
                    "use_intermediate_peak": False,
                },
            }
            if tuning_profile is not None
            else None
        ),
        "batch_tuning": (
            {
                "profile": args.batch_profile,
                "classification": "sample_matched_large_batch_ablation",
                "hypothesis": batch_profile.hypothesis,
                "scale": batch_profile.scale,
                "config_overrides": batch_profile.config_overrides,
                "learning_rate_rule": "linear_with_global_batch_scale",
                "environment_interaction_budget_unchanged": True,
                "optimization_sample_budgets_unchanged": True,
                "optimizer_update_counts_unchanged": False,
                "world_model_update_multiplier": 1 / batch_profile.scale,
                "actor_critic_update_multiplier": 1 / batch_profile.scale,
                "world_model_sampled_replay_frame_use_multiplier": 1.0,
                "actor_context_frame_use_multiplier": 1.0,
            }
            if batch_profile is not None
            else None
        ),
        "distributed_execution": {
            "enabled": args.devices > 1,
            "world_size": args.devices,
            "launcher": (
                "torch.distributed.run" if args.devices > 1 else "direct_python"
            ),
            "backend": "nccl" if args.devices > 1 else None,
            "gradient_parallelism": (
                "native PyTorch DistributedDataParallel with averaged gradients"
                if args.devices > 1
                else "none"
            ),
            "global_batch_policy": (
                f"sample-matched x{batch_profile.scale}; batches grow and "
                "optimizer steps shrink by the same factor"
                if batch_profile is not None
                else "fixed; no data or update budget increase"
            ),
            "world_model_sequences": {
                "global": config["mb_n_size"],
                "per_rank": local_world_model_sequences,
            },
            "pretrain_world_model_sequences": {
                "global": config["pretrain_mb_n_size"],
                "per_rank": local_pretrain_sequences,
            },
            "actor_context_sequences": {
                "global": config["ac_train_sync"],
                "per_rank": local_actor_sequences,
            },
            "replay": (
                "rank 0 makes one global FIFO/LTDM choice and one global draw; "
                "the sequence axis is scattered equally to all ranks"
                if args.devices > 1
                else "local authoritative ARROW replay"
            ),
            "replay_owner": "rank_0" if args.devices > 1 else "local_process",
            "collection": (
                "rank_0_only" if args.devices > 1 else "local_process"
            ),
            "evaluation": (
                "tasks partitioned by task_index modulo world_size"
                if args.devices > 1
                else "serial_tasks"
            ),
            "model_parameter_replicas": args.devices,
            "optimizer_state_replicas": args.devices,
        },
        "precision": {
            "profile": precision_profile_name,
            "autocast_enabled": precision_profile.autocast_enabled,
            "compute_dtype": precision_profile.compute_dtype,
            "parameter_dtype": "float32",
            "gradient_dtype": "float32 parameter gradients",
            "optimizer_state_dtype": "float32",
            "gradient_scaler": False,
            "sensitive_math_dtype": "float32",
            "sensitive_math": [
                "categorical sampling and KL",
                "symlog and symexp",
                "pixel, reward, and continuation losses",
                "lambda returns and value targets",
                "actor log-probabilities and critic distributions",
            ],
            "tf32_enabled_for_float32_matmuls": True,
            "dinov3_execution_chunk_size": (
                config["dinov3_max_batch_size"] if requires_dinov3 else None
            ),
            "world_model_optimization_batch": {
                "time": config["mb_t_size"],
                "sequences": config["mb_n_size"],
                "frames": config["mb_t_size"] * config["mb_n_size"],
                "unchanged": batch_profile is None,
            },
            "world_model_optimization_batch_per_rank": {
                "time": config["mb_t_size"],
                "sequences": local_world_model_sequences,
                "frames": config["mb_t_size"] * local_world_model_sequences,
            },
            "actor_context_batch_frames": 4 * config["ac_train_sync"],
            "actor_context_batch_frames_per_rank": 4 * local_actor_sequences,
            "optimizer_update_budgets_unchanged": batch_profile is None,
            "optimization_sample_budgets_unchanged": True,
        },
        "world_model": {
            "router": "hard_task_id",
            "routing_granularity": "one homogeneous-task minibatch",
            "allocated_experts": allocated_experts,
            "expert_modules": (
                [
                    *(
                        ["cnn_image_encoder"]
                        if variant.task_banked_image_encoder
                        else []
                    ),
                    "posterior_representation",
                    "recurrent_dynamics",
                    "latent_prior",
                    (
                        "pixel_decoder"
                        if variant.pixel_decoder
                        else "feature_predictor"
                    ),
                    "reward_head",
                    "continue_head",
                ]
                if variant.full_task_experts
                else [
                    "recurrent_dynamics",
                    "latent_prior",
                    "reward_head",
                    "continue_head",
                ]
            ),
            "shared_modules": (
                []
                if variant.task_banked_image_encoder
                else (
                    [
                        "frozen DINOv3 encoder",
                        "trainable shared DINO patch convolution adapter",
                    ]
                    if variant.patch_adapter != "none"
                    else ["frozen DINOv3 encoder"]
                )
                if variant.full_task_experts
                else [
                    "frozen DINOv3 encoder",
                    "posterior representation",
                    "feature predictor",
                ]
            ),
            "new_task_initialization": (
                "copy previous complete world-model expert once"
                if variant.full_task_experts
                else "copy previous task expert once"
            ),
            "old_task_expert_parameters_frozen": variant.full_task_experts,
            "old_task_parameters_frozen": (
                variant.full_task_experts and variant.patch_adapter == "none"
            ),
            "old_task_functionally_isolated": (
                variant.full_task_experts and variant.patch_adapter == "none"
            ),
            "shared_adapter_plastic_across_tasks": (
                variant.patch_adapter != "none"
            ),
            "pixel_decoder": variant.pixel_decoder,
        },
        "actor_critic": {
            "topology": "per_task_bank",
            "network": "DreamerV3 MLP actor and critic",
            "optimizer_state_shared": False,
            "new_task_initialization": (
                "fresh independent weights"
                if variant.full_task_experts
                else "copy previous actor-critic weights; fresh optimizer"
            ),
            "current_task_update_fraction": variant.current_task_fraction,
            "old_task_allocation": (
                "zero"
                if variant.full_task_experts
                else "uniform across replay-available old tasks"
            ),
            "old_task_parameters_frozen": variant.full_task_experts,
            "learning_rate": config.get("ac_lr", 1e-4),
            "entropy_scale": config.get("ac_entropy_scale", 3e-4),
            "total_updates_unchanged": batch_profile is None,
            "total_context_frame_uses_unchanged": True,
        },
        "observation": {
            "encoder": (
                "task-banked DreamerV3 CNN"
                if variant.observation_encoder == "cnn"
                else "frozen DINOv3 ViT-S/16"
            ),
            "encoder_topology": (
                "per_task_bank"
                if variant.task_banked_image_encoder
                else "shared"
            ),
            "encoder_parameters_per_task": (
                _cnn_encoder_parameter_count(
                    img_channels=3,
                    channels=source_config["cnn_depth"],
                )
                if variant.observation_encoder == "cnn"
                else 0
            ),
            "allocated_encoder_parameters": (
                _cnn_encoder_parameter_count(
                    img_channels=3,
                    channels=source_config["cnn_depth"],
                )
                * allocated_experts
                if variant.task_banked_image_encoder
                else 0
            ),
            "model_id": (
                None if variant.observation_encoder == "cnn" else DINOV3_MODEL_ID
            ),
            "model_artifact": model_artifact,
            "input_size": (
                source_config["img_size"]
                if variant.observation_encoder == "cnn"
                else DINOV3_INPUT_SIZE
            ),
            "feature_mode": (
                "flattened_conv_grid"
                if variant.observation_encoder == "cnn"
                else "patch_grid"
            ),
            "patch_pool_size": (
                None
                if variant.observation_encoder == "cnn"
                else variant.patch_pool_size
            ),
            "patch_feature_dim": (
                None
                if variant.observation_encoder == "cnn"
                else variant.patch_feature_dim
            ),
            "patch_projection": (
                None
                if variant.observation_encoder == "cnn"
                else variant.patch_projection
            ),
            "patch_projection_seed": (
                None
                if variant.observation_encoder == "cnn"
                else variant.patch_projection_seed
            ),
            "patch_projection_frames": (
                None if variant.observation_encoder == "cnn" else 0
            ),
            "feature_dim": feature_dim,
            "posterior_embedding_dim": posterior_embedding_dim,
            "posterior_parameters_per_task": posterior_parameters_per_task,
            "unadapted_posterior_parameters_per_task": (
                unadapted_posterior_parameters_per_task
            ),
            "patch_adapter": patch_adapter,
            "replay_feature_mode": (
                "not_applicable"
                if variant.observation_encoder == "cnn"
                else variant.replay_feature_mode
            ),
            "objective": variant.observation_description,
            "feature_loss": (
                "not_applicable" if variant.pixel_decoder else variant.feature_loss
            ),
            "pixel_decoder": variant.pixel_decoder,
            "task1_fitted_visual_projection": False,
        },
        "collection": {
            "random_policy_config": variant.random_policy,
            "new_task_first_epoch_policy": (
                "random" if variant.random_policy == "new" else "warm-started actor"
            ),
            "random_collection_epochs": random_collection_epochs,
            "extra_collections_per_random_epoch": extra_collections_per_random_epoch,
        },
        "replay": {
            "capacity_and_sampling": "ARROW-50 base allocation unchanged",
            "storage_device": (
                "cpu_addressable_file_mmap"
                if variant.code_id
                in {
                    "cnn_fullbank_arrow",
                    "dino_patchbank_arrow",
                    "dino_convbank_arrow",
                }
                else "cpu"
            ),
            "base_storage": base_replay_storage,
            "sampled_observation_dtype": "float32",
            "observation_decode": (
                "transfer uint8 to the training device, then convert to float32 "
                "and divide by 255"
                if config["replay_observation_dtype"] == "uint8"
                else "stored float32 values are returned unchanged"
            ),
            "feature_cache": feature_cache,
            "task_id_storage_bytes": task_id_storage_bytes,
            "task_sampling": (
                "current-task-only conditional uniform sequence sampling"
                if variant.full_task_experts
                else (
                    "fixed current/old update allocation, then conditional uniform "
                    "sequence sampling inside the selected task"
                )
            ),
            "subbuffer_selection": (
                "ARROW-50 weights renormalized only if a subbuffer lacks the selected task"
            ),
        },
        "residual_correction": "none",
        "extra_gradient_updates": 0,
        "extra_environment_interactions": extra_environment_interactions,
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
            "distributed_torch_seed_rule": (
                "base_seed + rank * 1000003; rank 0 matches single-GPU stream"
            ),
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

    if requires_dinov3 and dependency_versions != DINOV3_DEPENDENCIES:
        raise RuntimeError(
            f"{variant.method} requires pinned DINOv3 dependencies: "
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
