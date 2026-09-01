#!/usr/bin/env python3
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

ROOT = Path(__file__).resolve().parents[1]
ARROW_ROOT = ROOT / "third_party" / "arrow"
UPSTREAM_COMMIT = "cb05e7d97ed83c3cf6e528960db0da6868e29232"
DREAMERV3_REPVAL_REFERENCE_COMMIT = "e3f02248693a79dc8b0ebd62c93683888ddaccfe"
R2_DREAMER_COMMIT = "546e4fab8146ea4b14e1d7726bbc1a8a1d50322f"
R2_BARLOW_LOSS_SCALE = 0.05
R2_REDUNDANCY_SCALE = 5e-4
R2_NORMALIZATION_EPS = 1e-8
CONFIG_NAME = (
    "ALE_MsPacman,ALE_Boxing,ALE_CrazyClimber,ALE_Frostbite,"
    "ALE_Seaquest,ALE_Enduro-s{seed}-arrow.json"
)
SINGLE_TASK_CONFIGS = (
    ("ALE_MsPacman", "ALE/MsPacman-v5"),
    ("ALE_Boxing", "ALE/Boxing-v5"),
    ("ALE_CrazyClimber", "ALE/CrazyClimber-v5"),
    ("ALE_Frostbite", "ALE/Frostbite-v5"),
    ("ALE_Seaquest", "ALE/Seaquest-v5"),
    ("ALE_Enduro", "ALE/Enduro-v5"),
)
CURRICULUM_DIRS = {
    "original": "Original Order",
    "reversed": "Reversed Order",
    "two-cycle": "Two-Cycle Training",
}
SEEDS = [123456789, 1337, 31337, 42, 987654321]
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
KAN_ACTOR_NETWORKS = frozenset(
    {
        "relu_kan",
        "relu_kan_bounded",
        "relu_kan_adaptive",
        "fast_kan_ac",
        "fast_kan_ac_param_matched",
        "fast_kan_ac_stable",
    }
)
FASTKAN_AC_EPOCHS = 68
FASTKAN_AC_PARAM_MATCHED_EPOCHS = 136
FASTKAN_AC_STABLE_EPOCHS = 90
FASTKAN_AC_PAPER_ENVIRONMENT_STEPS = 1_100_000
FASTKAN_AC_CONFIG_OVERRIDES = {
    "actor_network": "fast_kan_ac",
    "fastkan_hidden_features": 34,
    "fastkan_hidden_layers": 3,
    "fastkan_grid_size": 8,
    "fastkan_input_min": -2.0,
    "fastkan_input_max": 2.0,
    "fastkan_rms_norm_epsilon": 1e-4,
    "fastkan_actor_output_scale": 0.01,
    "fastkan_actor_unimix": 0.01,
    "ac_optimizer": "laprop",
    "ac_lr": 4e-5,
    "ac_fresh_lr": 4e-5,
    "ac_optimizer_eps": 1e-20,
    "ac_optimizer_beta1": 0.9,
    "ac_optimizer_beta2": 0.999,
    "ac_optimizer_warmup_steps": 1000,
    "ac_agc_clip": 0.3,
    "ac_grad_clip": 0.0,
    "ac_dream_steps": 15,
    "ac_discount": 1.0 - 1.0 / 333.0,
    "ac_lambda": 0.95,
    "ac_entropy_scale": 3e-4,
    "ac_return_norm_decay": 0.99,
    "ac_persistent_return_norm": True,
    "ac_slow_critic_regularizer": 1.0,
    "ac_slow_critic_decay": 0.98,
    "ac_replay_critic_loss_scale": 0.0,
    "ac_use_slow_critic_targets": False,
    "ac_corrected_imagination_bootstrap": False,
}
FASTKAN_AC_PARAM_MATCHED_CONFIG_OVERRIDES = {
    **FASTKAN_AC_CONFIG_OVERRIDES,
    "actor_network": "fast_kan_ac_param_matched",
    "fastkan_hidden_features": 53,
    "ac_replay_critic_loss_scale": 0.3,
}
FASTKAN_AC_STABLE_CONFIG_OVERRIDES = {
    **FASTKAN_AC_PARAM_MATCHED_CONFIG_OVERRIDES,
    "actor_network": "fast_kan_ac_stable",
    "ac_use_slow_critic_targets": True,
    "ac_corrected_imagination_bootstrap": True,
}
KAN_ACTOR_METADATA = {
    "relu_kan": {
        "method": "ARROW-KANActor-50",
        "output_prefix": "arrow_kan_actor_ar50",
        "hidden_adapter": "none",
        "hidden_adapter_layer_norm_epsilon": None,
        "grid_trainable": False,
        "anchor_parameterization": None,
        "anchor_parameters": 0,
        "trainable_parameters": 795_730,
        "critic_network": "mlp",
        "critic_trainable_parameters": 917_759,
    },
    "relu_kan_bounded": {
        "method": "ARROW-KANActorBounded-50",
        "output_prefix": "arrow_kan_actor_bounded_ar50",
        "hidden_adapter": "layer_norm_sigmoid",
        "hidden_adapter_layer_norm_epsilon": 1e-3,
        "grid_trainable": False,
        "anchor_parameterization": None,
        "anchor_parameters": 0,
        "trainable_parameters": 795_858,
        "critic_network": "mlp",
        "critic_trainable_parameters": 917_759,
    },
    "relu_kan_adaptive": {
        "method": "ARROW-KANActorAdaptive-50",
        "output_prefix": "arrow_kan_actor_adaptive_ar50",
        "hidden_adapter": "layer_norm_sigmoid",
        "hidden_adapter_layer_norm_epsilon": 1e-3,
        "grid_trainable": True,
        "anchor_parameterization": "per_input_start_softplus_width",
        "anchor_parameters": 25_600,
        "trainable_parameters": 821_458,
        "critic_network": "mlp",
        "critic_trainable_parameters": 917_759,
    },
    "fast_kan_ac": {
        "method": "ARROW-FastKANAC-KDAligned-50",
        "output_prefix": "arrow_fastkan_ac_kd_aligned_ar50",
        "hidden_adapter": "rms_norm_per_fastkan_layer",
        "hidden_adapter_layer_norm_epsilon": 1e-4,
        "grid_trainable": False,
        "anchor_parameterization": "fixed_uniform_gaussian_centers",
        "anchor_parameters": 0,
        "trainable_parameters": 498_090,
        "critic_network": "fast_kan",
        "critic_trainable_parameters": 570_849,
    },
    "fast_kan_ac_param_matched": {
        "method": "ARROW-FastKANAC-ParamMatchedRepVal-50",
        "output_prefix": "arrow_fastkan_ac_param_matched_repval_ar50",
        "hidden_adapter": "rms_norm_per_fastkan_layer",
        "hidden_adapter_layer_norm_epsilon": 1e-4,
        "grid_trainable": False,
        "anchor_parameterization": "fixed_uniform_gaussian_centers",
        "anchor_parameters": 0,
        "trainable_parameters": 793_692,
        "critic_network": "fast_kan",
        "critic_trainable_parameters": 906_978,
    },
    "fast_kan_ac_stable": {
        "method": "ARROW-FastKANAC-StableTargets-50",
        "output_prefix": "arrow_fastkan_ac_stable_targets_ar50",
        "hidden_adapter": "rms_norm_per_fastkan_layer",
        "hidden_adapter_layer_norm_epsilon": 1e-4,
        "grid_trainable": False,
        "anchor_parameterization": "fixed_uniform_gaussian_centers",
        "anchor_parameters": 0,
        "trainable_parameters": 793_692,
        "critic_network": "fast_kan",
        "critic_trainable_parameters": 906_978,
    },
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the canonical ARROW-50 continual Atari method"
    )
    parser.add_argument("--seed", type=int, choices=range(5), default=0)
    parser.add_argument("--curriculum", choices=CURRICULUM_DIRS, default="original")
    parser.add_argument(
        "--single-task-index",
        type=int,
        choices=range(len(SINGLE_TASK_CONFIGS)),
        help=(
            "Run the corresponding published Atari single-task ARROW config "
            "instead of a continual curriculum. Indices follow the paper order: "
            "MsPacman, Boxing, CrazyClimber, Frostbite, Seaquest, Enduro."
        ),
    )
    parser.add_argument(
        "--observation-objective",
        choices=["reconstruction", "r2"],
        default="reconstruction",
        help=(
            "Use the published pixel-reconstruction objective or the decoder-free "
            "R2-Dreamer latent Barlow Twins objective"
        ),
    )
    parser.add_argument(
        "--actor-network",
        choices=[
            "mlp",
            "relu_kan",
            "relu_kan_bounded",
            "relu_kan_adaptive",
            "fast_kan_ac",
            "fast_kan_ac_param_matched",
            "fast_kan_ac_stable",
        ],
        default="mlp",
        help=(
            "Keep the MLP actor, select an actor-only ReLU-KAN variant, or replace "
            "both behavior heads with a named FastKAN actor-critic protocol"
        ),
    )
    parser.add_argument(
        "--task-prefix-length",
        type=int,
        choices=[1, 2, 3],
        help=(
            "Run only the first one, two, or three tasks as a named trainability or "
            "continual pilot; task order and per-task duration remain unchanged"
        ),
    )
    parser.add_argument(
        "--task-duration-epochs",
        type=_positive_int,
        help=(
            "Set the duration of a named one-task KAN trainability pilot. The "
            "sequential task boundary moves with the total duration."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Persistent run directory. Defaults to runs/arrow_ar50_<curriculum>_"
            "s<seed>_analysis under the repository."
        ),
    )
    parser.add_argument(
        "--profile-stages",
        action="store_true",
        help="Print synchronized per-stage timing",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--cpu-threads",
        type=_positive_int,
        help="Limit CPU thread pools and record the setting in the launch manifest",
    )
    parser.add_argument(
        "--replay-device",
        choices=["cuda", "cpu"],
        default="cuda",
        help=(
            "Store both full-capacity float32 ARROW replay buffers on CUDA "
            "(published config) or CPU (explicit storage-only execution profile)"
        ),
    )
    parser.add_argument(
        "--swanlab-project",
        help="Optionally mirror TensorBoard metrics to a configured SwanLab project",
    )
    parser.add_argument(
        "--swanlab-experiment-name",
        help="Optional SwanLab experiment name; no credential is accepted by this CLI",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _config_path(
    curriculum: str,
    seed: int,
    single_task_index: int | None = None,
) -> Path:
    if single_task_index is not None:
        config_stem, _ = SINGLE_TASK_CONFIGS[single_task_index]
        return (
            ARROW_ROOT
            / "Configs"
            / "Atari configs"
            / "Single-task configs"
            / f"{config_stem}-e{single_task_index}-s{seed}-arrow.json"
        )
    return (
        ARROW_ROOT
        / "Configs"
        / "Atari configs"
        / "CL-task configs"
        / CURRICULUM_DIRS[curriculum]
        / CONFIG_NAME.format(seed=seed)
    )


def _verify_primary_config(
    config_path: Path,
    curriculum: str,
    seed: int,
    single_task_index: int | None = None,
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    errors = []
    if config.get("algorithm") != "arrow":
        errors.append("algorithm must be arrow")
    if config.get("data_n_max") != 512 or config.get("data_t") != 512:
        errors.append("ARROW-50 requires data_n_max=512 and data_t=512")
    if config.get("sac_dv3_data_n_max") != 1024:
        errors.append("ARROW-50 requires a matched 1,024-trajectory total budget")
    if (
        config.get("gru_units", 512) != 512
        or config.get("mlp_features", 512) != 512
        or config.get("action_space") != 18
    ):
        errors.append(
            "KAN-Actor accounting requires gru_units=512, mlp_features=512, "
            "action_space=18"
        )
    if config.get("observation_objective", "reconstruction") != "reconstruction":
        errors.append("published ARROW config must use reconstruction before CLI override")
    r2_defaults = {
        "r2_barlow_loss_scale": R2_BARLOW_LOSS_SCALE,
        "r2_redundancy_scale": R2_REDUNDANCY_SCALE,
        "r2_normalization_eps": R2_NORMALIZATION_EPS,
    }
    for key, expected in r2_defaults.items():
        if key in config and config[key] != expected:
            errors.append(f"published ARROW config has unexpected {key}={config[key]!r}")
    replay_types = [item.get("rb_type") for item in config.get("replay_buffers", [])]
    if replay_types != ["FifoReplay", "LongTermReplay"]:
        errors.append("replay buffers must be FIFO followed by LTDM")
    replay_devices = [
        item.get("rb_device") for item in config.get("replay_buffers", [])
    ]
    if replay_devices != ["cuda", "cuda"]:
        errors.append("published ARROW replay buffers must be CUDA-resident")
    if config.get("replay_observation_dtype", "float32") != "float32":
        errors.append("published ARROW replay observations must use float32")
    if config.get("seed") != SEEDS[seed]:
        errors.append("numeric seed does not match the published seed ID")
    envs = config.get("esc", {}).get("env_configs", [])
    if single_task_index is None:
        if len(envs) != 6:
            errors.append("continual Atari config must contain six tasks")
        swap_sched = config.get("esc", {}).get("kwargs", {}).get("swap_sched")
        expected_swap = 45 if curriculum == "two-cycle" else 90
        if swap_sched != expected_swap:
            errors.append(f"swap_sched must be {expected_swap} for {curriculum}")
    else:
        _, expected_environment = SINGLE_TASK_CONFIGS[single_task_index]
        if len(envs) != 1 or envs[0].get("name") != expected_environment:
            errors.append(
                "single-task config must contain only " f"{expected_environment}"
            )
        if config.get("esc", {}).get("env_schedule_type") != "AllEnvironments":
            errors.append("single-task config must use AllEnvironments")
        if config.get("epochs") != 91:
            errors.append("published Atari single-task config must use epochs=91")
        if config.get("esc", {}).get("kwargs") not in ({}, None):
            errors.append("single-task config must not define a task-swap schedule")
    if errors:
        raise RuntimeError("Invalid primary ARROW config: " + "; ".join(errors))
    return config


def _arrow_replay_storage_budget(config: dict) -> dict:
    """Exact persistent tensor allocation for the configured replay dtypes."""
    slots_per_buffer = config["data_n_max"]
    sequence_length = config["data_t"]
    transitions_per_buffer = slots_per_buffer * sequence_length
    observation_elements = 3 * config["img_size"] * config["img_size"]
    auxiliary_elements = config["action_space"] + 3
    observation_dtype = config.get("replay_observation_dtype", "float32")
    try:
        observation_bytes_per_element = {
            "float32": 4,
            "uint8": 1,
        }[observation_dtype]
    except KeyError as exc:
        raise ValueError(
            f"Unknown replay observation dtype: {observation_dtype!r}"
        ) from exc
    auxiliary_bytes_per_element = 4
    observation_bytes_per_buffer = (
        transitions_per_buffer
        * observation_elements
        * observation_bytes_per_element
    )
    tensor_bytes_per_buffer = (
        observation_bytes_per_buffer
        + transitions_per_buffer
        * auxiliary_elements
        * auxiliary_bytes_per_element
    )
    replay_devices = {
        replay["rb_type"]: replay["rb_device"] for replay in config["replay_buffers"]
    }
    buffers = {
        "fifo": {
            "slots": slots_per_buffer,
            "device": replay_devices["FifoReplay"],
            "observation_bytes": observation_bytes_per_buffer,
            "allocated_tensor_bytes": tensor_bytes_per_buffer,
        },
        "ltdm": {
            "slots": slots_per_buffer,
            "device": replay_devices["LongTermReplay"],
            "observation_bytes": observation_bytes_per_buffer,
            "allocated_tensor_bytes": tensor_bytes_per_buffer,
            "priority_index_entries": slots_per_buffer,
        },
    }
    return {
        "dtype": observation_dtype,
        "bytes_per_element": observation_bytes_per_element,
        "observation_dtype": observation_dtype,
        "observation_bytes_per_element": observation_bytes_per_element,
        "auxiliary_dtype": "float32",
        "auxiliary_bytes_per_element": auxiliary_bytes_per_element,
        "transitions": 2 * transitions_per_buffer,
        "observation_bytes": 2 * observation_bytes_per_buffer,
        "allocated_tensor_bytes": 2 * tensor_bytes_per_buffer,
        "buffers": buffers,
        "python_index_bytes_included": False,
        "actor_comparison_difference_bytes": 0,
    }


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    single_task = args.single_task_index is not None
    if single_task and (
        args.task_prefix_length is not None or args.task_duration_epochs is not None
    ):
        parser.error(
            "--single-task-index cannot be combined with task-prefix or task-duration "
            "overrides"
        )
    if single_task and (
        args.actor_network != "mlp" or args.observation_objective != "reconstruction"
    ):
        parser.error(
            "--single-task-index is reserved for the published ARROW-50 MLP "
            "reconstruction baseline"
        )
    if (
        args.actor_network in KAN_ACTOR_NETWORKS
        and args.observation_objective != "reconstruction"
    ):
        parser.error("KAN-Actor must be tested independently from the R2 ablation")
    if (
        args.task_prefix_length is not None
        and args.observation_objective != "reconstruction"
    ):
        parser.error("Task-prefix pilots must be tested independently from the R2 ablation")
    if args.task_duration_epochs is not None:
        if args.task_prefix_length != 1:
            parser.error("--task-duration-epochs requires --task-prefix-length 1")
        if args.actor_network not in {
            "relu_kan_bounded",
            "relu_kan_adaptive",
            "fast_kan_ac",
            "fast_kan_ac_param_matched",
            "fast_kan_ac_stable",
        }:
            parser.error(
                "--task-duration-epochs requires a named trainable KAN protocol"
            )
    if args.actor_network == "fast_kan_ac" and (
        args.task_prefix_length != 1
        or args.task_duration_epochs != FASTKAN_AC_EPOCHS
    ):
        parser.error(
            "fast_kan_ac currently requires --task-prefix-length 1 and "
            f"--task-duration-epochs {FASTKAN_AC_EPOCHS}"
        )
    if args.actor_network == "fast_kan_ac_param_matched" and (
        args.task_prefix_length != 1
        or args.task_duration_epochs != FASTKAN_AC_PARAM_MATCHED_EPOCHS
    ):
        parser.error(
            "fast_kan_ac_param_matched currently requires --task-prefix-length 1 "
            f"and --task-duration-epochs {FASTKAN_AC_PARAM_MATCHED_EPOCHS}"
        )
    stable_fastkan_t1 = (
        args.task_prefix_length == 1
        and args.task_duration_epochs == FASTKAN_AC_STABLE_EPOCHS
    )
    stable_fastkan_full_curriculum = (
        args.task_prefix_length is None and args.task_duration_epochs is None
    )
    if args.actor_network == "fast_kan_ac_stable" and not (
        stable_fastkan_t1 or stable_fastkan_full_curriculum
    ):
        parser.error(
            "fast_kan_ac_stable requires either the frozen full curriculum without "
            "task overrides or --task-prefix-length 1 and "
            f"--task-duration-epochs {FASTKAN_AC_STABLE_EPOCHS}"
        )
    if args.swanlab_experiment_name is not None and args.swanlab_project is None:
        parser.error("--swanlab-experiment-name requires --swanlab-project")
    project_git = (
        git_state(ROOT) if args.dry_run else require_synced_training_git_state(ROOT)
    )
    python = args.python.resolve()
    config_path = _config_path(
        args.curriculum,
        args.seed,
        args.single_task_index,
    )
    config = _verify_primary_config(
        config_path,
        args.curriculum,
        args.seed,
        args.single_task_index,
    )
    if args.actor_network in KAN_ACTOR_NETWORKS:
        output_prefix = KAN_ACTOR_METADATA[args.actor_network]["output_prefix"]
    elif args.observation_objective == "r2":
        output_prefix = "arrow_r2rep_ar50"
    else:
        output_prefix = "arrow_ar50"
    if args.replay_device == "cpu":
        output_prefix += "_cpu_fp32_replay"
    if single_task:
        config_stem, _ = SINGLE_TASK_CONFIGS[args.single_task_index]
        output_prefix += (
            f"_single_task_e{args.single_task_index}_"
            f"{config_stem.removeprefix('ALE_').lower()}"
        )
    if args.task_prefix_length is not None:
        output_prefix += f"_t{args.task_prefix_length}_pilot"
    if args.task_duration_epochs is not None:
        output_prefix += f"_e{args.task_duration_epochs}"
    run_schedule_label = "single_task" if single_task else args.curriculum
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else ROOT / "runs" / f"{output_prefix}_{run_schedule_label}_s{args.seed}_analysis"
    )
    snapshot_dir = output_dir / "analysis_snapshots"
    env = os.environ.copy()
    thread_env = {}
    if args.cpu_threads is not None:
        thread_env = {key: str(args.cpu_threads) for key in THREAD_ENV_KEYS}
        env.update(thread_env)
    # The vendored Atari trainer imports project-owned runtime helpers even for
    # the unmodified ARROW-50 path.  Make the package source explicit rather
    # than relying on an editable install or the caller's working directory.
    project_pythonpath = str(ROOT / "src")
    inherited_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (project_pythonpath, inherited_pythonpath) if part
    )

    swap_sched = (
        config["epochs"] - 1
        if single_task
        else config["esc"]["kwargs"]["swap_sched"]
    )
    task_duration_epochs = args.task_duration_epochs or swap_sched
    if (
        args.task_duration_epochs is not None
        and task_duration_epochs <= swap_sched
        and args.actor_network
        not in {"fast_kan_ac", "fast_kan_ac_param_matched", "fast_kan_ac_stable"}
    ):
        parser.error(
            "--task-duration-epochs must exceed the frozen 90-epoch task duration"
        )
    training_epochs = (
        config["epochs"]
        if args.task_prefix_length is None
        else task_duration_epochs * args.task_prefix_length
    )
    adaptive_kan = args.actor_network == "relu_kan_adaptive"
    fastkan_ac = args.actor_network in {
        "fast_kan_ac",
        "fast_kan_ac_param_matched",
        "fast_kan_ac_stable",
    }
    extended_fastkan_ac = args.actor_network == "fast_kan_ac_param_matched"
    stable_fastkan_ac = args.actor_network == "fast_kan_ac_stable"
    full_stable_fastkan_ac = stable_fastkan_ac and args.task_prefix_length is None
    parameter_matched_fastkan_ac = extended_fastkan_ac or stable_fastkan_ac
    resolved_training_config = None
    launch_config_path = config_path
    config_overrides = {}
    if (
        adaptive_kan
        or fastkan_ac
        or args.task_duration_epochs is not None
        or args.replay_device == "cpu"
    ):
        resolved_training_config = json.loads(json.dumps(config))
        if args.replay_device == "cpu":
            for replay_config in resolved_training_config["replay_buffers"]:
                replay_config["rb_device"] = "cpu"
            resolved_training_config["replay_observation_dtype"] = "float32"
            config_overrides.update(
                {
                    "replay_buffers": resolved_training_config["replay_buffers"],
                    "replay_observation_dtype": "float32",
                }
            )
        if adaptive_kan:
            resolved_training_config["actor_network"] = args.actor_network
            resolved_training_config["actor_kan_trainable_grid"] = True
            config_overrides.update(
                {
                    "actor_network": args.actor_network,
                    "actor_kan_trainable_grid": True,
                }
            )
        if fastkan_ac:
            fastkan_overrides = {
                "fast_kan_ac": FASTKAN_AC_CONFIG_OVERRIDES,
                "fast_kan_ac_param_matched": (
                    FASTKAN_AC_PARAM_MATCHED_CONFIG_OVERRIDES
                ),
                "fast_kan_ac_stable": FASTKAN_AC_STABLE_CONFIG_OVERRIDES,
            }[args.actor_network]
            resolved_training_config.update(fastkan_overrides)
            config_overrides.update(fastkan_overrides)
        if args.task_duration_epochs is not None:
            resolved_training_config["epochs"] = training_epochs
            resolved_training_config["esc"]["kwargs"]["swap_sched"] = (
                task_duration_epochs
            )
            config_overrides.update(
                {
                    "epochs": training_epochs,
                    "esc.kwargs.swap_sched": task_duration_epochs,
                }
            )
        launch_config_path = output_dir / "resolved_training_config.json"

    effective_config = resolved_training_config or config

    command = [
        str(python),
        "Code/ARROW_and_DV3/Atari/train.py",
        "--config",
        str(launch_config_path),
        "--arrow-replay-ratio",
        "50-50",
        "--log-dir",
        str(output_dir),
        "--analysis-snapshot-dir",
        str(snapshot_dir),
    ]
    if args.observation_objective == "r2":
        command.extend(
            (
                "--observation-objective",
                "r2",
                "--r2-barlow-loss-scale",
                str(R2_BARLOW_LOSS_SCALE),
                "--r2-redundancy-scale",
                str(R2_REDUNDANCY_SCALE),
                "--r2-normalization-eps",
                str(R2_NORMALIZATION_EPS),
            )
        )
    if args.actor_network in KAN_ACTOR_NETWORKS:
        command.extend(("--actor-network", args.actor_network))
    if adaptive_kan:
        command.append("--actor-kan-trainable-grid")
    if args.task_prefix_length is not None:
        command.extend(("--epochs", str(training_epochs), "--evaluate-final"))
    milestone_completed_epochs = [68] if extended_fastkan_ac else []
    for milestone_completed_epoch in milestone_completed_epochs:
        command.extend(
            ("--milestone-completed-epoch", str(milestone_completed_epoch))
        )
    if args.swanlab_project is not None:
        command.extend(("--swanlab-project", args.swanlab_project))
    if args.swanlab_experiment_name is not None:
        command.extend(("--swanlab-experiment-name", args.swanlab_experiment_name))
    if args.profile_stages:
        command.append("--profile-stages")
    command.extend(("--compile-world-model", "--fused-adam", "--tf32"))
    boundary_epochs = (
        [training_epochs - 1]
        if single_task
        else list(
            range(task_duration_epochs - 1, training_epochs, task_duration_epochs)
        )
    )
    is_r2 = args.observation_objective == "r2"
    is_kan_actor = args.actor_network in KAN_ACTOR_NETWORKS
    if is_kan_actor:
        method = KAN_ACTOR_METADATA[args.actor_network]["method"]
        role = "actor-architecture-ablation"
    elif is_r2:
        method = "ARROW-R2Rep-50"
        role = "representation-objective-ablation"
    else:
        method = "ARROW-50"
        role = "primary-method"
    if single_task:
        method += "-SingleTask"
        role = "single-task-normalization-reproduction"
    elif args.task_prefix_length is not None:
        if args.task_prefix_length == 1 and is_kan_actor:
            if args.task_duration_epochs is None:
                method += "-T1TrainabilityPilot"
                role = "actor-trainability-pilot"
            else:
                method += f"-T1-{task_duration_epochs}EpochTrainabilityPilot"
                role = "actor-trainability-budget-extension"
        else:
            method += f"-T{args.task_prefix_length}Pilot"
            role = (
                "actor-architecture-pilot"
                if is_kan_actor
                else "matched-short-pilot-control"
            )
    if full_stable_fastkan_ac:
        role = "actor-critic-continual-retention-pilot"
    elif stable_fastkan_ac:
        role = "actor-critic-stable-target-correction-pilot"
    elif extended_fastkan_ac:
        role = "actor-critic-param-matched-replay-value-budget-extension"
    elif fastkan_ac:
        role = "actor-critic-kan-dreamer-aligned-pilot"

    decisions_per_regular_epoch = config["n_sync"] * config["gen_seq_len"]
    collection_epoch_equivalents = training_epochs
    if config.get("pretrain_enabled", True):
        collection_epoch_equivalents += config.get("pretrain_data_multiplier", 4) - 1
    agent_decisions = decisions_per_regular_epoch * collection_epoch_equivalents
    raw_environment_frames = agent_decisions * config["env_repeat"]
    launch = {
        "method": method,
        "role": role,
        "runtime": (
            "vendored-optimized-cpu-float32-replay"
            if args.replay_device == "cpu"
            else "vendored-optimized"
        ),
        "started_at_utc": None,
        "project_git": project_git,
        "profile_stages": args.profile_stages,
        "optimizations": [
            "distribution-free-categorical-kernels",
            "compiled-world-model-loss",
            "fused-adam",
            "tf32-matmul",
            "set-to-none-gradients",
            *(
                ["cpu-resident-float32-replay"]
                if args.replay_device == "cpu"
                else []
            ),
        ],
        "upstream_commit": UPSTREAM_COMMIT,
        "source": str(ARROW_ROOT),
        "config": str(launch_config_path),
        "source_config": str(config_path),
        "resolved_training_config": (
            str(launch_config_path) if resolved_training_config is not None else None
        ),
        "config_overrides": config_overrides,
        "output_dir": str(output_dir),
        "analysis_snapshot_dir": str(snapshot_dir),
        "model_parameter_accounting": str(output_dir / "model_parameter_accounting.json"),
        "actor_critic_parameter_accounting": str(
            output_dir / "actor_critic_parameter_accounting.json"
        ),
        "analysis_snapshot_semantics": {
            "artifact_kind": "analysis_snapshot",
            "resumable": False,
            "task_boundary_epochs": boundary_epochs,
            "milestone_completed_epochs": milestone_completed_epochs,
            "final_epoch": training_epochs - 1,
            "final_coincides_with_task_boundary": (
                (training_epochs - 1) in boundary_epochs
            ),
            "omitted_state": [
                "optimizers",
                "replay",
                "RNG",
                "environment schedule",
                *(
                    ["slow critic target", "return-normalizer EMA"]
                    if fastkan_ac
                    else []
                ),
            ],
        },
        "training_scope": {
            "single_task_index": args.single_task_index,
            "task_prefix_length": args.task_prefix_length,
            "epochs": training_epochs,
            "task_duration_epochs": task_duration_epochs,
            "baseline_task_duration_epochs": swap_sched,
            "task_duration_epoch_override": args.task_duration_epochs,
            "full_curriculum": args.task_prefix_length is None and not single_task,
            "tasks": [
                task["name"]
                for task in config["esc"]["env_configs"][
                    : args.task_prefix_length
                    if args.task_prefix_length is not None
                    else None
                ]
            ],
            "agent_decisions": agent_decisions,
            "raw_environment_frames": raw_environment_frames,
            "agent_decisions_per_regular_epoch": decisions_per_regular_epoch,
            "epoch_zero_collection_multiplier": (
                config.get("pretrain_data_multiplier", 4)
                if config.get("pretrain_enabled", True)
                else 1
            ),
            "kan_dreamer_target_environment_steps": (
                FASTKAN_AC_PAPER_ENVIRONMENT_STEPS if fastkan_ac else None
            ),
            "kan_dreamer_step_mapping": (
                {
                    "mapped_counter": "ARROW agent decisions",
                    "target": FASTKAN_AC_PAPER_ENVIRONMENT_STEPS,
                    "actual": agent_decisions,
                    "relative_difference": (
                        agent_decisions / FASTKAN_AC_PAPER_ENVIRONMENT_STEPS - 1.0
                    ),
                    "raw_atari_frames_are_not_equated_to_dmc_steps": True,
                }
                if fastkan_ac
                else None
            ),
            "midpoint_completed_epochs": (
                FASTKAN_AC_EPOCHS if extended_fastkan_ac else None
            ),
            "midpoint_agent_decisions": (
                FASTKAN_AC_EPOCHS * decisions_per_regular_epoch
                if extended_fastkan_ac
                else None
            ),
        },
        "continual_retention_evaluation": (
            {
                "hypothesis": (
                    "The stable FastKAN actor-critic retains previously learned "
                    "behavior better than the matched ARROW-50 MLP actor-critic."
                ),
                "comparison_method": "ARROW-50",
                "comparison_seed_id": args.seed,
                "comparison_seed": config["seed"],
                "same_curriculum_interaction_update_and_replay_budgets": True,
                "task_identity_exposed_to_agent": False,
                "evaluation_isolated_from_training_and_replay": True,
                "periodic_evaluation_epochs": list(range(0, training_epochs, 10)),
                "acquisition_evaluation_epochs": list(
                    range(task_duration_epochs, training_epochs, task_duration_epochs)
                ),
                "final_comparable_evaluation_epoch": training_epochs - 1,
                "raw_return_metric": "Perf/eval_raw_return_mean",
                "per_task_forgetting_definition": (
                    "maximum periodic raw return from the task acquisition "
                    "evaluation through epoch 540 minus the epoch-540 raw return"
                ),
                "backward_transfer_definition": (
                    "epoch-540 raw return minus the task acquisition-evaluation "
                    "raw return"
                ),
                "aggregate_normalization": (
                    "derived separately with fixed cited constants; raw per-task "
                    "returns remain the source metrics"
                ),
                "multiple_seeds_required_for_claim": True,
            }
            if full_stable_fastkan_ac
            else None
        ),
        "curriculum": "single-task" if single_task else args.curriculum,
        "seed_id": args.seed,
        "seed": config["seed"],
        "determinism": {
            "python_random_seed": config["seed"],
            "numpy_seed": config["seed"],
            "torch_cpu_cuda_seed": config["seed"],
            "replay_buffer_selection_rng": "python_random",
            "replay_within_buffer_rng": "numpy",
            "environment_seed_streams": {
                "collection": "dedicated_numpy_generator",
                "evaluation": "dedicated_numpy_generator",
                "derivation": "SeedSequence(run_seed).spawn(2)",
                "per_worker_reset_and_action_derivation": "SeedSequence(call_seed).spawn(2)",
            },
            "environment_reset_seeded": True,
            "action_space_seeded": True,
            "evaluation_rng_state_restored": True,
            "torch_deterministic_algorithms": False,
            "tf32_enabled": True,
            "known_nondeterminism": [
                "CUDA kernels are not forced into deterministic-only mode",
            ],
        },
        "fifo_slots": 512,
        "ltdm_slots": 512,
        "sequence_length": 512,
        "replay_buffer_selection": {"fifo": 0.5, "ltdm": 0.5},
        "replay_execution_profile": {
            "storage_device": args.replay_device,
            "observation_dtype": "float32",
            "published_storage_device": "cuda",
            "storage_device_changed_from_published_config": (
                args.replay_device != "cuda"
            ),
            "capacity_unchanged": True,
            "fifo_ltdm_retention_unchanged": True,
            "buffer_selection_probability_unchanged": True,
            "sampled_tensor_values_and_dtype_unchanged": True,
            "minibatches_transferred_to_cuda_after_sampling": (
                args.replay_device == "cpu"
            ),
        },
        "replay_storage_budget": _arrow_replay_storage_budget(effective_config),
        "actor": {
            "network": args.actor_network,
            "critic_network": (
                KAN_ACTOR_METADATA[args.actor_network]["critic_network"]
                if is_kan_actor
                else "mlp"
            ),
            "input_features": 1536,
            "action_features": config["action_space"],
            "recurrent_features": config["gru_units"],
            "kan_hidden_features": (
                53
                if parameter_matched_fastkan_ac
                else 34
                if fastkan_ac
                else 64
                if is_kan_actor
                else None
            ),
            "kan_hidden_layers": 3 if fastkan_ac else None,
            "kan_grid_size": 8 if fastkan_ac else 5 if is_kan_actor else None,
            "kan_spline_order": None if fastkan_ac else 3 if is_kan_actor else None,
            "kan_basis_count": 8 if is_kan_actor else None,
            "kan_basis": (
                "gaussian_rbf"
                if fastkan_ac
                else "relu_spline"
                if is_kan_actor
                else None
            ),
            "kan_input_range": (
                [-2.0, 2.0]
                if fastkan_ac
                else [0.0, 1.0]
                if is_kan_actor
                else None
            ),
            "kan_grid_trainable": (
                KAN_ACTOR_METADATA[args.actor_network]["grid_trainable"]
                if is_kan_actor
                else None
            ),
            "kan_anchor_parameterization": (
                KAN_ACTOR_METADATA[args.actor_network]["anchor_parameterization"]
                if is_kan_actor
                else None
            ),
            "kan_anchor_parameters": (
                KAN_ACTOR_METADATA[args.actor_network]["anchor_parameters"]
                if is_kan_actor
                else None
            ),
            "kan_normalize_recurrent_state": (
                None if fastkan_ac else True if is_kan_actor else None
            ),
            "kan_hidden_adapter": (
                KAN_ACTOR_METADATA[args.actor_network]["hidden_adapter"]
                if is_kan_actor
                else None
            ),
            "kan_hidden_adapter_layer_norm_epsilon": (
                KAN_ACTOR_METADATA[args.actor_network][
                    "hidden_adapter_layer_norm_epsilon"
                ]
                if is_kan_actor
                else None
            ),
            "kan_rms_norm_epsilon": 1e-4 if fastkan_ac else None,
            "trainable_parameters": (
                KAN_ACTOR_METADATA[args.actor_network]["trainable_parameters"]
                if is_kan_actor
                else 797_202
            ),
            "critic_trainable_parameters": (
                KAN_ACTOR_METADATA[args.actor_network][
                    "critic_trainable_parameters"
                ]
                if is_kan_actor
                else 917_759
            ),
            "combined_trainable_parameters": (
                KAN_ACTOR_METADATA[args.actor_network]["trainable_parameters"]
                + KAN_ACTOR_METADATA[args.actor_network][
                    "critic_trainable_parameters"
                ]
                if is_kan_actor
                else 1_714_961
            ),
            "mlp_combined_trainable_parameters": 1_714_961,
            "combined_parameter_difference_from_mlp": (
                KAN_ACTOR_METADATA[args.actor_network]["trainable_parameters"]
                + KAN_ACTOR_METADATA[args.actor_network][
                    "critic_trainable_parameters"
                ]
                - 1_714_961
                if is_kan_actor
                else 0
            ),
            "actor_output_scale": 0.01 if fastkan_ac else None,
            "actor_unimix": 0.01 if fastkan_ac else None,
            "critic_output_scale": 0.0 if fastkan_ac else None,
            "base_branch": "silu_linear" if fastkan_ac else None,
            "rbf_bandwidth": 4.0 / 7.0 if fastkan_ac else None,
            "implementation": (
                "project-owned-independent-pytorch" if is_kan_actor else "vendored-mlp"
            ),
            "reference": (
                "https://arxiv.org/abs/2512.07437"
                if fastkan_ac
                else "https://arxiv.org/abs/2406.02075"
                if is_kan_actor
                else None
            ),
        },
        "actor_critic_training": {
            "optimizer": "laprop" if fastkan_ac else "adam",
            "learning_rate": 4e-5 if fastkan_ac else None,
            "optimizer_epsilon": 1e-20 if fastkan_ac else None,
            "optimizer_betas": [0.9, 0.999] if fastkan_ac else None,
            "optimizer_warmup_updates": 1000 if fastkan_ac else None,
            "gradient_clipping": (
                {"type": "per_tensor_agc", "coefficient": 0.3}
                if fastkan_ac
                else {"type": "global_norm", "coefficient": 100.0}
            ),
            "imagination_horizon": 15 if fastkan_ac else 16,
            "discount_horizon": 333 if fastkan_ac else None,
            "return_lambda": 0.95,
            "entropy_regularizer": 3e-4,
            "return_normalization": {
                "percentiles": [5, 95],
                "minimum_scale": 1.0,
                "decay": 0.99,
                "persists_across_epochs": fastkan_ac,
            },
            "critic_ema_regularizer": 1.0 if fastkan_ac else 0.0,
            "critic_ema_decay": 0.98 if fastkan_ac else None,
            "critic_replay_loss_scale": (
                0.3 if parameter_matched_fastkan_ac else 0.0
            ),
            "paper_critic_replay_loss_scale": 0.3 if fastkan_ac else None,
            "critic_replay_loss_deviation": (
                "not ported because ARROW trains behavior separately from its world-model "
                "replay batches"
                if fastkan_ac and not parameter_matched_fastkan_ac
                else None
            ),
            "critic_replay_loss_semantics": (
                "TD-lambda targets over the same four posterior context frames used "
                "to initialize imagination; no extra replay minibatch is sampled"
                if parameter_matched_fastkan_ac
                else None
            ),
            "critic_replay_reward_timing": (
                "ARROW same-index reward and continuation convention"
                if parameter_matched_fastkan_ac
                else None
            ),
            "dreamerv3_repval_reference": (
                {
                    "repository": "https://github.com/danijar/dreamerv3",
                    "commit": DREAMERV3_REPVAL_REFERENCE_COMMIT,
                }
                if parameter_matched_fastkan_ac
                else None
            ),
            "imagination_value_target": (
                "ema_slow_critic" if stable_fastkan_ac else "online_critic"
            ),
            "actor_advantage_baseline": (
                "ema_slow_critic" if stable_fastkan_ac else "online_critic"
            ),
            "terminal_bootstrap_state": (
                "post_transition_imagined_state"
                if stable_fastkan_ac
                else "legacy_last_pre_transition_state"
            ),
        },
        "metric_logging": {
            "tensorboard": True,
            "actor_critic_metrics": [
                "actor_reinforce_loss",
                "actor_entropy",
                "critic_imagination_loss",
                "critic_replay_loss",
                "total_loss",
                "return_mean",
                "return_scale",
                "gradient_norm",
            ],
            "actor_critic_counter": "actor_critic_updates",
            "swanlab_enabled": args.swanlab_project is not None,
            "swanlab_project": args.swanlab_project,
            "swanlab_experiment_name": args.swanlab_experiment_name,
            "swanlab_credentials_source": (
                "external SwanLab configuration or environment"
                if args.swanlab_project is not None
                else None
            ),
        },
        "final_evaluation": {
            "enabled": args.task_prefix_length is not None,
            "path": (
                str(output_dir / "final_evaluation.json")
                if args.task_prefix_length is not None
                else None
            ),
            "seen_tasks_only": True if args.task_prefix_length is not None else None,
            "policy": "stochastic" if args.task_prefix_length is not None else None,
            "rollouts_per_task": 16 if args.task_prefix_length is not None else None,
            "enters_replay": False if args.task_prefix_length is not None else None,
            "reports_raw_and_scaled_returns": (
                True if args.task_prefix_length is not None else None
            ),
            "periodic_epoch_index_modulo": 10,
            "milestone_completed_epochs": milestone_completed_epochs,
        },
        "observation_objective": {
            "name": args.observation_objective,
            "decoder_enabled": not is_r2,
            "barlow_loss_scale": R2_BARLOW_LOSS_SCALE if is_r2 else None,
            "redundancy_scale": R2_REDUNDANCY_SCALE if is_r2 else None,
            "normalization_eps": R2_NORMALIZATION_EPS if is_r2 else None,
            "target_gradient": "stopped" if is_r2 else None,
            "sample_axes": "time*batch" if is_r2 else None,
        },
        "r2_dreamer_reference": (
            {
                "paper": "https://arxiv.org/abs/2603.18202",
                "repository": "https://github.com/NM512/r2dreamer",
                "commit": R2_DREAMER_COMMIT,
            }
            if is_r2
            else None
        ),
        "cpu_threads": args.cpu_threads,
        "environment": thread_env,
        "project_pythonpath_prepend": project_pythonpath,
        "command": command,
    }
    print(json.dumps(launch, indent=2))
    rendered_env = [f"{key}={value}" for key, value in thread_env.items()]
    if project_pythonpath is not None:
        rendered_env.append(f"PYTHONPATH={env['PYTHONPATH']}")
    print(f"command: {shlex.join([*rendered_env, *command])}")
    if args.dry_run:
        return 0

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {output_dir}")
    runtime_environment = _runtime_info(python, env)
    output_dir.mkdir(parents=True)
    if resolved_training_config is not None:
        _write_json(launch_config_path, resolved_training_config)
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
