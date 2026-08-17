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

ROOT = Path(__file__).resolve().parents[1]
ARROW_ROOT = ROOT / "third_party" / "arrow"
UPSTREAM_COMMIT = "cb05e7d97ed83c3cf6e528960db0da6868e29232"
R2_DREAMER_COMMIT = "546e4fab8146ea4b14e1d7726bbc1a8a1d50322f"
R2_BARLOW_LOSS_SCALE = 0.05
R2_REDUNDANCY_SCALE = 5e-4
R2_NORMALIZATION_EPS = 1e-8
CONFIG_NAME = (
    "ALE_MsPacman,ALE_Boxing,ALE_CrazyClimber,ALE_Frostbite,"
    "ALE_Seaquest,ALE_Enduro-s{seed}-arrow.json"
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
    {"relu_kan", "relu_kan_bounded", "relu_kan_adaptive", "fast_kan_ac"}
)
FASTKAN_AC_EPOCHS = 68
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
        ],
        default="mlp",
        help=(
            "Keep the MLP actor, select an actor-only ReLU-KAN variant, or replace "
            "both behavior heads with the KAN-Dreamer-aligned FastKAN network"
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
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _config_path(curriculum: str, seed: int) -> Path:
    return (
        ARROW_ROOT
        / "Configs"
        / "Atari configs"
        / "CL-task configs"
        / CURRICULUM_DIRS[curriculum]
        / CONFIG_NAME.format(seed=seed)
    )


def _verify_primary_config(config_path: Path, curriculum: str, seed: int) -> dict:
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
    if config.get("seed") != SEEDS[seed]:
        errors.append("numeric seed does not match the published seed ID")
    envs = config.get("esc", {}).get("env_configs", [])
    if len(envs) != 6:
        errors.append("continual Atari config must contain six tasks")
    swap_sched = config.get("esc", {}).get("kwargs", {}).get("swap_sched")
    expected_swap = 45 if curriculum == "two-cycle" else 90
    if swap_sched != expected_swap:
        errors.append(f"swap_sched must be {expected_swap} for {curriculum}")
    if errors:
        raise RuntimeError("Invalid primary ARROW config: " + "; ".join(errors))
    return config


def _arrow_replay_storage_budget(config: dict) -> dict:
    """Exact persistent tensor allocation in the vendored float32 replay."""
    slots_per_buffer = config["data_n_max"]
    sequence_length = config["data_t"]
    transitions_per_buffer = slots_per_buffer * sequence_length
    observation_elements = 3 * config["img_size"] * config["img_size"]
    auxiliary_elements = config["action_space"] + 3
    bytes_per_element = 4
    observation_bytes_per_buffer = (
        transitions_per_buffer * observation_elements * bytes_per_element
    )
    tensor_bytes_per_buffer = (
        transitions_per_buffer
        * (observation_elements + auxiliary_elements)
        * bytes_per_element
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
        "dtype": "float32",
        "bytes_per_element": bytes_per_element,
        "transitions": 2 * transitions_per_buffer,
        "observation_bytes": 2 * observation_bytes_per_buffer,
        "allocated_tensor_bytes": 2 * tensor_bytes_per_buffer,
        "buffers": buffers,
        "python_index_bytes_included": False,
        "actor_comparison_difference_bytes": 0,
    }


def _runtime_info(python: Path, env: dict[str, str]) -> dict:
    probe_code = """
import json
import os
import platform
import sys
from importlib import metadata

import torch

assert torch.cuda.is_available() and torch.cuda.device_count() >= 1
properties = torch.cuda.get_device_properties(0)
packages = (
    "ale-py",
    "gymnasium",
    "numpy",
    "opencv-python",
    "sortedcontainers",
    "tensorboard",
    "torch",
    "torchaudio",
    "torchvision",
    "tqdm",
)
print(json.dumps({
    "python": sys.version,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "cpu_count": os.cpu_count(),
    "packages": {name: metadata.version(name) for name in packages},
    "torch_cuda_build": torch.version.cuda,
    "cudnn_version": torch.backends.cudnn.version(),
    "cuda_device_count": torch.cuda.device_count(),
    "cuda_device_name": properties.name,
    "cuda_total_memory_bytes": properties.total_memory,
}))
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
    return json.loads(probe.stdout.strip())


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _run_and_tee(
    command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path
) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return process.wait()


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
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
    project_git = (
        git_state(ROOT) if args.dry_run else require_synced_training_git_state(ROOT)
    )
    python = args.python.resolve()
    config_path = _config_path(args.curriculum, args.seed)
    config = _verify_primary_config(config_path, args.curriculum, args.seed)
    if args.actor_network in KAN_ACTOR_NETWORKS:
        output_prefix = KAN_ACTOR_METADATA[args.actor_network]["output_prefix"]
    elif args.observation_objective == "r2":
        output_prefix = "arrow_r2rep_ar50"
    else:
        output_prefix = "arrow_ar50"
    if args.task_prefix_length is not None:
        output_prefix += f"_t{args.task_prefix_length}_pilot"
    if args.task_duration_epochs is not None:
        output_prefix += f"_e{args.task_duration_epochs}"
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else ROOT / "runs" / f"{output_prefix}_{args.curriculum}_s{args.seed}_analysis"
    )
    snapshot_dir = output_dir / "analysis_snapshots"
    env = os.environ.copy()
    thread_env = {}
    if args.cpu_threads is not None:
        thread_env = {key: str(args.cpu_threads) for key in THREAD_ENV_KEYS}
        env.update(thread_env)
    project_pythonpath = None
    if args.observation_objective == "r2" or args.actor_network in KAN_ACTOR_NETWORKS:
        project_pythonpath = str(ROOT / "src")
        inherited_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (project_pythonpath, inherited_pythonpath) if part
        )

    swap_sched = config["esc"]["kwargs"]["swap_sched"]
    task_duration_epochs = args.task_duration_epochs or swap_sched
    if (
        args.task_duration_epochs is not None
        and task_duration_epochs <= swap_sched
        and args.actor_network != "fast_kan_ac"
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
    fastkan_ac = args.actor_network == "fast_kan_ac"
    resolved_training_config = None
    launch_config_path = config_path
    config_overrides = {}
    if adaptive_kan or fastkan_ac or args.task_duration_epochs is not None:
        resolved_training_config = json.loads(json.dumps(config))
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
            resolved_training_config.update(FASTKAN_AC_CONFIG_OVERRIDES)
            config_overrides.update(FASTKAN_AC_CONFIG_OVERRIDES)
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
    if args.profile_stages:
        command.append("--profile-stages")
    command.extend(("--compile-world-model", "--fused-adam", "--tf32"))
    boundary_epochs = list(
        range(task_duration_epochs - 1, training_epochs, task_duration_epochs)
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
    if args.task_prefix_length is not None:
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
    if fastkan_ac:
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
        "runtime": "vendored-optimized",
        "started_at_utc": None,
        "project_git": project_git,
        "profile_stages": args.profile_stages,
        "optimizations": [
            "distribution-free-categorical-kernels",
            "compiled-world-model-loss",
            "fused-adam",
            "tf32-matmul",
            "set-to-none-gradients",
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
            "task_prefix_length": args.task_prefix_length,
            "epochs": training_epochs,
            "task_duration_epochs": task_duration_epochs,
            "baseline_task_duration_epochs": swap_sched,
            "task_duration_epoch_override": args.task_duration_epochs,
            "full_curriculum": args.task_prefix_length is None,
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
        },
        "curriculum": args.curriculum,
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
        "replay_storage_budget": _arrow_replay_storage_budget(config),
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
                34 if fastkan_ac else 64 if is_kan_actor else None
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
            "critic_replay_loss_scale": 0.0,
            "paper_critic_replay_loss_scale": 0.3 if fastkan_ac else None,
            "critic_replay_loss_deviation": (
                "not ported because ARROW trains behavior separately from its world-model "
                "replay batches"
                if fastkan_ac
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
