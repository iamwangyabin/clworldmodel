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
        choices=["reconstruction", ],
        default="reconstruction",
        help=(
            "Use the published pixel-reconstruction objective. "
            ""
        ),
    )
    parser.add_argument(
        "--actor-network",
        choices=["mlp"],
        default="mlp",
        help=(
            "Use the retained DreamerV3 MLP actor and critic"
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
    if single_task and args.task_prefix_length is not None:
        parser.error(
            "--single-task-index cannot be combined with task-prefix overrides"
        )
    if single_task and (
        args.actor_network != "mlp" or args.observation_objective != "reconstruction"
    ):
        parser.error(
            "--single-task-index is reserved for the published ARROW-50 MLP "
            "reconstruction baseline"
        )
    if (
        args.task_prefix_length is not None
        and args.observation_objective != "reconstruction"
    ):
        parser.error("Task-prefix pilots must be tested independently from the R2 ablation")
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
    task_duration_epochs = swap_sched
    training_epochs = (
        config["epochs"]
        if args.task_prefix_length is None
        else task_duration_epochs * args.task_prefix_length
    )


    resolved_training_config = None
    launch_config_path = config_path
    config_overrides = {}
    if args.replay_device == "cpu":
        resolved_training_config = json.loads(json.dumps(config))
        for replay_config in resolved_training_config["replay_buffers"]:
            replay_config["rb_device"] = "cpu"
        resolved_training_config["replay_observation_dtype"] = "float32"
        config_overrides.update(
            {
                "replay_buffers": resolved_training_config["replay_buffers"],
                "replay_observation_dtype": "float32",
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
    if args.task_prefix_length is not None:
        command.extend(("--epochs", str(training_epochs), "--evaluate-final"))
    milestone_completed_epochs = ([])
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

    method = "ARROW-50"
    role = "primary-method"
    if single_task:
        method += "-SingleTask"
        role = "single-task-normalization-reproduction"
    elif args.task_prefix_length is not None:
        method += f"-T{args.task_prefix_length}Pilot"
        role = (
            ("matched-short-pilot-control")
        )

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
                    ([])
                ),
            ],
        },
        "training_scope": {
            "single_task_index": args.single_task_index,
            "task_prefix_length": args.task_prefix_length,
            "epochs": training_epochs,
            "task_duration_epochs": task_duration_epochs,
            "baseline_task_duration_epochs": swap_sched,
            "task_duration_epoch_override": None,
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
                (None)
            ),
            "kan_dreamer_step_mapping": (
                (None)
            ),
            "midpoint_completed_epochs": (None),
            "midpoint_agent_decisions": (None),
        },
        "continual_retention_evaluation": (
            (None)
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
                ("mlp")
            ),
            "input_features": 1536,
            "action_features": config["action_space"],
            "recurrent_features": config["gru_units"],
            "kan_hidden_layers": (None),
            "kan_basis_count": (None),
            "kan_basis": (
                ((None))
            ),
            "kan_input_range": (
                ((None))
            ),
            "kan_grid_trainable": (
                (None)
            ),
            "kan_anchor_parameterization": (
                (None)
            ),
            "kan_anchor_parameters": (
                (None)
            ),
            "kan_hidden_adapter": (
                (None)
            ),
            "kan_hidden_adapter_layer_norm_epsilon": (
                (None)
            ),
            "kan_rms_norm_epsilon": (None),
            "trainable_parameters": (
                (797_202)
            ),
            "critic_trainable_parameters": (
                (918_783)
            ),
            "combined_trainable_parameters": (
                (1_715_985)
            ),
            "mlp_combined_trainable_parameters": 1_715_985,
            "combined_parameter_difference_from_mlp": (
                (0)
            ),
            "actor_output_scale": (None),
            "actor_unimix": (None),
            "critic_output_scale": (None),
            "base_branch": (None),
            "rbf_bandwidth": (None),
            "implementation": (
                ("vendored-mlp")
            ),
            "reference": (
                ((None))
            ),
        },
        "actor_critic_training": {
            "optimizer": ("adam"),
            "learning_rate": (None),
            "optimizer_epsilon": (None),
            "optimizer_betas": (None),
            "optimizer_warmup_updates": (None),
            "gradient_clipping": (
                ({"type": "global_norm", "coefficient": 100.0})
            ),
            "imagination_horizon": (16),
            "discount_horizon": (None),
            "return_lambda": 0.95,
            "entropy_regularizer": 3e-4,
            "return_normalization": {
                "percentiles": [5, 95],
                "minimum_scale": 1.0,
                "decay": 0.99,
                "persists_across_epochs": False,
            },
            "critic_ema_regularizer": (0.0),
            "critic_ema_decay": (None),
            "critic_replay_loss_scale": (
                (0.0)
            ),
            "paper_critic_replay_loss_scale": (None),
            "critic_replay_loss_deviation": (
                (None)
            ),
            "critic_replay_loss_semantics": (
                (None)
            ),
            "critic_replay_reward_timing": (
                (None)
            ),
            "dreamerv3_repval_reference": (
                (None)
            ),
            "imagination_value_target": (
                ("online_critic")
            ),
            "actor_advantage_baseline": (
                ("online_critic")
            ),
            "terminal_bootstrap_state": (
                ("legacy_last_pre_transition_state")
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
            "decoder_enabled": True,
            "barlow_loss_scale": (None),
            "redundancy_scale": (None),
            "normalization_eps": (None),
            "target_gradient": (None),
            "sample_axes": (None),
        },
        "r2_dreamer_reference": (None),
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
