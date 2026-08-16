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
        choices=["mlp", "relu_kan"],
        default="mlp",
        help="Keep the published MLP actor or use the parameter-matched KAN actor",
    )
    parser.add_argument(
        "--task-prefix-length",
        type=int,
        choices=[2, 3],
        help=(
            "Run only the first two or three tasks as a named one- or two-switch "
            "pilot; task order and per-task duration remain unchanged"
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
        args.actor_network == "relu_kan"
        and args.observation_objective != "reconstruction"
    ):
        parser.error("KAN-Actor must be tested independently from the R2 ablation")
    if (
        args.task_prefix_length is not None
        and args.observation_objective != "reconstruction"
    ):
        parser.error("Task-prefix pilots must be tested independently from the R2 ablation")
    project_git = (
        git_state(ROOT) if args.dry_run else require_synced_training_git_state(ROOT)
    )
    python = args.python.resolve()
    config_path = _config_path(args.curriculum, args.seed)
    config = _verify_primary_config(config_path, args.curriculum, args.seed)
    if args.actor_network == "relu_kan":
        output_prefix = "arrow_kan_actor_ar50"
    elif args.observation_objective == "r2":
        output_prefix = "arrow_r2rep_ar50"
    else:
        output_prefix = "arrow_ar50"
    if args.task_prefix_length is not None:
        output_prefix += f"_t{args.task_prefix_length}_pilot"
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
    if args.observation_objective == "r2" or args.actor_network == "relu_kan":
        project_pythonpath = str(ROOT / "src")
        inherited_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (project_pythonpath, inherited_pythonpath) if part
        )

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
    if args.actor_network == "relu_kan":
        command.extend(("--actor-network", "relu_kan"))
    swap_sched = config["esc"]["kwargs"]["swap_sched"]
    training_epochs = (
        config["epochs"]
        if args.task_prefix_length is None
        else swap_sched * args.task_prefix_length
    )
    if args.task_prefix_length is not None:
        command.extend(("--epochs", str(training_epochs), "--evaluate-final"))
    if args.profile_stages:
        command.append("--profile-stages")
    command.extend(("--compile-world-model", "--fused-adam", "--tf32"))
    boundary_epochs = list(range(swap_sched - 1, training_epochs, swap_sched))
    is_r2 = args.observation_objective == "r2"
    is_kan_actor = args.actor_network == "relu_kan"
    if is_kan_actor:
        method = "ARROW-KANActor-50"
        role = "actor-architecture-ablation"
    elif is_r2:
        method = "ARROW-R2Rep-50"
        role = "representation-objective-ablation"
    else:
        method = "ARROW-50"
        role = "primary-method"
    if args.task_prefix_length is not None:
        method += f"-T{args.task_prefix_length}Pilot"
        role = (
            "actor-architecture-pilot"
            if is_kan_actor
            else "matched-short-pilot-control"
        )
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
        "config": str(config_path),
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
            "omitted_state": ["optimizers", "replay", "RNG", "environment schedule"],
        },
        "training_scope": {
            "task_prefix_length": args.task_prefix_length,
            "epochs": training_epochs,
            "task_duration_epochs": swap_sched,
            "full_curriculum": args.task_prefix_length is None,
            "tasks": [
                task["name"]
                for task in config["esc"]["env_configs"][
                    : args.task_prefix_length
                    if args.task_prefix_length is not None
                    else None
                ]
            ],
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
            "environment_reset_seeded": False,
            "action_space_seeded": False,
            "torch_deterministic_algorithms": False,
            "tf32_enabled": True,
            "known_nondeterminism": [
                "environment construction and reset do not receive explicit seeds",
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
            "critic_network": "mlp",
            "input_features": 1536,
            "action_features": config["action_space"],
            "recurrent_features": config["gru_units"],
            "kan_hidden_features": 64 if is_kan_actor else None,
            "kan_grid_size": 5 if is_kan_actor else None,
            "kan_spline_order": 3 if is_kan_actor else None,
            "kan_basis_count": 8 if is_kan_actor else None,
            "kan_input_range": [0.0, 1.0] if is_kan_actor else None,
            "kan_grid_trainable": False if is_kan_actor else None,
            "kan_normalize_recurrent_state": True if is_kan_actor else None,
            "trainable_parameters": 795730 if is_kan_actor else 797202,
            "implementation": (
                "project-owned-independent-pytorch" if is_kan_actor else "vendored-mlp"
            ),
            "reference": (
                "https://arxiv.org/abs/2406.02075" if is_kan_actor else None
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
