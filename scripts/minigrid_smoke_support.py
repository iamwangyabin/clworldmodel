"""Shared, provenance-checked execution support for MiniGrid replay smokes."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from git_provenance import git_state, require_synced_training_git_state
from launcher_support import run_and_tee, runtime_info, write_json


ROOT = Path(__file__).resolve().parents[1]
ARROW_ROOT = ROOT / "third_party" / "arrow"
ARROW_UPSTREAM_COMMIT = "cb05e7d97ed83c3cf6e528960db0da6868e29232"
CONTINUAL_DREAMER_SOURCE_COMMIT = "77f05bcebc56ad2f9bc22f82f6d4d02e62da87f6"
TASKS = (
    "MiniGrid-DoorKey-9x9-v0",
    "MiniGrid-LavaCrossingS9N1-v0",
    "MiniGrid-SimpleCrossingS9N1-v0",
)
RUNTIME_PACKAGES = (
    "gymnasium",
    "minigrid",
    "numpy",
    "opencv-python",
    "sortedcontainers",
    "tensorboard",
    "torch",
    "tqdm",
)


def parse_args(
    description: str, argv: list[str] | None = None
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--profile-stages", action="store_true")
    args = parser.parse_args(argv)
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    return args


def load_common_config(source_config: Path, seed: int) -> dict[str, Any]:
    config = json.loads(source_config.read_text(encoding="utf-8"))
    config["seed"] = seed
    observed_tasks = tuple(task["name"] for task in config["esc"]["env_configs"])
    if observed_tasks != TASKS:
        raise RuntimeError(f"MiniGrid smoke task order changed: {observed_tasks}")
    if any(
        task.get("family") != "minigrid"
        for task in config["esc"]["env_configs"]
    ):
        raise RuntimeError("Every MiniGrid smoke task must declare family=minigrid")
    expected = {
        "action_space": 7,
        "img_size": 64,
        "env_repeat": 1,
        "epochs": 3,
        "data_parallel_world_size": 1,
        "replay_observation_dtype": "float32",
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key, 1 if key == "data_parallel_world_size" else None)
        != value
    }
    if mismatches:
        raise RuntimeError(f"Invalid MiniGrid smoke config: {mismatches}")
    if config["esc"]["kwargs"] != {"swap_sched": 1}:
        raise RuntimeError("MiniGrid smoke must cross all three task boundaries")
    return config


def assert_configs_match_outside_replay(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> None:
    method_keys = {
        "algorithm",
        "arrow_replay_capacity_ratio",
        "replay_buffers",
        "sac_dv3_data_n_max",
    }
    candidate_shared = {
        key: value for key, value in candidate.items() if key not in method_keys
    }
    reference_shared = {
        key: value for key, value in reference.items() if key not in method_keys
    }
    if candidate_shared != reference_shared:
        raise RuntimeError(
            "MiniGrid methods differ outside algorithm and replay retention"
        )


def replay_accounting(
    config: dict[str, Any],
    *,
    total_slots: int,
    fifo_slots: int,
    reservoir_slots: int,
    selection_probability: dict[str, float],
) -> dict[str, Any]:
    if fifo_slots + reservoir_slots != total_slots:
        raise ValueError("Replay sub-buffer slots do not sum to total capacity")
    frames = total_slots * int(config["data_t"])
    observation_element_bytes = 4
    image_bytes = (
        frames
        * 3
        * int(config["img_size"]) ** 2
        * observation_element_bytes
    )
    auxiliary_bytes = frames * (4 * int(config["action_space"]) + 3 * 4)
    return {
        "total_trajectory_slots": total_slots,
        "fifo_trajectory_slots": fifo_slots,
        "reservoir_trajectory_slots": reservoir_slots,
        # Keep the ARROW implementation name explicit for artifact consumers.
        "ltdm_trajectory_slots": reservoir_slots,
        "retention_unit": "fixed-length-collected-trajectory",
        "sequence_length": int(config["data_t"]),
        "observation_dtype": "float32",
        "storage_device": "cuda",
        "observation_bytes": image_bytes,
        "action_reward_continue_reset_bytes": auxiliary_bytes,
        "tensor_bytes_excluding_allocator_overhead": image_bytes + auxiliary_bytes,
        "buffer_selection_probability": selection_probability,
    }


def launch_smoke(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    source_config: Path,
    method: str,
    protocol: str,
    default_output_stem: str,
    replay: dict[str, Any],
    method_semantics: dict[str, Any],
    command_options: Sequence[str] = (),
) -> int:
    python = args.python.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else ROOT / "runs" / f"{default_output_stem}_s{args.seed}"
    )
    project_git = (
        git_state(ROOT) if args.dry_run else require_synced_training_git_state(ROOT)
    )
    resolved_config_path = output_dir / "resolved_training_config.json"
    snapshot_dir = output_dir / "analysis_snapshots"

    environment = os.environ.copy()
    project_pythonpath = str(ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (project_pythonpath, environment.get("PYTHONPATH"))
        if part
    )
    command = [
        str(python),
        "Code/ARROW_and_DV3/Atari/train.py",
        "--config",
        str(resolved_config_path),
        *command_options,
        "--log-dir",
        str(output_dir),
        "--analysis-snapshot-dir",
        str(snapshot_dir),
        "--evaluate-final",
        "--fused-adam",
        "--tf32",
    ]
    if args.profile_stages:
        command.append("--profile-stages")

    decisions_per_epoch = int(config["n_sync"]) * int(config["gen_seq_len"])
    environment_decisions = decisions_per_epoch * int(config["epochs"])
    launch = {
        "schema_version": 1,
        "method": method,
        "protocol": protocol,
        "claim_scope": "execution-correctness-smoke-only",
        "started_at_utc": None,
        "project_git": project_git,
        "upstream_arrow_commit": ARROW_UPSTREAM_COMMIT,
        "source_config": str(source_config),
        "resolved_config": str(resolved_config_path),
        "output_dir": str(output_dir),
        "method_semantics": method_semantics,
        "task_schedule": {
            "type": "sequential",
            "task_identity_exposed_to_agent": False,
            "tasks": list(TASKS),
            "epochs_per_task": [1, 1, 1],
            "environment_decisions_per_task": [decisions_per_epoch] * 3,
        },
        "budgets": {
            "environment_decisions": environment_decisions,
            "raw_environment_frames": environment_decisions,
            "collected_transitions": environment_decisions,
            "world_model_updates": int(config["steps_per_batch"])
            * int(config["epochs"]),
            "actor_critic_updates": int(config["ac_train_steps"])
            * int(config["epochs"]),
        },
        "observation": {
            "source": "agent-centred MiniGrid RGB partial observation",
            "mission_text_removed": True,
            "source_shape": [56, 56, 3],
            "model_shape": [64, 64, 3],
            "source_dtype": "uint8",
            "resize": "OpenCV INTER_AREA",
        },
        "action": {"type": "discrete", "count": 7, "repeat": 1},
        "replay": replay,
        "seeds": {
            "root": args.seed,
            "python_numpy_torch_cuda": True,
            "environment_reset_and_action_space": True,
            "collection_validation_final_streams_disjoint": True,
        },
        "evaluation": {
            "training_data_flow_separate": True,
            "enters_replay": False,
            "final_enabled": True,
            "rollouts_per_seen_task": 16,
            "policy": "stochastic vendored DreamerV3 policy",
            "metrics": ["raw_return_mean", "raw_return_std"],
        },
        "profile_stages": args.profile_stages,
        "python": str(python),
        "project_pythonpath_prepend": project_pythonpath,
        "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
        "command": command,
    }
    print(json.dumps(launch, indent=2))
    print(
        "command: "
        + shlex.join([f"PYTHONPATH={environment['PYTHONPATH']}", *command])
    )
    if args.dry_run:
        return 0
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run directory: {output_dir}")

    runtime = runtime_info(python, environment, package_names=RUNTIME_PACKAGES)
    output_dir.mkdir(parents=True)
    write_json(resolved_config_path, config)
    launch["started_at_utc"] = datetime.now(timezone.utc).isoformat()
    launch["runtime_environment"] = runtime
    write_json(output_dir / "launch.json", launch)
    return_code = run_and_tee(
        command,
        cwd=ARROW_ROOT,
        env=environment,
        log_path=output_dir / "train.log",
    )
    write_json(
        output_dir / "run_status.json",
        {
            "complete": return_code == 0,
            "return_code": return_code,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return 0
