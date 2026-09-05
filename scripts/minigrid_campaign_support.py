"""Shared formal launcher support for the MiniGrid replay comparison."""

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
from typing import Any, Sequence

from git_provenance import git_state, require_synced_training_git_state
from launcher_support import run_and_tee, runtime_info, write_json


ROOT = Path(__file__).resolve().parents[1]
ARROW_ROOT = ROOT / "third_party" / "arrow"
ARROW_UPSTREAM_COMMIT = "cb05e7d97ed83c3cf6e528960db0da6868e29232"
CONTINUAL_DREAMER_SOURCE_COMMIT = "77f05bcebc56ad2f9bc22f82f6d4d02e62da87f6"
PROTOCOL = "ARROW-DV3RS-MiniGrid-3Task-v1"
SEEDS = (123456789, 1337, 31337, 42, 987654321)
TASKS = (
    "MiniGrid-DoorKey-9x9-v0",
    "MiniGrid-LavaCrossingS9N1-v0",
    "MiniGrid-SimpleCrossingS9N1-v0",
)
TASK_EPOCHS = (741, 750, 750)
TASK_INTERACTIONS = (750_000, 750_000, 750_000)
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
MINIMUM_MEMORY_HEADROOM_BYTES = 8 * 1024**3


def parse_args(description: str, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed-index", type=int, choices=range(len(SEEDS)), required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--profile-stages", action="store_true")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: dict[str, Any]) -> str:
    encoded = (json.dumps(value, indent=2) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_config(path: Path, seed_index: int) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    config["seed"] = SEEDS[seed_index]
    tasks = tuple(item["name"] for item in config["esc"]["env_configs"])
    if tasks != TASKS:
        raise RuntimeError(f"Formal MiniGrid task order changed: {tasks}")
    if any(item.get("family") != "minigrid" for item in config["esc"]["env_configs"]):
        raise RuntimeError("Every formal MiniGrid environment must declare family=minigrid")
    expected = {
        "action_space": 7,
        "img_size": 64,
        "epochs": sum(TASK_EPOCHS),
        "n_sync": 4,
        "gen_seq_len": 250,
        "env_repeat": 1,
        "data_n": 20,
        "data_t": 50,
        "mb_t_size": 32,
        "mb_n_size": 16,
        "random_policy": "first",
        "pretrain_enabled": True,
        "pretrain_data_multiplier": 10,
        "pretrain_steps": 610,
        "steps_per_batch": 61,
        "ac_train_steps": 49,
        "replay_observation_dtype": "uint8",
        "evaluation_seed_protocol": "fixed_validation_heldout_final",
        "deterministic_evaluation": True,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if config["esc"]["kwargs"] != {"task_durations": list(TASK_EPOCHS)}:
        mismatches["task_durations"] = (
            config["esc"]["kwargs"],
            {"task_durations": list(TASK_EPOCHS)},
        )
    if any(item.get("rb_device") != "cpu" for item in config["replay_buffers"]):
        mismatches["replay_devices"] = (
            [item.get("rb_device") for item in config["replay_buffers"]],
            ["cpu"] * len(config["replay_buffers"]),
        )
    if mismatches:
        raise RuntimeError(f"Invalid formal MiniGrid config: {mismatches}")
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
            "Formal MiniGrid methods differ outside algorithm and replay retention"
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
    if config["replay_observation_dtype"] != "uint8":
        raise ValueError("The formal MiniGrid campaign requires uint8 replay")
    transitions = total_slots * int(config["data_t"])
    observation_bytes = transitions * 3 * int(config["img_size"]) ** 2
    auxiliary_bytes = transitions * (int(config["action_space"]) + 3) * 4
    return {
        "total_trajectory_slots": total_slots,
        "fifo_trajectory_slots": fifo_slots,
        "reservoir_trajectory_slots": reservoir_slots,
        "ltdm_trajectory_slots": reservoir_slots,
        "retention_unit": "fixed-length-collected-trajectory",
        "sequence_length": int(config["data_t"]),
        "transition_capacity": transitions,
        "observation_dtype": "uint8",
        "sampled_observation_dtype": "float32",
        "storage_device": "cpu",
        "observation_bytes": observation_bytes,
        "action_reward_continue_reset_bytes": auxiliary_bytes,
        "tensor_bytes_excluding_allocator_overhead": (
            observation_bytes + auxiliary_bytes
        ),
        "buffer_selection_probability": selection_probability,
    }


def _available_memory_bytes() -> int:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        raise RuntimeError("Formal memory preflight requires Linux /proc/meminfo")
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def _physical_gpu_info() -> dict[str, Any]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in query.stdout.splitlines():
        index, name, memory_mib, driver = (part.strip() for part in line.split(",", 3))
        rows.append(
            {
                "index": int(index),
                "name": name,
                "memory_total_mib": int(memory_mib),
                "driver_version": driver,
            }
        )
    return {"physical_gpu_count": len(rows), "physical_gpus": rows}


def _single_visible_gpu(environment: dict[str, str], *, dry_run: bool) -> str | None:
    value = environment.get("CUDA_VISIBLE_DEVICES")
    if dry_run and not value:
        return None
    if value is None or len([item for item in value.split(",") if item.strip()]) != 1:
        raise RuntimeError(
            "Formal MiniGrid runs require CUDA_VISIBLE_DEVICES to select exactly one GPU"
        )
    return value


def _interaction_and_update_budgets(
    config: dict[str, Any],
    task_interactions: Sequence[int] = TASK_INTERACTIONS,
    task_epochs: Sequence[int] = TASK_EPOCHS,
) -> dict[str, Any]:
    regular_collection = int(config["n_sync"]) * int(config["gen_seq_len"])
    first_collection = regular_collection * int(config["pretrain_data_multiplier"])
    total_collections = first_collection + regular_collection * (int(config["epochs"]) - 1)
    if sum(task_epochs) != int(config["epochs"]):
        raise ValueError("Task epochs must sum to the configured training epochs")
    expected_by_task = [regular_collection * count for count in task_epochs]
    expected_by_task[0] += first_collection - regular_collection
    if list(task_interactions) != expected_by_task:
        raise ValueError("Declared per-task collection budgets do not match the schedule")
    if total_collections != sum(task_interactions):
        raise RuntimeError(
            f"MiniGrid collection budget changed: {total_collections} != {sum(task_interactions)}"
        )
    world_model_updates = int(config["pretrain_steps"]) + int(
        config["steps_per_batch"]
    ) * (int(config["epochs"]) - 1)
    actor_critic_updates = int(config["ac_train_steps"]) * int(config["epochs"])
    return {
        "environment_decisions": total_collections,
        "raw_environment_frames": total_collections,
        "collected_transitions": total_collections,
        "evaluation_interactions_included": False,
        "world_model_updates": world_model_updates,
        "actor_critic_updates": actor_critic_updates,
        "per_task_environment_decisions": list(task_interactions),
        "per_task_world_model_updates": [
            int(config["pretrain_steps"]) + int(config["steps_per_batch"]) * (task_epochs[0] - 1),
            *[int(config["steps_per_batch"]) * count for count in task_epochs[1:]],
        ],
        "per_task_actor_critic_updates": [int(config["ac_train_steps"]) * count for count in task_epochs],
        "regular_environment_decisions_per_epoch": regular_collection,
        "initial_random_prefill_decisions": first_collection,
    }


def launch_formal(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    source_config: Path,
    method: str,
    output_stem: str,
    replay: dict[str, Any],
    method_semantics: dict[str, Any],
    command_options: Sequence[str] = (),
    protocol: str = PROTOCOL,
    evidence_level: str = "official-candidate",
    claim_scope: str = "matched-five-seed-MiniGrid-comparison-after-aggregation",
    task_names: Sequence[str] = TASKS,
    task_epochs: Sequence[int] = TASK_EPOCHS,
    task_interactions: Sequence[int] = TASK_INTERACTIONS,
) -> int:
    python = args.python.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else ROOT / "runs" / f"{output_stem}_s{args.seed_index}"
    )
    if not args.dry_run:
        # Fetch before checking; stale remote-tracking refs are not launch proof.
        subprocess.run(["git", "fetch", "--prune"], cwd=ROOT, check=True)
    project_git = (
        git_state(ROOT) if args.dry_run else require_synced_training_git_state(ROOT)
    )
    environment = os.environ.copy()
    visible_gpu = _single_visible_gpu(environment, dry_run=args.dry_run)
    project_pythonpath = str(ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (project_pythonpath, environment.get("PYTHONPATH"))
        if part
    )
    resolved_config_path = output_dir / "resolved_training_config.json"
    snapshot_dir = output_dir / "analysis_snapshots"
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
        "--project-git-commit",
        str(project_git["commit"]),
        "--evaluate-final",
        "--fused-adam",
        "--tf32",
    ]
    if args.profile_stages:
        command.append("--profile-stages")

    budgets = _interaction_and_update_budgets(config, task_interactions, task_epochs)
    if config.get("learning_diagnostics", False):
        nominal = budgets["environment_decisions"]
        initial_rows = nominal // config["gen_seq_len"]
        budgets = {
            "nominal_stored_rows": nominal,
            "initial_reset_rows": initial_rows,
            "actual_environment_actions": (
                nominal - initial_rows
                if config["collection_autoreset_mode"] == "same_step"
                else None
            ),
            "raw_environment_frames": (
                (nominal - initial_rows) * config["env_repeat"]
                if config["collection_autoreset_mode"] == "same_step"
                else None
            ),
            "legacy_next_step_actual_count_source": "collection_diagnostics.jsonl",
            "world_model_updates": budgets["world_model_updates"],
            "actor_critic_updates": budgets["actor_critic_updates"],
            "evaluation_interactions_included": False,
            "counter_semantics": "Diagnostic train/evaluation counters count executed environment actions; initial reset rows and ignored NEXT_STEP actions are excluded. Stored rows are reported separately.",
        }
    launch = {
        "schema_version": 1,
        "artifact_kind": "diagnostic_training_launch" if evidence_level != "official-candidate" else "formal_training_launch",
        "method": method,
        "protocol": protocol,
        "evidence_level": evidence_level,
        "claim_scope": claim_scope,
        "started_at_utc": None,
        "project_git": project_git,
        "upstream_arrow_commit": ARROW_UPSTREAM_COMMIT,
        "continual_dreamer_source": {
            "repository": "https://github.com/skezle/continual-dreamer",
            "commit": CONTINUAL_DREAMER_SOURCE_COMMIT,
        },
        "source_config": {
            "path": str(source_config),
            "sha256": _sha256(source_config),
        },
        "resolved_config": {
            "path": str(resolved_config_path),
            "sha256": _json_sha256(config),
            "values": config,
        },
        "dependency_spec": {
            "path": str(ROOT / "pyproject.toml"),
            "sha256": _sha256(ROOT / "pyproject.toml"),
        },
        "output_dir": str(output_dir),
        "method_semantics": method_semantics,
        "seed": {
            "index": args.seed_index,
            "value": config["seed"],
            "predeclared_seed_set": list(SEEDS),
        },
        "task_schedule": {
            "type": "sequential",
            "task_identity_exposed_to_agent": False,
            "tasks": list(task_names),
            "epochs_per_task": list(task_epochs),
            "environment_decisions_per_task": (
                list(task_interactions) if not config.get("learning_diagnostics", False) else None
            ),
            "nominal_stored_rows_per_task": list(task_interactions),
            "initial_random_collection_only_at_run_start": True,
        },
        "budgets": budgets,
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
        "seeding": {
            "python_numpy_torch_cuda": True,
            "environment_reset_and_action_space": True,
            "replay_sampler": True,
            "collection_validation_final_streams_disjoint": True,
        },
        "evaluation": {
            "metric_schema_version": "raw-taskwise-evaluation-v1",
            "training_data_flow_separate": True,
            "enters_replay": False,
            "periodic_every_regular_environment_decisions": (
                10 * config["n_sync"] * (config["gen_seq_len"] - 1)
                if config.get("learning_diagnostics", False)
                and config["collection_autoreset_mode"] == "same_step"
                else None if config.get("learning_diagnostics", False) else 10_000
            ),
            "periodic_every_regular_collection_epochs": 10,
            "task_boundary_evaluation": True,
            "future_tasks_evaluated": len(task_names) > 1,
            "final_enabled": True,
            "rollouts_per_task": 16,
            "seed_protocol": "fixed_validation_heldout_final",
            "policy": "deterministic_argmax_and_latent_mode",
            "metrics": ["raw_return_mean", "raw_return_std"],
        },
        "checkpointing": {
            "analysis_snapshots_enabled": True,
            "replay_checkpointed": False,
            "resumable": False,
        },
        "backend": {
            "tf32": True,
            "fused_adam": True,
            "training_deterministic": False,
            "known_nondeterminism": ["CUDA kernels may be nondeterministic"],
        },
        "memory_preflight": {
            "minimum_headroom_bytes_after_replay": MINIMUM_MEMORY_HEADROOM_BYTES,
            "required_available_bytes": (
                replay["tensor_bytes_excluding_allocator_overhead"]
                + MINIMUM_MEMORY_HEADROOM_BYTES
            ),
        },
        "profile_stages": args.profile_stages,
        "python": str(python),
        "project_pythonpath_prepend": project_pythonpath,
        "cuda_visible_devices": visible_gpu,
        "command": command,
    }
    print(json.dumps(launch, indent=2))
    print("command: " + shlex.join([f"PYTHONPATH={environment['PYTHONPATH']}", *command]))
    if args.dry_run:
        return 0
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run directory: {output_dir}")

    available_memory = _available_memory_bytes()
    required_memory = int(launch["memory_preflight"]["required_available_bytes"])
    launch["memory_preflight"]["available_bytes_at_launch"] = available_memory
    if available_memory < required_memory:
        raise RuntimeError(
            "Insufficient available RAM for fixed-capacity CPU replay: "
            f"available={available_memory} required={required_memory}"
        )

    runtime = runtime_info(python, environment, package_names=RUNTIME_PACKAGES)
    runtime.update(_physical_gpu_info())
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
            "expected_budgets": budgets,
        },
    )
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return 0
