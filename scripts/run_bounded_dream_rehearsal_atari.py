#!/usr/bin/env python3
"""Launch Dream Rehearsal on DreamerV3 with a fixed replay budget."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from git_provenance import git_state, require_synced_training_git_state
from launcher_support import run_and_tee as _run_and_tee, write_json as _write_json
from run_dv3_fifo_atari import (
    ARROW_ROOT,
    CURRICULUM_DIRS,
    ROOT,
    SEEDS,
    THREAD_ENV_KEYS,
    UPSTREAM_COMMIT,
    _config_path,
    _verify_control_config,
)


METHOD = "Bounded-Dream-Rehearsal-v1-Atari"
DREAM_REHEARSAL_PAPER = "https://arxiv.org/abs/2607.19749"
DREAM_REHEARSAL_REPOSITORY = "https://github.com/gurpnijjer/dream-rehearsal"
DREAM_REHEARSAL_COMMIT = "7680778f798be3a27a17c320cc875b573c45f0e1"
DEFAULT_INTERVAL_AGENT_DECISIONS = 2_000
DEFAULT_UPDATES_PER_PRIOR_TASK = 50
DEFAULT_BATCH_SEQUENCES = 4
DEFAULT_CONTEXT_STEPS = 16
DEFAULT_HORIZON = 15
DEFAULT_TOP_FRACTION = 0.25
DEFAULT_REALIZED_THRESHOLD = 0.3
DEFAULT_REALIZED_BONUS = 10.0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch actor-only graded Dream Rehearsal with a bounded, "
            "ARROW-capacity-matched trajectory reservoir"
        )
    )
    parser.add_argument("--seed", type=int, choices=range(5), default=0)
    parser.add_argument("--curriculum", choices=CURRICULUM_DIRS, default="original")
    parser.add_argument(
        "--replay-capacity-transitions",
        type=_positive_int,
        help=(
            "Fixed replay sample capacity. The default is 524,288 transitions "
            "(1,024 x 512), exactly matching total ARROW-50 capacity."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Persistent run directory. Defaults to "
            "runs/bounded_dream_rehearsal_<curriculum>_s<seed>_analysis."
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


def _resolved_config(source: dict, *, replay_slots: int) -> dict:
    config = copy.deepcopy(source)
    config.update(
        {
            "continual_method": "bounded_dream_rehearsal",
            "sac_dv3_data_n_max": replay_slots,
            "replay_buffers": [
                {"rb_type": "LongTermReplay", "rb_device": "cpu"}
            ],
            # Storage dtype never changes the sample-capacity match. It only
            # makes the fixed Atari library practical as a CPU mmap.
            "replay_observation_dtype": "uint8",
            "dream_rehearsal_interval_agent_decisions": (
                DEFAULT_INTERVAL_AGENT_DECISIONS
            ),
            "dream_rehearsal_updates_per_prior_task": (
                DEFAULT_UPDATES_PER_PRIOR_TASK
            ),
            "dream_rehearsal_batch_sequences": DEFAULT_BATCH_SEQUENCES,
            "dream_rehearsal_context_steps": DEFAULT_CONTEXT_STEPS,
            "dream_rehearsal_horizon": DEFAULT_HORIZON,
            "dream_rehearsal_top_fraction": DEFAULT_TOP_FRACTION,
            "dream_rehearsal_realized_threshold": DEFAULT_REALIZED_THRESHOLD,
            "dream_rehearsal_realized_bonus": DEFAULT_REALIZED_BONUS,
            "dream_rehearsal_grad_clip": 100.0,
        }
    )
    return config


def _storage_budget(config: dict) -> dict:
    slots = int(config["sac_dv3_data_n_max"])
    sequence_length = int(config["data_t"])
    transitions = slots * sequence_length
    observation_elements = 3 * int(config["img_size"]) ** 2
    action_elements = int(config["action_space"])
    observation_bytes = transitions * observation_elements  # uint8
    auxiliary_bytes = transitions * (action_elements + 3) * 4
    task_metadata_bytes = slots * 8
    return {
        "comparison_basis": "trajectory_and_transition_capacity_primary",
        "byte_accounting_role": "secondary_required_resource_report",
        "trajectory_slots": slots,
        "sequence_length": sequence_length,
        "transition_capacity": transitions,
        "retention": "uniform_random_key_top_k_reservoir",
        "observation": {
            "dtype": "uint8",
            "device": "cpu_mmap",
            "elements_per_transition": observation_elements,
            "allocated_bytes": observation_bytes,
        },
        "auxiliary": {
            "dtype": "float32",
            "device": "cpu",
            "elements_per_transition": action_elements + 3,
            "fields": ["one_hot_action", "reward", "continue", "reset"],
            "allocated_bytes": auxiliary_bytes,
        },
        "task_id_metadata": {
            "dtype": "int64",
            "elements": slots,
            "allocated_bytes": task_metadata_bytes,
            "model_input": False,
        },
        "allocated_tensor_bytes": (
            observation_bytes + auxiliary_bytes + task_metadata_bytes
        ),
        "python_sorted_key_index_bytes_included": False,
    }


def _task_for_epoch(config: dict, epoch: int) -> int:
    kwargs = config["esc"]["kwargs"]
    if "task_durations" in kwargs:
        durations = [int(value) for value in kwargs["task_durations"]]
    else:
        durations = [int(kwargs["swap_sched"])] * len(
            config["esc"]["env_configs"]
        )
    position = epoch % sum(durations)
    cumulative = 0
    for task_id, duration in enumerate(durations):
        cumulative += duration
        if position < cumulative:
            return task_id
    raise AssertionError("Task schedule position is outside its duration")


def _project_rehearsal_compute(config: dict) -> dict:
    decisions_per_epoch = int(config["n_sync"]) * int(config["gen_seq_len"])
    interval = int(config["dream_rehearsal_interval_agent_decisions"])
    updates_per_task = int(config["dream_rehearsal_updates_per_prior_task"])
    encountered: set[int] = set()
    updates_by_task: dict[int, int] = {}
    completed_intervals = 0
    for epoch in range(int(config["epochs"])):
        current_task = _task_for_epoch(config, epoch)
        encountered.add(current_task)
        before = epoch * decisions_per_epoch
        after = (epoch + 1) * decisions_per_epoch
        crossed = after // interval - before // interval
        completed_intervals += crossed
        for old_task in encountered - {current_task}:
            updates_by_task[old_task] = (
                updates_by_task.get(old_task, 0) + crossed * updates_per_task
            )
    extra_updates = sum(updates_by_task.values())
    starts_per_update = (
        int(config["dream_rehearsal_batch_sequences"])
        * int(config["dream_rehearsal_context_steps"])
    )
    selected_per_update = max(
        1, math.floor(starts_per_update * float(config["dream_rehearsal_top_fraction"]))
    )
    return {
        "completed_global_intervals": completed_intervals,
        "extra_actor_only_updates": extra_updates,
        "extra_actor_only_updates_by_replay_task": dict(sorted(updates_by_task.items())),
        "imagined_trajectories_per_update": starts_per_update,
        "selected_trajectories_per_update": selected_per_update,
        "projected_imagined_trajectories": extra_updates * starts_per_update,
        "projected_selected_trajectories": extra_updates * selected_per_update,
        "world_model_updates_from_rehearsal": 0,
        "critic_updates_from_rehearsal": 0,
    }


def _cuda_info(python: Path, env: dict[str, str]) -> dict:
    probe_code = (
        "import json, torch; "
        "assert torch.cuda.is_available() and torch.cuda.device_count() >= 1; "
        "p=torch.cuda.get_device_properties(0); "
        "print(json.dumps({'device_count': torch.cuda.device_count(), "
        "'device_name': p.name, 'total_memory_gib': p.total_memory / 1024**3}))"
    )
    probe = subprocess.run(
        [str(python), "-c", probe_code],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    return json.loads(probe.stdout.strip())


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    project_git = (
        git_state(ROOT) if args.dry_run else require_synced_training_git_state(ROOT)
    )
    source_config_path = _config_path(args.curriculum, args.seed, "dv3")
    source_config = _verify_control_config(
        source_config_path, args.curriculum, args.seed
    )
    sequence_length = int(source_config["data_t"])
    matched_capacity = int(source_config["sac_dv3_data_n_max"]) * sequence_length
    capacity = args.replay_capacity_transitions or matched_capacity
    if capacity % sequence_length:
        parser.error(
            "--replay-capacity-transitions must be divisible by the fixed "
            f"sequence length ({sequence_length})"
        )
    replay_slots = capacity // sequence_length
    config = _resolved_config(source_config, replay_slots=replay_slots)

    capacity_suffix = "" if capacity == matched_capacity else f"_m{capacity}"
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else ROOT
        / "runs"
        / (
            f"bounded_dream_rehearsal_{args.curriculum}_s{args.seed}"
            f"{capacity_suffix}_analysis"
        )
    )
    config_path = output_dir / "resolved_training_config.json"
    snapshot_dir = output_dir / "analysis_snapshots"

    python = args.python.expanduser().resolve()
    env = os.environ.copy()
    thread_env: dict[str, str] = {}
    if args.cpu_threads is not None:
        thread_env = {key: str(args.cpu_threads) for key in THREAD_ENV_KEYS}
        env.update(thread_env)
    project_pythonpath = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
    inherited_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (project_pythonpath, inherited_pythonpath) if value
    )

    command = [
        str(python),
        "Code/ARROW_and_DV3/Atari/train.py",
        "--config",
        str(config_path),
        "--log-dir",
        str(output_dir),
        "--analysis-snapshot-dir",
        str(snapshot_dir),
        "--compile-world-model",
        "--fused-adam",
        "--tf32",
    ]
    if args.profile_stages:
        command.append("--profile-stages")

    decisions_per_epoch = int(config["n_sync"]) * int(config["gen_seq_len"])
    agent_decisions = int(config["epochs"]) * decisions_per_epoch
    base_actor_updates = int(config["epochs"]) * int(config["ac_train_steps"])
    projected_rehearsal = _project_rehearsal_compute(config)
    swap_sched = int(config["esc"]["kwargs"]["swap_sched"])
    role = (
        "bounded-baseline-capacity-matched-to-arrow-50"
        if capacity == matched_capacity
        else "bounded-storage-capacity-ablation"
    )
    launch = {
        "method": METHOD,
        "role": role,
        "protocol": "Bounded-Dream-Rehearsal-v1-Atari",
        "started_at_utc": None,
        "project_git": project_git,
        "upstream_arrow_commit": UPSTREAM_COMMIT,
        "dream_rehearsal_reference": {
            "paper": DREAM_REHEARSAL_PAPER,
            "repository": DREAM_REHEARSAL_REPOSITORY,
            "commit": DREAM_REHEARSAL_COMMIT,
            "license": "Apache-2.0",
        },
        "source": str(ARROW_ROOT),
        "source_config": str(source_config_path),
        "resolved_training_config": str(config_path),
        "config_overrides": {
            key: config[key]
            for key in (
                "continual_method",
                "sac_dv3_data_n_max",
                "replay_buffers",
                "replay_observation_dtype",
                "dream_rehearsal_interval_agent_decisions",
                "dream_rehearsal_updates_per_prior_task",
                "dream_rehearsal_batch_sequences",
                "dream_rehearsal_context_steps",
                "dream_rehearsal_horizon",
                "dream_rehearsal_top_fraction",
                "dream_rehearsal_realized_threshold",
                "dream_rehearsal_realized_bonus",
                "dream_rehearsal_grad_clip",
            )
        },
        "project_pythonpath_prepend": project_pythonpath,
        "output_dir": str(output_dir),
        "analysis_snapshot_dir": str(snapshot_dir),
        "curriculum": args.curriculum,
        "seed_id": args.seed,
        "seed": SEEDS[args.seed],
        "training_scope": {
            "epochs": int(config["epochs"]),
            "task_duration_epochs": swap_sched,
            "tasks": [item["name"] for item in config["esc"]["env_configs"]],
            "agent_decisions": agent_decisions,
            "raw_environment_frames": agent_decisions * int(config["env_repeat"]),
            "world_model_updates": int(config["epochs"])
            * int(config["steps_per_batch"]),
            "base_actor_critic_updates": base_actor_updates,
            "actor_updates_total_including_rehearsal": (
                base_actor_updates
                + projected_rehearsal["extra_actor_only_updates"]
            ),
            "task_boundary_epochs": list(
                range(swap_sched - 1, int(config["epochs"]), swap_sched)
            ),
        },
        "bounded_replay": {
            **_storage_budget(config),
            "matched_arrow_50_transition_capacity": matched_capacity,
            "capacity_matches_arrow_50": capacity == matched_capacity,
            "never_clear_unbounded_replay": False,
        },
        "rehearsal": {
            "start_state_source": (
                "every posterior state in a task-filtered real replay context"
            ),
            "task_metadata_use": "replay_filter_and_scheduler_only",
            "task_id_exposed_to_world_model_or_actor": False,
            "policy": "single_shared_actor",
            "grading": "realized_first_reward_continuation_and_critic_bootstrap",
            "selection": "top_25_percent_by_score",
            "optimization": "actor_only_behavior_cloning_of_sampled_dream_actions",
            "interval_agent_decisions": DEFAULT_INTERVAL_AGENT_DECISIONS,
            "updates_per_prior_task_per_interval": DEFAULT_UPDATES_PER_PRIOR_TASK,
            "due_updates_batched_at_collection_epoch_boundary": True,
            **projected_rehearsal,
        },
        "declared_deviations_from_reference_artifact": [
            "The never-clear raw replay is replaced by one fixed-capacity random-key reservoir.",
            "Atari/DreamerV3 replaces the paper artifact's MiniGrid NM512 training stack.",
            "Due 2,000-decision rehearsal events are batched at the next 16,384-decision collection boundary while preserving the exact update count.",
        ],
        "comparison_contract": {
            "primary_replay_match": "stored_transition_capacity",
            "observation_compression_does_not_increase_sample_capacity": True,
            "extra_actor_compute_is_not_compute_matched_to_plain_dreamer": True,
            "report_replay_bytes_separately": True,
        },
        "analysis_snapshot_semantics": {
            "artifact_kind": "analysis_snapshot",
            "resumable": False,
            "separate_base_and_rehearsal_actor_update_counters": True,
            "omitted_state": ["optimizers", "replay", "RNG", "environment schedule"],
        },
        "cpu_threads": args.cpu_threads,
        "environment": thread_env,
        "command": command,
    }
    print(json.dumps(launch, indent=2))
    rendered_env = [f"{key}={value}" for key, value in thread_env.items()]
    print(f"command: {shlex.join([*rendered_env, *command])}")
    if args.dry_run:
        return 0

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {output_dir}")
    cuda = _cuda_info(python, env)
    output_dir.mkdir(parents=True)
    _write_json(config_path, config)
    launch["started_at_utc"] = datetime.now(timezone.utc).isoformat()
    launch["cuda"] = cuda
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
