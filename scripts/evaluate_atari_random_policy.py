#!/usr/bin/env python3
"""Evaluate one shared uniform-random Atari reference without using a GPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VENDORED_ATARI = ROOT / "third_party" / "arrow" / "Code" / "ARROW_and_DV3" / "Atari"
SEED_ROOTS = (123456789, 1337, 31337, 42, 987654321)
TASKS = (
    "ALE/MsPacman-v5",
    "ALE/Boxing-v5",
    "ALE/CrazyClimber-v5",
    "ALE/Frostbite-v5",
    "ALE/Seaquest-v5",
    "ALE/Enduro-v5",
)
COHORTS = ("validation", "heldout_final")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the shared uniform-random lower reference for the frozen "
            "ARROW Atari environment protocol. No model or GPU is used."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seed-indices",
        type=int,
        nargs="+",
        default=list(range(len(SEED_ROOTS))),
        choices=range(len(SEED_ROOTS)),
        help="published ARROW seed indices (default: all five)",
    )
    parser.add_argument(
        "--cohorts",
        nargs="+",
        default=["validation"],
        choices=COHORTS,
        help="fixed evaluation cohorts to measure (default: validation)",
    )
    parser.add_argument("--rollouts", type=int, default=16)
    parser.add_argument("--n-sync", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.rollouts < 1:
        raise ValueError("--rollouts must be positive")
    if args.n_sync < 1:
        raise ValueError("--n-sync must be positive")
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")
    if len(set(args.seed_indices)) != len(args.seed_indices):
        raise ValueError("--seed-indices must not contain duplicates")
    if len(set(args.cohorts)) != len(args.cohorts):
        raise ValueError("--cohorts must not contain duplicates")


def _task_seed_table(seed_root: int) -> dict[str, dict[str, list[int]]]:
    """Match the fixed validation/final streams and add a disjoint policy stream."""
    (
        _,
        validation_seed,
        final_seed,
        validation_policy_seed,
        final_policy_seed,
    ) = np.random.SeedSequence(seed_root).spawn(5)

    def draw(seed: np.random.SeedSequence) -> list[int]:
        rng = np.random.default_rng(seed)
        return [int(rng.integers(0, 2**32, dtype=np.uint64)) for _ in TASKS]

    return {
        "validation": {
            "environment": draw(validation_seed),
            "policy": draw(validation_policy_seed),
        },
        "heldout_final": {
            "environment": draw(final_seed),
            "policy": draw(final_policy_seed),
        },
    }


def _episode_returns(rewards: Any, continuations: Any, resets: Any) -> list[float]:
    """Recover complete episode returns using the vendored evaluator semantics."""
    reward_values = np.asarray(rewards).reshape(-1)
    continuation_values = np.asarray(continuations).reshape(-1)
    reset_values = np.asarray(resets).reshape(-1)
    terminals = np.flatnonzero(continuation_values == 0)
    starts = np.flatnonzero(reset_values == 1)
    markers = [(int(index), "E") for index in terminals]
    markers.extend((int(index), "S") for index in starts)
    markers.sort()
    returns = []
    for (start_index, start_kind), (end_index, end_kind) in zip(markers, markers[1:]):
        if start_kind == "S" and end_kind == "E":
            returns.append(float(reward_values[start_index : end_index + 1].sum()))
    return returns


def _git_state() -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from git_provenance import require_synced_training_git_state

    return require_synced_training_git_state(ROOT)


def _runtime_info() -> dict[str, Any]:
    packages = ("ale-py", "gymnasium", "numpy", "opencv-python", "torch")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "packages": versions,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _manifest(args: argparse.Namespace, git: dict[str, Any] | None) -> dict[str, Any]:
    selected_roots = [SEED_ROOTS[index] for index in args.seed_indices]
    return {
        "schema_version": "atari-random-reference-v1",
        "classification": "pilot" if len(selected_roots) < 5 else "reference",
        "method": "uniform-random-policy",
        "shared_across_methods": True,
        "metric": "raw_episode_return",
        "git": git,
        "tasks": list(TASKS),
        "seed_indices": list(args.seed_indices),
        "seed_roots": selected_roots,
        "cohorts": list(args.cohorts),
        "rollouts_target_per_task_cohort_seed": args.rollouts,
        "n_sync": args.n_sync,
        "cpu_threads": args.cpu_threads,
        "gpu_required": False,
        "environment": {
            "base": {
                "frameskip": 1,
                "repeat_action_probability": 0,
                "full_action_space": True,
            },
            "preprocessing": {
                "wrapper": "gymnasium.wrappers.AtariPreprocessing",
                "frame_skip": 4,
                "screen_size": 64,
                "grayscale_obs": False,
            },
            "action_count": 18,
            "action_selection": "uniform randint over [0, 18)",
            "reward": "raw and unscaled",
            "episode_extraction": "vendored ARROW evaluator semantics",
        },
        "seed_protocol": {
            "environment": (
                "ARROW fixed_validation_heldout_final task seeds from SeedSequence "
                "children 1 and 2"
            ),
            "policy": (
                "disjoint SeedSequence children 3 and 4 for validation and final, "
                "one action RNG seed per task"
            ),
        },
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _summaries(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for record in records:
        grouped.setdefault((record["cohort"], record["task"]), []).append(record["mean"])
    summaries = {}
    for (cohort, task), seed_means in grouped.items():
        values = np.asarray(seed_means, dtype=np.float64)
        summaries.setdefault(cohort, {})[task] = {
            "seed_means": values.tolist(),
            "median_seed_mean": float(np.median(values)),
            "iqr_seed_mean": [
                float(np.percentile(values, 25)),
                float(np.percentile(values, 75)),
            ],
        }
    return summaries


def _load_experiment_runtime(cpu_threads: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = str(cpu_threads)
    import ale_py
    import gymnasium as gym
    import torch

    gym.register_envs(ale_py)
    torch.set_num_threads(cpu_threads)
    sys.path.insert(0, str(VENDORED_ATARI))
    from generate_trajectory import generate_trajectories

    return gym, generate_trajectories


def _evaluate_one(
    *,
    gym: Any,
    generate_trajectories: Any,
    task: str,
    environment_seed: int,
    policy_seed: int,
    rollouts: int,
    n_sync: int,
) -> list[float]:
    np.random.seed(policy_seed)
    env_fn = partial(
        gym.make,
        task,
        frameskip=1,
        repeat_action_probability=0,
        full_action_space=True,
    )
    _, _, rewards, continuations, resets = generate_trajectories(
        rollouts * 2**13 // n_sync,
        n_sync,
        wm=None,
        ac=None,
        env_fns=[env_fn for _ in range(n_sync)],
        env_repeat=4,
        target_terminals=rollouts,
        no_images=True,
        seed=environment_seed,
    )
    returns = _episode_returns(rewards, continuations, resets)
    if not returns:
        raise RuntimeError(f"No complete episodes recovered for {task}")
    return returns


def main() -> int:
    args = _parser().parse_args()
    _validate_args(args)
    if args.dry_run:
        print(json.dumps(_manifest(args, git=None), indent=2))
        return 0

    git = _git_state()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    manifest = _manifest(args, git)
    manifest["runtime"] = _runtime_info()
    manifest["started_at_unix"] = time.time()
    _write_json(args.output_dir / "launch.json", manifest)
    _write_json(args.output_dir / "run_status.json", {"complete": False})

    gym, generate_trajectories = _load_experiment_runtime(args.cpu_threads)
    records: list[dict[str, Any]] = []
    try:
        for seed_index in args.seed_indices:
            seed_root = SEED_ROOTS[seed_index]
            task_seeds = _task_seed_table(seed_root)
            for cohort in args.cohorts:
                for task_index, task in enumerate(TASKS):
                    started = time.monotonic()
                    environment_seed = task_seeds[cohort]["environment"][task_index]
                    policy_seed = task_seeds[cohort]["policy"][task_index]
                    returns = _evaluate_one(
                        gym=gym,
                        generate_trajectories=generate_trajectories,
                        task=task,
                        environment_seed=environment_seed,
                        policy_seed=policy_seed,
                        rollouts=args.rollouts,
                        n_sync=args.n_sync,
                    )
                    values = np.asarray(returns, dtype=np.float64)
                    record = {
                        "seed_index": seed_index,
                        "seed_root": seed_root,
                        "cohort": cohort,
                        "task_index": task_index,
                        "task": task,
                        "environment_seed": environment_seed,
                        "policy_seed": policy_seed,
                        "episode_returns": values.tolist(),
                        "episode_count_actual": int(values.size),
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "elapsed_seconds": time.monotonic() - started,
                    }
                    records.append(record)
                    _write_json(
                        args.output_dir / "partial_results.json",
                        {"records": records, "summaries": _summaries(records)},
                    )
                    print(
                        f"seed={seed_index} cohort={cohort} task={task} "
                        f"episodes={values.size} raw_return={values.mean():.6g}"
                    )
    except BaseException as error:
        _write_json(
            args.output_dir / "run_status.json",
            {"complete": False, "error": f"{type(error).__name__}: {error}"},
        )
        raise

    result = {"records": records, "summaries": _summaries(records)}
    _write_json(args.output_dir / "random_policy_reference.json", result)
    manifest_bytes = (args.output_dir / "random_policy_reference.json").read_bytes()
    _write_json(
        args.output_dir / "run_status.json",
        {
            "complete": True,
            "record_count": len(records),
            "result_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "finished_at_unix": time.time(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
