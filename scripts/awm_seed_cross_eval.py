#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only: frozen v4 task-0 boundary model × two validation cohorts.

Never resumes training, creates Replay, selects weights, or changes the legacy
evaluator. Run each model on the same host/GPU; keep outputs outside its run.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from artifact_io import sha256_file, write_json_atomic, write_sha256_sidecar
from git_provenance import require_synced_training_git_state
from launcher_support import runtime_info

ROOT = Path(__file__).resolve().parents[1]
TRAINING_COMMIT = "12d317eaf98eaebeb462894894b6fc83bb5e7041"
TRAINING_SEEDS = (123456789, 1337)
VALIDATION_SEEDS = (2127245496, 1981893629)
PROTOCOL = "AWM-AutoRoute-v4-Task0-Boundary90-ValidationCrossSeed-Diagnostic-v1"


def validate_snapshot(payload, seed_index):
    expected = {
        "artifact_kind": "task_bank_boundary_inference_snapshot",
        "schema_version": 1, "resumable": False,
        "saved_after_final_task_update_before_schedule_advance": True,
        "project_git_commit": TRAINING_COMMIT, "completed_epochs": 90,
        "epoch": 89, "world_model_updates": 92000, "actor_critic_updates": 72000,
        "seed": TRAINING_SEEDS[seed_index],
        "completed_task": {"boundary_index": 1, "task_index": 0,
                           "task_name": "ALE/MsPacman-v5", "task_reward_scale": .05},
        "inference_routing": {"mode": "two_frame_probability_reconstruction",
                              "protocol_version": 4, "eligible_route_ids": [0],
                              "task_identity_input": False},
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Not the authorized boundary snapshot: {key}")
    for key, value in {"seed": TRAINING_SEEDS[seed_index], "n_sync": 4,
                       "env_repeat": 4, "actor_network": "mlp",
                       "task_route_inference_version": 4,
                       "evaluation_episode_count_mode": "legacy",
                       "evaluation_seed_protocol": "fixed_validation_heldout_final"}.items():
        if payload.get("config", {}).get(key) != value:
            raise ValueError(f"Snapshot config differs from diagnostic protocol: {key}")


def evaluate_cohort(trajectory, wm, behavior, config, seed, output):
    """Observe the unchanged evaluator's returned tensors; never alter inputs."""
    import torch
    generate = trajectory.generate_trajectories
    counts = {}

    def capture(*args, **kwargs):
        tensors = generate(*args, **kwargs)
        torch.save(tensors, output / "legacy_evaluation_tensors.pt")
        counts["returned_tensor_samples"] = len(tensors[0])
        return tensors

    diagnostics = {}
    with torch.no_grad(), patch.object(trajectory, "generate_trajectories", side_effect=capture):
        mean, std = trajectory.evaluate(
            config.n_sync, wm=wm, ac=behavior,
            env_fns=config.get_env_schedule().eval_funcs()[0],
            env_repeat=config.env_repeat, n_rollouts=16, seed=seed,
            deterministic_policy=True, eligible_route_ids=(0,),
            task_route_inference=config.task_route_inference, diagnostics=diagnostics,
        )
    if not math.isfinite(mean) or not math.isfinite(std) or diagnostics.get("completed_episodes", 0) < 1:
        raise RuntimeError("Non-finite or empty diagnostic evaluation")
    return {"evaluation_seed": seed, "scaled_return_mean": float(mean),
            "scaled_return_std": float(std), "raw_return_mean": float(mean / .05),
            "raw_return_std": float(std / .05), "diagnostics": diagnostics, **counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--snapshot-sha256", required=True)
    parser.add_argument("--seed-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    subprocess.run(["git", "fetch", "--prune", "origin"], cwd=ROOT, check=True)
    provenance = require_synced_training_git_state(ROOT)
    snapshot, output = args.snapshot.resolve(), args.output_dir.resolve()
    if output.exists() or snapshot.parent.parent == output or snapshot.parent.parent in output.parents:
        raise ValueError("Use a NEW diagnostic directory outside the source training run")
    if sha256_file(snapshot) != args.snapshot_sha256:
        raise ValueError("Snapshot checksum mismatch")

    # Reuse the existing production-shaped constructor; no smoke/update is run.
    from smoke_evolving_atomic_rssm import Config, _world_model, np, torch
    from ac import ActorCritic
    from clworldmodel.routing import RoutedActorBank
    import generate_trajectory as trajectory

    payload = torch.load(snapshot, map_location="cpu", weights_only=False)
    validate_snapshot(payload, args.seed_index)
    config = Config.from_dict(payload["config"])
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly the assigned evaluation CUDA device")
    torch.set_num_threads(8)
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("high")
    random.seed(20260908)
    np.random.seed(20260908)
    torch.manual_seed(20260908)
    wm = _world_model(config, torch.device("cuda:0"))
    wm.load_state_dict(payload["world_model_state_dict"], strict=True)
    ac = ActorCritic(int(np.prod(wm.ls)) + wm.h_dim, wm.a_dim,
                     actor_network=config.actor_network, h_dim=wm.h_dim).to("cuda:0")
    ac.load_state_dict(payload["completed_task_actor_critic_state_dict"], strict=True)
    wm.eval().requires_grad_(False)
    ac.eval().requires_grad_(False)
    behavior = RoutedActorBank({0: ac.actor}).eval()
    output.mkdir(parents=True)
    manifest = {
        "classification": "debug", "protocol": PROTOCOL, "metric_schema_version": 1,
        "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "audit_git": provenance, "training_git_commit": payload["project_git_commit"],
        "upstream_arrow_commit": "cb05e7d97ed83c3cf6e528960db0da6868e29232",
        "source_snapshot": str(snapshot), "snapshot_sha256": args.snapshot_sha256,
        "seed_index": args.seed_index, "training_seed": config.seed,
        "resolved_training_config": config.to_dict(), "validation_seeds": VALIDATION_SEEDS,
        "initialization_seed": 20260908, "parameters_restored_strictly": True,
        "adaptive_compression_layout": wm.rssm.adaptive_compression_layout(),
        "runtime": runtime_info(Path(sys.executable), dict(os.environ)),
        "cpu": subprocess.check_output(["lscpu"], text=True),
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv"], text=True),
        "threads": torch.get_num_threads(), "tf32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "optimizer_updates": 0, "replay_capacity": 0, "replay_bytes": 0,
        "task_identity_input": False, "eligible_route_ids": [0],
        "nominal_rollouts_per_cohort": 16, "nominal_decision_cap_per_cohort": 32768,
        "frame_repeat": 4, "observation_dtype": "uint8 -> float32; BF16 compute",
        "evaluation_data_enters_training": False, "formal_results": False,
        "cells": [],
    }
    write_json_atomic(output / "manifest.json", manifest)
    for cohort_index, seed in enumerate(VALIDATION_SEEDS):
        cohort = output / f"validation_s{cohort_index}"
        cohort.mkdir()
        start = time.monotonic()
        cell = evaluate_cohort(trajectory, wm, behavior, config, seed, cohort)
        cell["elapsed_seconds"] = time.monotonic() - start
        cell["trajectory_sha256"] = write_sha256_sidecar(cohort / "legacy_evaluation_tensors.pt")
        write_json_atomic(cohort / "result.json", cell)
        manifest["cells"].append(cell)
        write_json_atomic(output / "manifest.json", manifest)
        print(f"S{args.seed_index} × validation S{cohort_index}: {cell['raw_return_mean']}", flush=True)
    for module, key in ((wm, "world_model_state_dict"), (ac, "completed_task_actor_critic_state_dict")):
        for name, value in module.state_dict().items():
            if not torch.equal(value.cpu(), payload[key][name]):
                raise RuntimeError(f"Evaluation changed checkpoint state: {key}/{name}")
    if sha256_file(snapshot) != args.snapshot_sha256:
        raise RuntimeError("Source snapshot changed during evaluation")
    manifest.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
                    parameters_and_buffers_unchanged=True, source_snapshot_unchanged=True)
    write_json_atomic(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
