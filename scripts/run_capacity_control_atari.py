#!/usr/bin/env python3
"""Launch one explicitly named capacity control through the common Atari trainer."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

from git_provenance import git_state, require_synced_training_git_state
from launcher_support import run_and_tee, runtime_info, write_json

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = ("shared", "wide", "fullbank", "frozen", "independent")
NAMES = {"shared": "Shared WM + Private AC", "wide": "Wider Shared WM + Private AC",
         "fullbank": "Full per-task WMs + Private AC", "frozen": "Frozen shared core + residuals",
         "independent": "Independent residuals + plastic shared core"}


def resolved_config(control, seed_index=0):
    if control not in CONTROLS or seed_index not in range(5):
        raise ValueError("Unknown control or seed index")
    source = next((ROOT / "third_party/arrow/Configs/Atari configs/CL-task configs/Original Order").glob(f"*-s{seed_index}-arrow.json"))
    config = json.loads(source.read_text())
    config.update(continual_method="capacity_control_v1", capacity_control=control,
                  capacity_residual_atoms=4, epochs=540, rssm_num_experts=6,
                  task_private_actor_critic=True, compute_dtype="bfloat16",
                  replay_observation_dtype="uint8",
                  evaluation_seed_protocol="fixed_validation_heldout_final",
                  mlp_features=1536 if control == "wide" else 512)
    for buffer in config["replay_buffers"]:
        buffer["rb_device"] = "cpu"
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", choices=CONTROLS, required=True)
    parser.add_argument("--seed", type=int, choices=range(5), default=0)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("CPU thread count must be positive")
    state = git_state(ROOT) if args.dry_run else require_synced_training_git_state(ROOT)
    config = resolved_config(args.control, args.seed)
    output = args.output_dir.resolve()
    env = os.environ.copy()
    env.update({key: str(args.cpu_threads) for key in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT / "src"), env.get("PYTHONPATH"))))
    command = [str(args.python.resolve()), "Code/ARROW_and_DV3/Atari/train.py",
               "--config", str(output / "resolved_training_config.json"),
               "--log-dir", str(output), "--task-bank-snapshot-dir", str(output / "task_boundary_snapshots"),
               "--project-git-commit", state["commit"], "--evaluate-final", "--fused-adam", "--tf32", "--profile-stages"]
    manifest = {"method": NAMES[args.control], "classification": "pilot",
                "protocol": f"CapacityOrganization-{args.control}-OriginalSix-Atari-TaskAware-v1",
                "project_git": state, "seed_id": args.seed, "seed": config["seed"],
                "resolved_training_config": config, "cpu_threads": args.cpu_threads,
                "command": command, "output_dir": str(output),
                "upstream_arrow_pin": "cb05e7d97ed83c3cf6e528960db0da6868e29232",
                "budgets": {"world_model_updates": 540000, "actor_critic_updates": 432000,
                            "agent_decisions": 8847360, "raw_environment_frames": 35389440,
                            "replay_trajectories": 1024, "replay_observation_bytes": 6442450944,
                            "replay_tensor_bytes_without_task_metadata": 6486491136},
                "research_limits": ["No functional protection, conflict projection, boundary consolidation or RCC",
                    "AWM has 12000 extra boundary WM updates; not a compute-matched AWM superiority test",
                    "Frozen/independent are organization controls, not exact one-switch AWM ablations",
                    "Private oracle AC for training and evaluation; no task-agnostic claim",
                    "Snapshots are inference-only; no equivalent resume or automatic retry"],
                "started_at_utc": None}
    print(json.dumps(manifest, indent=2))
    print("command:", shlex.join(command))
    if args.dry_run:
        return 0
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    manifest["runtime_environment"] = runtime_info(args.python.resolve(), env)
    output.mkdir(parents=True)
    manifest["started_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "resolved_training_config.json", config)
    write_json(output / "launch.json", manifest)
    result = run_and_tee(command, cwd=ROOT / "third_party/arrow", env=env, log_path=output / "train.log")
    write_json(output / "run_status.json", {"complete": result == 0, "return_code": result,
        "finished_at_utc": datetime.now(timezone.utc).isoformat()})
    return result


if __name__ == "__main__":
    raise SystemExit(main())
