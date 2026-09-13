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


def resolved_config(control, seed_index=0, *, smoke=False):
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
    if smoke:
        config["esc"]["env_configs"] = config["esc"]["env_configs"][:2]
        config["esc"]["kwargs"]["swap_sched"] = 1
        config.update(epochs=2, rssm_num_experts=2, steps_per_batch=2,
                      ac_train_steps=2, n_sync=4, gen_seq_len=64,
                      data_n=4, data_t=64, data_n_max=8, log_frequency=1)
    return config


def budgets(config):
    slots = 2 * config["data_n_max"]
    transitions = slots * config["data_t"]
    observations = transitions * 3 * config["img_size"] ** 2
    decisions = config["epochs"] * config["n_sync"] * config["gen_seq_len"]
    return {"world_model_updates": config["epochs"] * config["steps_per_batch"],
            "actor_critic_updates": config["epochs"] * config["ac_train_steps"],
            "agent_decisions": decisions, "raw_environment_frames": decisions * config["env_repeat"],
            "replay_trajectories": slots, "replay_observation_bytes": observations,
            "replay_tensor_bytes_without_task_metadata": observations + transitions * (config["action_space"] + 3) * 4,
            "replay_task_id_tensor_bytes": slots * 8,
            "replay_index_overhead": "Python LTDM priority index; not included in tensor bytes"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", choices=CONTROLS, required=True)
    parser.add_argument("--seed", type=int, choices=range(5), default=0)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Two-task, production-width/minibatch validation, not a result")
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("CPU thread count must be positive")
    state = git_state(ROOT) if args.dry_run else require_synced_training_git_state(ROOT)
    config = resolved_config(args.control, args.seed, smoke=args.smoke)
    output = args.output_dir.resolve()
    env = os.environ.copy()
    env.update({key: str(args.cpu_threads) for key in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT / "src"), env.get("PYTHONPATH"))))
    command = [str(args.python.resolve()), "Code/ARROW_and_DV3/Atari/train.py",
               "--config", str(output / "resolved_training_config.json"),
               "--log-dir", str(output), "--task-bank-snapshot-dir", str(output / "task_boundary_snapshots"),
               "--project-git-commit", state["commit"], "--evaluate-final", "--fused-adam", "--tf32", "--profile-stages"]
    manifest = {"method": NAMES[args.control], "classification": "smoke" if args.smoke else "pilot",
                "protocol": f"CapacityOrganization-{args.control}-OriginalSix-Atari-TaskAware-v1",
                "smoke_override": args.smoke,
                "project_git": state, "seed_id": args.seed, "seed": config["seed"],
                "resolved_training_config": config, "cpu_threads": args.cpu_threads,
                "command": command, "output_dir": str(output),
                "upstream_arrow_pin": "cb05e7d97ed83c3cf6e528960db0da6868e29232",
                "budgets": budgets(config),
                "fully_resolved_schema_config": "config.json (written by common trainer)",
                "backend_settings": {"compute": "BF16", "parameter_dtype": "FP32", "fused_adam": True,
                    "tf32": True, "compile_world_model": False, "deterministic_algorithms": False,
                    "known_nondeterminism": "CUDA reductions and platform scheduling; seeds do not promise bitwise replay"},
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
