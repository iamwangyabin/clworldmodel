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
from launcher_support import run_and_tee as _run_and_tee, write_json as _write_json

ROOT = Path(__file__).resolve().parents[1]
ARROW_ROOT = ROOT / "third_party" / "arrow"
UPSTREAM_COMMIT = "cb05e7d97ed83c3cf6e528960db0da6868e29232"
CONFIG_STEM = (
    "ALE_MsPacman,ALE_Boxing,ALE_CrazyClimber,ALE_Frostbite,"
    "ALE_Seaquest,ALE_Enduro-s{seed}-{method}.json"
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
        description="Launch the matched DreamerV3 FIFO continual Atari control"
    )
    parser.add_argument("--seed", type=int, choices=range(5), default=0)
    parser.add_argument("--curriculum", choices=CURRICULUM_DIRS, default="original")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Persistent run directory. Defaults to runs/dv3_fifo_<curriculum>_"
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


def _config_path(curriculum: str, seed: int, method: str) -> Path:
    return (
        ARROW_ROOT
        / "Configs"
        / "Atari configs"
        / "CL-task configs"
        / CURRICULUM_DIRS[curriculum]
        / CONFIG_STEM.format(seed=seed, method=method)
    )


def _verify_control_config(config_path: Path, curriculum: str, seed: int) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    errors = []
    if config.get("algorithm") != "dv3":
        errors.append("algorithm must be dv3")
    if config.get("data_t") != 512:
        errors.append("matched control requires data_t=512")
    if config.get("sac_dv3_data_n_max") != 1024:
        errors.append("matched control requires a 1,024-trajectory FIFO")
    replay_types = [item.get("rb_type") for item in config.get("replay_buffers", [])]
    if replay_types != ["FifoReplay"]:
        errors.append("matched control requires exactly one FIFO replay buffer")
    if config.get("seed") != SEEDS[seed]:
        errors.append("numeric seed does not match the published seed ID")
    envs = config.get("esc", {}).get("env_configs", [])
    if len(envs) != 6:
        errors.append("continual Atari config must contain six tasks")
    swap_sched = config.get("esc", {}).get("kwargs", {}).get("swap_sched")
    expected_swap = 45 if curriculum == "two-cycle" else 90
    if swap_sched != expected_swap:
        errors.append(f"swap_sched must be {expected_swap} for {curriculum}")

    arrow_path = _config_path(curriculum, seed, "arrow")
    arrow_config = json.loads(arrow_path.read_text(encoding="utf-8"))
    control_shared = {
        key: value
        for key, value in config.items()
        if key not in {"algorithm", "replay_buffers"}
    }
    arrow_shared = {
        key: value
        for key, value in arrow_config.items()
        if key not in {"algorithm", "replay_buffers"}
    }
    if control_shared != arrow_shared:
        errors.append("DV3 and ARROW configs differ outside method/replay selection")
    if config.get("sac_dv3_data_n_max") != 2 * arrow_config.get("data_n_max", -1):
        errors.append("DV3 FIFO capacity does not match total ARROW trajectory capacity")

    if errors:
        raise RuntimeError("Invalid DreamerV3 control config: " + "; ".join(errors))
    return config


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
    args = _parser().parse_args()
    project_git = (
        git_state(ROOT) if args.dry_run else require_synced_training_git_state(ROOT)
    )
    python = args.python.resolve()
    config_path = _config_path(args.curriculum, args.seed, "dv3")
    config = _verify_control_config(config_path, args.curriculum, args.seed)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else ROOT
        / "runs"
        / f"dv3_fifo_{args.curriculum}_s{args.seed}_analysis"
    )
    snapshot_dir = output_dir / "analysis_snapshots"

    env = os.environ.copy()
    thread_env = {}
    if args.cpu_threads is not None:
        thread_env = {key: str(args.cpu_threads) for key in THREAD_ENV_KEYS}
        env.update(thread_env)

    command = [
        str(python),
        "Code/ARROW_and_DV3/Atari/train.py",
        "--config",
        str(config_path),
        "--log-dir",
        str(output_dir),
        "--analysis-snapshot-dir",
        str(snapshot_dir),
    ]
    if args.profile_stages:
        command.append("--profile-stages")
    command.extend(("--compile-world-model", "--fused-adam", "--tf32"))

    swap_sched = config["esc"]["kwargs"]["swap_sched"]
    boundary_epochs = list(range(swap_sched - 1, config["epochs"], swap_sched))
    launch = {
        "method": "DreamerV3/FIFO",
        "role": "matched-control",
        "runtime": "vendored-optimized",
        "started_at_utc": None,
        "profile_stages": args.profile_stages,
        "optimizations": [
            "distribution-free-categorical-kernels",
            "compiled-world-model-loss",
            "fused-adam",
            "tf32-matmul",
            "set-to-none-gradients",
        ],
        "project_git": project_git,
        "upstream_commit": UPSTREAM_COMMIT,
        "source": str(ARROW_ROOT),
        "config": str(config_path),
        "output_dir": str(output_dir),
        "analysis_snapshot_dir": str(snapshot_dir),
        "analysis_snapshot_semantics": {
            "artifact_kind": "analysis_snapshot",
            "resumable": False,
            "task_boundary_epochs": boundary_epochs,
            "final_epoch": config["epochs"] - 1,
            "omitted_state": ["optimizers", "replay", "RNG", "environment schedule"],
        },
        "curriculum": args.curriculum,
        "seed_id": args.seed,
        "seed": config["seed"],
        "fifo_slots": config["sac_dv3_data_n_max"],
        "ltdm_slots": 0,
        "sequence_length": config["data_t"],
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
    launch["started_at_utc"] = datetime.now(timezone.utc).isoformat()
    launch["cuda"] = cuda
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
