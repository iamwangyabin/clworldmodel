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


def _cuda_info(python: Path, env: dict[str, str]) -> dict:
    probe_code = (
        "import json, torch; "
        "assert torch.cuda.is_available() and torch.cuda.device_count() >= 1; "
        "p=torch.cuda.get_device_properties(0); "
        "print(json.dumps({'device_count': torch.cuda.device_count(), "
        "'device_name': p.name, 'total_memory_gib': p.total_memory / 1024**3}))"
    )
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
    args = _parser().parse_args()
    project_git = (
        git_state(ROOT) if args.dry_run else require_synced_training_git_state(ROOT)
    )
    python = args.python.resolve()
    config_path = _config_path(args.curriculum, args.seed)
    config = _verify_primary_config(config_path, args.curriculum, args.seed)
    output_prefix = (
        "arrow_r2rep_ar50"
        if args.observation_objective == "r2"
        else "arrow_ar50"
    )
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
    if args.observation_objective == "r2":
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
    if args.profile_stages:
        command.append("--profile-stages")
    command.extend(("--compile-world-model", "--fused-adam", "--tf32"))
    swap_sched = config["esc"]["kwargs"]["swap_sched"]
    boundary_epochs = list(range(swap_sched - 1, config["epochs"], swap_sched))
    is_r2 = args.observation_objective == "r2"
    launch = {
        "method": "ARROW-R2Rep-50" if is_r2 else "ARROW-50",
        "role": "representation-objective-ablation" if is_r2 else "primary-method",
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
        "fifo_slots": 512,
        "ltdm_slots": 512,
        "sequence_length": 512,
        "replay_buffer_selection": {"fifo": 0.5, "ltdm": 0.5},
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
