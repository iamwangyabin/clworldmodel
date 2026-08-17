#!/usr/bin/env python3
"""Reproducible launcher for native R2-Dreamer plus ARROW-50 replay."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from git_provenance import git_state, require_synced_training_git_state


ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = ROOT / "src"
ARROW_ROOT = ROOT / "third_party" / "arrow"
sys.path.insert(0, str(PROJECT_SRC))
from clworldmodel.r2dreamer.config import R2_DREAMER_SOURCE_COMMIT, R2DreamerConfig


ARROW_UPSTREAM_COMMIT = "cb05e7d97ed83c3cf6e528960db0da6868e29232"
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
R2_ATARI_100K_RAW_FRAMES = 410_000


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least one")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch R2-Dreamer size12M with ARROW-50 FIFO/LTDM replay"
    )
    parser.add_argument("--seed", type=int, choices=range(5), default=0)
    parser.add_argument("--curriculum", choices=CURRICULUM_DIRS, default="original")
    parser.add_argument(
        "--scope",
        choices=["single-task", "continual"],
        default="single-task",
        help="Run the required single-task R2 sanity check or the six-task hybrid protocol",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Persistent run directory; defaults under runs/.",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cpu-threads", type=_positive_int)
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One epoch and one update; establishes execution only.",
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


def _read_and_verify_arrow_config(
    config_path: Path, curriculum: str, seed_index: int
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    errors = []
    if config.get("algorithm") != "arrow":
        errors.append("source config must use ARROW replay")
    if config.get("data_n_max") != 512 or config.get("data_t") != 512:
        errors.append("ARROW-50 storage requires 512 trajectory slots per sub-buffer")
    if [item.get("rb_type") for item in config.get("replay_buffers", [])] != [
        "FifoReplay",
        "LongTermReplay",
    ]:
        errors.append("source config must declare FIFO followed by LTDM")
    if config.get("arrow_replay_capacity_ratio", "50-50") != "50-50":
        errors.append("source config must preserve ARROW-50 capacity and sampling weights")
    if config.get("seed") != SEEDS[seed_index]:
        errors.append("numeric seed ID does not match the published seed")
    task_count = len(config.get("esc", {}).get("env_configs", []))
    if task_count != 6:
        errors.append("source continual curriculum must contain six Atari tasks")
    expected_swap = 45 if curriculum == "two-cycle" else 90
    if config.get("esc", {}).get("kwargs", {}).get("swap_sched") != expected_swap:
        errors.append(f"curriculum must use swap_sched={expected_swap}")
    if config.get("mb_t_size", 0) < 1 or config.get("mb_n_size", 0) < 1:
        errors.append("source config must define a positive world-model minibatch")
    if errors:
        raise RuntimeError("Invalid R2-Dreamer ARROW source config: " + "; ".join(errors))
    return config


def _resolved_budget(config: dict[str, Any], *, scope: str, smoke: bool) -> dict[str, int]:
    original_samples_per_update = int(config["mb_t_size"]) * int(config["mb_n_size"])
    r2_config = R2DreamerConfig()
    decisions_per_epoch = int(config["n_sync"]) * int(config["gen_seq_len"])
    raw_frames_per_epoch = decisions_per_epoch * int(config["env_repeat"])
    updates_numerator = decisions_per_epoch * r2_config.native_train_ratio
    if updates_numerator % r2_config.sample_count:
        raise RuntimeError("R2 native train ratio cannot be represented by full R2 batches")
    updates = updates_numerator // r2_config.sample_count
    if scope == "single-task":
        epochs = math.ceil(R2_ATARI_100K_RAW_FRAMES / raw_frames_per_epoch)
        task_count = 1
    else:
        epochs = int(config["epochs"])
        task_count = 6
    if smoke:
        epochs = 1
        task_count = 1
        updates = 1
    return {
        "task_count": task_count,
        "epochs": epochs,
        "nominal_world_model_updates_per_epoch": updates,
        "nominal_trajectory_positions_per_epoch": decisions_per_epoch,
        "nominal_raw_frames_per_epoch": raw_frames_per_epoch,
        "native_train_ratio": r2_config.native_train_ratio,
        "single_task_target_raw_frames": (
            R2_ATARI_100K_RAW_FRAMES if scope == "single-task" else 0
        ),
        "total_nominal_trajectory_positions": epochs * decisions_per_epoch,
        "total_nominal_raw_frames": epochs * raw_frames_per_epoch,
        "total_nominal_r2_model_sample_transitions": epochs * updates * r2_config.sample_count,
        "source_samples_per_update": original_samples_per_update,
        "r2_samples_per_update": r2_config.sample_count,
        "source_samples_per_epoch": int(config["steps_per_batch"]) * original_samples_per_update,
        "r2_samples_per_epoch": updates * r2_config.sample_count,
    }


def _cuda_info(python: Path, environment: dict[str, str]) -> dict[str, Any]:
    probe_code = (
        "import json, torch; "
        "assert torch.cuda.is_available() and torch.cuda.device_count() >= 1; "
        "p=torch.cuda.get_device_properties(0); "
        "print(json.dumps({'device_count': torch.cuda.device_count(), "
        "'device_name': p.name, 'total_memory_gib': p.total_memory / 1024**3, "
        "'torch_version': torch.__version__}))"
    )
    result = subprocess.run(
        [str(python), "-c", probe_code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return json.loads(result.stdout.strip())


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _run_and_tee(
    command: list[str], *, cwd: Path, environment: dict[str, str], log_path: Path
) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
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
    source_config = _read_and_verify_arrow_config(config_path, args.curriculum, args.seed)
    budget = _resolved_budget(source_config, scope=args.scope, smoke=args.smoke)
    r2_config = R2DreamerConfig()
    prefix = "r2dreamer_arrow50_smoke" if args.smoke else "r2dreamer_arrow50"
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else ROOT / "runs" / f"{prefix}_{args.scope}_{args.curriculum}_s{args.seed}"
    )
    snapshot_dir = output_dir / "analysis_snapshots"
    environment = os.environ.copy()
    thread_environment = {}
    if args.cpu_threads is not None:
        thread_environment = {key: str(args.cpu_threads) for key in THREAD_ENV_KEYS}
        environment.update(thread_environment)
    inherited_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        str(path) for path in (PROJECT_SRC, ROOT, inherited_pythonpath) if path
    )
    command = [
        str(python),
        str(ROOT / "scripts" / "train_r2dreamer_arrow_atari.py"),
        "--config",
        str(config_path),
        "--log-dir",
        str(output_dir),
        "--launcher-created-log-dir",
        "--task-count",
        str(budget["task_count"]),
        "--epochs",
        str(budget["epochs"]),
        "--device",
        "cuda",
        "--analysis-snapshot-dir",
        str(snapshot_dir),
    ]
    if args.smoke:
        command.extend(["--world-model-updates-per-epoch", "1"])
    else:
        command.extend(["--native-train-ratio", str(r2_config.native_train_ratio)])
    if args.profile_stages:
        command.append("--profile-stages")

    launch = {
        "method": "R2Dreamer-ARROW-50",
        "role": "native-r2dreamer-with-arrow-replay",
        "status_label": "smoke" if args.smoke else "pilot",
        "started_at_utc": None,
        "project_git": project_git,
        "upstream": {
            "r2dreamer": {
                "repository": "https://github.com/NM512/r2dreamer",
                "commit": R2_DREAMER_SOURCE_COMMIT,
                "profile": "size12M",
            },
            "arrow": {
                "repository": "https://github.com/Cerenaut/ARROW",
                "commit": ARROW_UPSTREAM_COMMIT,
            },
        },
        "scope": args.scope,
        "curriculum": args.curriculum,
        "seed_id": args.seed,
        "seed": source_config["seed"],
        "source_arrow_config": str(config_path),
        "output_dir": str(output_dir),
        "analysis_snapshot_dir": str(snapshot_dir),
        "analysis_snapshot_semantics": {
            "artifact_kind": "analysis_snapshot",
            "resumable": False,
            "omitted_state": ["optimizer", "replay", "RNG", "environment schedule"],
        },
        "r2dreamer": {
            "decoder_enabled": False,
            "embedding_dim": r2_config.embedding_dim,
            "rssm_feature_dim": r2_config.feature_dim,
            "batch_size": r2_config.batch_size,
            "batch_length": r2_config.batch_length,
            "flattened_barlow_samples": r2_config.sample_count,
            "barlow_loss_scale": r2_config.loss_scale_barlow,
            "redundancy_scale": r2_config.barlow_redundancy_scale,
            "optimizer": "LaProp",
            "learning_rate": r2_config.learning_rate,
            "agc": r2_config.agc,
            "warmup_updates": r2_config.warmup_updates,
            "native_train_ratio": r2_config.native_train_ratio,
            "float32_matmul_precision": "high",
            "torch_compile_update": False,
            "rmsnorm_compatibility": "native-or-project-equivalent-fallback",
        },
        "arrow_replay": {
            "fifo_slots": 512,
            "ltdm_slots": 512,
            "sequence_length": 512,
            "buffer_selection": {"fifo": 0.5, "ltdm": 0.5},
            "sample_context_steps": 1,
            "terminal_observation_semantics": "gymnasium-next-step-autoreset",
            "retention_and_sampling_unchanged": True,
        },
        "budget": budget,
        "protocol_deviations_from_upstream_r2dreamer": [
            "Uses the ARROW continual Atari task schedule and its environment adapter.",
            "Uses ARROW FIFO/LTDM trajectory storage instead of the upstream TorchRL buffer.",
            "Uses R2-Dreamer's native 128 model-sample-per-decision train ratio; it is intentionally not compute-matched to ARROW.",
            "Uses the native R2-Dreamer integrated world-model and actor-critic update rather than ARROW's separate controller trainer.",
            "Uses an equivalent project RMSNorm fallback on the pinned PyTorch 2.3 runtime when torch.nn.RMSNorm is unavailable.",
            "Disables upstream torch.compile for the pinned runtime; model geometry and update equations are unchanged.",
        ],
        "profile_stages": args.profile_stages,
        "cpu_threads": args.cpu_threads,
        "environment": thread_environment,
        "project_pythonpath": environment["PYTHONPATH"],
        "command": command,
    }
    print(json.dumps(launch, indent=2))
    rendered_environment = [f"{key}={value}" for key, value in thread_environment.items()]
    rendered_environment.append(f"PYTHONPATH={environment['PYTHONPATH']}")
    print(f"command: {shlex.join([*rendered_environment, *command])}")
    if args.dry_run:
        return 0

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {output_dir}")
    cuda = _cuda_info(python, environment)
    output_dir.mkdir(parents=True)
    launch["started_at_utc"] = datetime.now(timezone.utc).isoformat()
    launch["cuda"] = cuda
    _write_json(output_dir / "launch.json", launch)
    return_code = _run_and_tee(
        command,
        cwd=ROOT,
        environment=environment,
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
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
