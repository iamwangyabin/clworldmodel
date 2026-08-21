#!/usr/bin/env python3
"""Run a controlled Task-2 acquisition experiment from a Task-1 snapshot.

This is intentionally a trainability diagnostic, not a resumable continual run:
the analysis snapshot has no replay, optimizer, RNG, or environment-schedule
state, so Task 2 starts with an empty ARROW buffer and a fresh optimizer.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from git_provenance import git_state, require_synced_training_git_state
from run_karrow_ar50_atari import (
    ARROW_ROOT,
    DINOV3_DEPENDENCIES,
    ROOT,
    THREAD_ENV_KEYS,
    UPSTREAM_COMMIT,
    _dinov3_dependency_versions,
    _run_and_tee,
    _runtime_info,
    _write_json,
)


ADAPTATION_MODES = ("kan_only", "kan_plus_heads")
CONSOLIDATION_DEFAULTS = {
    "residual_consolidation_batches": 16,
    "residual_consolidation_imagination_horizon": 8,
    "residual_consolidation_gradient_power": 2.0,
    "residual_consolidation_min_plasticity": 0.01,
    "residual_consolidation_anchor_loss_scale": 1.0,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Task-2 acquisition from a Task-1 KARROW analysis snapshot"
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--source-config", type=Path)
    parser.add_argument("--task-index", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--adaptation-mode", choices=ADAPTATION_MODES, default="kan_only")
    parser.add_argument("--dinov3-model-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_config_path(snapshot: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    inferred = snapshot.parent.parent / "resolved_training_config.json"
    if not inferred.is_file():
        raise FileNotFoundError(
            "Could not infer the source config next to the snapshot; pass "
            "--source-config explicitly: "
            f"{inferred}"
        )
    return inferred


def _task2_config(
    source: dict,
    *,
    task_index: int,
    epochs: int,
    dinov3_model_path: Path | None,
) -> dict:
    if epochs < 1:
        raise ValueError("--epochs must be positive")
    config = copy.deepcopy(source)
    schedule = config["esc"]
    tasks = schedule["env_configs"]
    if not 0 <= task_index < len(tasks):
        raise ValueError(
            f"--task-index must lie in [0, {len(tasks) - 1}], got {task_index}"
        )
    schedule["env_configs"] = [tasks[task_index]]
    schedule.setdefault("kwargs", {})["swap_sched"] = epochs
    config["epochs"] = epochs
    config["shared_core_mode"] = "snapshot_adaptation"
    config["residual_consolidation"] = "none"
    config.update(CONSOLIDATION_DEFAULTS)
    if dinov3_model_path is not None:
        config["dinov3_model_path"] = str(dinov3_model_path.expanduser().resolve())
    return config


def main() -> int:
    args = _parser().parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    snapshot = args.snapshot.expanduser().resolve()
    if not snapshot.is_file():
        raise FileNotFoundError(f"Task-1 snapshot does not exist: {snapshot}")
    checksum_path = snapshot.with_suffix(snapshot.suffix + ".sha256")
    if not checksum_path.is_file():
        raise FileNotFoundError(f"Task-1 snapshot checksum is missing: {checksum_path}")

    source_config_path = _source_config_path(snapshot, args.source_config)
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    model_path = (
        args.dinov3_model_path.expanduser().resolve()
        if args.dinov3_model_path is not None
        else Path(source_config["dinov3_model_path"]).expanduser().resolve()
    )
    config = _task2_config(
        source_config,
        task_index=args.task_index,
        epochs=args.epochs,
        dinov3_model_path=model_path,
    )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else snapshot.parent.parent
        / f"task{args.task_index + 1}_from_{snapshot.stem}_{args.adaptation_mode}"
    )
    config_path = output_dir / "resolved_training_config.json"
    snapshot_dir = output_dir / "analysis_snapshots"
    python = args.python.expanduser().resolve()

    project_git = (
        git_state(ROOT) if args.dry_run else require_synced_training_git_state(ROOT)
    )
    env = os.environ.copy()
    thread_env: dict[str, str] = {}
    if args.cpu_threads is not None:
        if args.cpu_threads < 1:
            raise ValueError("--cpu-threads must be positive")
        thread_env = {key: str(args.cpu_threads) for key in THREAD_ENV_KEYS}
        env.update(thread_env)
    project_pythonpath = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
    inherited_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (project_pythonpath, inherited_pythonpath) if value
    )
    dependency_versions = _dinov3_dependency_versions(python, env)

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
        "--init-analysis-snapshot",
        str(snapshot),
        "--resume-adaptation-mode",
        args.adaptation_mode,
        "--compile-world-model",
        "--fused-adam",
        "--tf32",
        "--evaluate-final",
    ]
    if args.profile_stages:
        command.append("--profile-stages")

    launch = {
        "schema_version": 1,
        "method": "KARROW-Task2-SnapshotAcquisition",
        "protocol": "KARROW-Task2-SnapshotAcquisition-v1-Atari",
        "status": "dry_run" if args.dry_run else "launching",
        "project_git": project_git,
        "upstream_arrow_commit": UPSTREAM_COMMIT,
        "source_config": str(source_config_path),
        "source_snapshot": str(snapshot),
        "source_snapshot_sha256": _sha256(snapshot),
        "source_snapshot_checksum": checksum_path.read_text(encoding="ascii").strip(),
        "task_index": args.task_index,
        "task_name": config["esc"]["env_configs"][0]["name"],
        "epochs": args.epochs,
        "adaptation_mode": args.adaptation_mode,
        "replay_state": "reset_empty",
        "optimizer_state": "reset_new_optimizer",
        "rng_state": "reset_from_config_seed",
        "task_identity_exposed_to_agent": False,
        "resolved_training_config": str(config_path),
        "output_dir": str(output_dir),
        "analysis_snapshot_dir": str(snapshot_dir),
        "runtime_dependencies": dependency_versions,
        "cpu_threads": args.cpu_threads,
        "environment": thread_env,
        "project_pythonpath_prepend": project_pythonpath,
        "command": command,
    }
    print(json.dumps(launch, indent=2))
    rendered_env = [f"{key}={value}" for key, value in thread_env.items()]
    rendered_env.append(f"PYTHONPATH={env['PYTHONPATH']}")
    print(f"command: {shlex.join([*rendered_env, *command])}")
    if args.dry_run:
        return 0

    if dependency_versions != DINOV3_DEPENDENCIES:
        raise RuntimeError(
            "KARROW requires the pinned DINOv3 dependencies before launch: "
            f"expected={DINOV3_DEPENDENCIES} observed={dependency_versions}"
        )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {output_dir}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"DINOv3 model directory does not exist: {model_path}")

    runtime_environment = _runtime_info(python, env)
    runtime_environment["packages"].update(dependency_versions)
    output_dir.mkdir(parents=True)
    _write_json(config_path, config)
    launch["status"] = "running"
    launch["started_at_utc"] = datetime.now(timezone.utc).isoformat()
    launch["runtime_environment"] = runtime_environment
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
