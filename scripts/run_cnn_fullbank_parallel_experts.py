#!/usr/bin/env python3
"""Launch six independent CNN-FullBank experts across a fixed GPU pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from git_provenance import git_state, require_synced_training_git_state
from run_arrow_ar50_atari import ROOT
from run_moe_arrow_atari import INDEPENDENT_EXPERT_PROFILE


PROFILE = "six-parallel-independent-single-gpu-experts-v1"
TASKS = (
    ("ALE/MsPacman-v5", "mspacman"),
    ("ALE/Boxing-v5", "boxing"),
    ("ALE/CrazyClimber-v5", "crazyclimber"),
    ("ALE/Frostbite-v5", "frostbite"),
    ("ALE/Seaquest-v5", "seaquest"),
    ("ALE/Enduro-v5", "enduro"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def _gpu_ids(value: str) -> list[int]:
    try:
        ids = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "GPU ids must be comma-separated integers"
        ) from error
    if not ids or len(ids) != len(set(ids)) or any(item < 0 for item in ids):
        raise argparse.ArgumentTypeError("GPU ids must be unique non-negative integers")
    return ids


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one independent task-aware CNN-FullBank expert per process, "
            "using at most one process on each listed GPU"
        )
    )
    parser.add_argument("--profile", choices=(PROFILE,), required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--replay-mmap-root", type=Path, required=True)
    parser.add_argument("--arrow-reference-matrix", type=Path, required=True)
    parser.add_argument("--gpu-ids", type=_gpu_ids, default=_gpu_ids("0,1,2,3"))
    parser.add_argument("--seed", type=int, choices=(0,), default=0)
    parser.add_argument("--cpu-threads", type=int, default=12)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _task_command(
    *,
    python: Path,
    task_index: int,
    output_dir: Path,
    replay_root: Path,
    reference: Path,
    cpu_threads: int,
) -> list[str]:
    return [
        str(python),
        str(ROOT / "scripts" / "run_moe_arrow_atari.py"),
        "--seed",
        "0",
        "--method",
        "cnn-fullbank",
        "--devices",
        "1",
        "--independent-expert-profile",
        INDEPENDENT_EXPERT_PROFILE,
        "--independent-task-index",
        str(task_index),
        "--task-duration-multiplier",
        "2",
        "--evaluation-audit-profile",
        "fixed-cohort-snapshots",
        "--arrow-reference-matrix",
        str(reference),
        "--replay-mmap-root",
        str(replay_root),
        "--profile-stages",
        "--cpu-threads",
        str(cpu_threads),
        "--output-dir",
        str(output_dir),
    ]


def main() -> int:
    args = _parser().parse_args()
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    if not args.campaign_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("--campaign-id may contain only letters, digits, '-' and '_'")

    project_git = (
        git_state(ROOT) if args.dry_run else require_synced_training_git_state(ROOT)
    )
    output_root = args.output_root.expanduser().resolve()
    campaign_dir = output_root / args.campaign_id
    replay_campaign_root = (
        args.replay_mmap_root.expanduser().resolve() / args.campaign_id
    )
    reference = args.arrow_reference_matrix.expanduser().resolve()
    python = args.python.expanduser().resolve()
    reference_payload = json.loads(reference.read_text(encoding="utf-8"))
    expected_names = [name for name, _ in TASKS]
    if reference_payload.get("task_order") != expected_names:
        raise ValueError("The frozen ARROW reference task order is incompatible")

    task_plans: list[dict[str, Any]] = []
    for task_index, (task_name, slug) in enumerate(TASKS):
        output_dir = campaign_dir / f"task_{task_index:02d}_{slug}"
        task_plans.append(
            {
                "task_index": task_index,
                "task_name": task_name,
                "assembly_slot": task_index,
                "output_dir": str(output_dir),
                "launcher_log": str(
                    campaign_dir / f"task_{task_index:02d}_{slug}.launcher.log"
                ),
                "command": _task_command(
                    python=python,
                    task_index=task_index,
                    output_dir=output_dir,
                    replay_root=replay_campaign_root,
                    reference=reference,
                    cpu_threads=args.cpu_threads,
                ),
                "status": "pending",
                "gpu_id": None,
                "pid": None,
                "return_code": None,
                "started_at_utc": None,
                "finished_at_utc": None,
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "profile": PROFILE,
        "classification": "single_seed_parallel_independent_expert_bank_pilot",
        "campaign_id": args.campaign_id,
        "started_at_utc": None,
        "finished_at_utc": None,
        "project_git": project_git,
        "seed_id": args.seed,
        "gpu_ids": args.gpu_ids,
        "maximum_concurrent_tasks": len(args.gpu_ids),
        "task_order": expected_names,
        "task_duration_epochs": 180,
        "per_task_device_count": 1,
        "per_task_world_model_global_batch": 16,
        "per_task_actor_global_batch": 128,
        "per_task_world_model_updates": 180_000,
        "per_task_actor_critic_updates": 144_000,
        "frozen_arrow_reference": {
            "source": str(reference),
            "sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        },
        "semantics": {
            "sequential_continual_learning": False,
            "parallel_independent_training": True,
            "task_identity_required": True,
            "cross_task_transfer_measured": False,
            "retention_and_forgetting_measured": False,
            "incremental_assembly": (
                "append completed immutable experts to their fixed task slots"
            ),
            "strict_fair_superiority_claim_allowed": False,
            "reason": (
                "each expert receives twice the original task duration and no "
                "sequential warm start or shared-parameter interference"
            ),
        },
        "checkpoint_semantics": {
            "child_boundary_snapshots_resumable": False,
            "assembly_requires_all_six_verified_child_snapshots": True,
        },
        "tasks": task_plans,
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    if campaign_dir.exists():
        raise FileExistsError(f"Refusing to overwrite campaign: {campaign_dir}")
    if replay_campaign_root.exists():
        raise FileExistsError(
            f"Refusing to reuse campaign replay scratch: {replay_campaign_root}"
        )
    campaign_dir.mkdir(parents=True)
    replay_campaign_root.mkdir(parents=True)
    manifest["started_at_utc"] = _utc_now()
    manifest_path = campaign_dir / "campaign_manifest.json"
    _write_json_atomic(manifest_path, manifest)

    active: dict[int, tuple[subprocess.Popen[bytes], Any, int]] = {}
    pending = list(range(len(task_plans)))
    free_gpus = list(args.gpu_ids)
    interrupted = False

    def terminate_active(signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        for process, _log, _gpu in active.values():
            _terminate_process_group(process)

    signal.signal(signal.SIGTERM, terminate_active)
    signal.signal(signal.SIGINT, terminate_active)

    failure_code = 0
    while pending or active:
        while pending and free_gpus and not interrupted and failure_code == 0:
            task_index = pending.pop(0)
            gpu_id = free_gpus.pop(0)
            task = task_plans[task_index]
            log_handle = Path(task["launcher_log"]).open("wb")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            process = subprocess.Popen(
                task["command"],
                cwd=ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            task.update(
                {
                    "status": "running",
                    "gpu_id": gpu_id,
                    "pid": process.pid,
                    "started_at_utc": _utc_now(),
                }
            )
            active[task_index] = (process, log_handle, gpu_id)
            _write_json_atomic(manifest_path, manifest)
            print(
                f"Started task {task_index} ({task['task_name']}) on GPU {gpu_id} "
                f"with PID {process.pid}",
                flush=True,
            )

        completed: list[int] = []
        for task_index, (process, log_handle, gpu_id) in active.items():
            return_code = process.poll()
            if return_code is None:
                continue
            log_handle.close()
            task = task_plans[task_index]
            task["return_code"] = return_code
            task["finished_at_utc"] = _utc_now()
            task["status"] = "complete" if return_code == 0 else "failed"
            free_gpus.append(gpu_id)
            completed.append(task_index)
            print(
                f"Task {task_index} ({task['task_name']}) exited rc={return_code}",
                flush=True,
            )
            if return_code != 0 and failure_code == 0:
                failure_code = return_code

        for task_index in completed:
            del active[task_index]
        if failure_code != 0 or interrupted:
            for process, _log, _gpu in active.values():
                _terminate_process_group(process)
            for task_index, (process, log_handle, _gpu) in active.items():
                return_code = process.wait()
                log_handle.close()
                task_plans[task_index].update(
                    {
                        "status": (
                            "interrupted"
                            if interrupted
                            else "terminated_after_peer_failure"
                        ),
                        "return_code": return_code,
                        "finished_at_utc": _utc_now(),
                    }
                )
            for task_index in pending:
                task_plans[task_index]["status"] = "not_started"
            break
        _write_json_atomic(manifest_path, manifest)
        if pending or active:
            time.sleep(args.poll_seconds)

    manifest["finished_at_utc"] = _utc_now()
    if interrupted:
        final_status = "interrupted"
        return_code = 128 + signal.SIGTERM
    elif failure_code:
        final_status = "failed"
        return_code = failure_code
    else:
        final_status = "complete"
        return_code = 0
    manifest["status"] = final_status
    manifest["return_code"] = return_code
    _write_json_atomic(manifest_path, manifest)
    _write_json_atomic(
        campaign_dir / "run_status.json",
        {
            "complete": return_code == 0,
            "status": final_status,
            "return_code": return_code,
            "finished_at_utc": manifest["finished_at_utc"],
        },
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
