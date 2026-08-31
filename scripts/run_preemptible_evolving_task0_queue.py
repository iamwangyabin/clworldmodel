#!/usr/bin/env python3
"""Run the EnvParallel16 Task-0 cohort only on opportunistically idle GPUs.

External CUDA processes always have priority. If one appears on an assigned
GPU, this supervisor terminates only its own process group, preserves that
attempt as partial/ineligible, and requeues the profile from scratch.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from git_provenance import require_synced_training_git_state
from run_evolving_task0_sweep import ENV16_PROTOCOL, PROFILE_OVERRIDES
from select_evolving_task0_profile import LR_EXPECTED_PROFILES


ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("fixed_v1", *PROFILE_OVERRIDES)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"at_utc": _utc_now(), **event}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _run_nvidia_smi(*arguments: str) -> str:
    completed = subprocess.run(
        ["nvidia-smi", *arguments],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def _split_csv(line: str) -> list[str]:
    return [part.strip() for part in line.split(",")]


def _gpu_snapshot() -> dict[int, dict[str, Any]]:
    gpu_output = _run_nvidia_smi(
        "--query-gpu=index,uuid,name,memory.used,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    )
    snapshot: dict[int, dict[str, Any]] = {}
    for line in gpu_output.splitlines():
        if not line.strip():
            continue
        index, uuid, name, memory, utilization, temperature, power = _split_csv(
            line
        )
        snapshot[int(index)] = {
            "uuid": uuid,
            "name": name,
            "memory_used_mib": int(memory),
            "utilization_percent": int(utilization),
            "temperature_c": int(temperature),
            "power_w": float(power),
            "compute_pids": [],
        }
    try:
        process_output = _run_nvidia_smi(
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        )
    except subprocess.CalledProcessError:
        # Some drivers return a nonzero status when there are no compute apps.
        process_output = ""
    by_uuid = {data["uuid"]: data for data in snapshot.values()}
    for line in process_output.splitlines():
        if not line.strip():
            continue
        uuid, pid, process_name, used_memory = _split_csv(line)
        if uuid not in by_uuid:
            continue
        by_uuid[uuid]["compute_pids"].append(
            {
                "pid": int(pid),
                "process_name": process_name,
                "used_gpu_memory_mib": int(used_memory),
            }
        )
    return snapshot


def _pid_process_group(pid: int) -> int | None:
    try:
        return os.getpgid(pid)
    except (ProcessLookupError, PermissionError):
        return None


def _external_compute_pids(
    gpu: dict[str, Any], *, owned_process_group: int | None
) -> list[int]:
    external: list[int] = []
    for process in gpu["compute_pids"]:
        pid = int(process["pid"])
        if owned_process_group is None or _pid_process_group(pid) != owned_process_group:
            external.append(pid)
    return sorted(external)


def _gpu_is_idle(gpu: dict[str, Any], *, idle_memory_mib: int) -> bool:
    return not gpu["compute_pids"] and gpu["memory_used_mib"] <= idle_memory_mib


def _memory_snapshot() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        name, raw = line.split(":", 1)
        values[name] = int(raw.strip().split()[0]) * 1024
    return {
        "total_bytes": values.get("MemTotal", 0),
        "available_bytes": values.get("MemAvailable", 0),
    }


@dataclass
class ActiveJob:
    profile: str
    gpu_index: int
    gpu_uuid: str
    attempt: int
    attempt_dir: Path
    run_dir: Path
    replay_dir: Path
    process: subprocess.Popen[bytes]
    process_group: int
    launcher_log: Any
    started_at_utc: str
    preemption_external_pids: list[int] | None = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--gpu", type=int, action="append", required=True)
    parser.add_argument("--max-active", type=int)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--idle-confirm-seconds", type=float, default=30.0)
    parser.add_argument("--idle-memory-mib", type=int, default=64)
    parser.add_argument("--termination-grace-seconds", type=float, default=20.0)
    return parser


def _terminate_job(job: ActiveJob, *, grace_seconds: float) -> int | None:
    if job.process.poll() is None:
        try:
            os.killpg(job.process_group, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + grace_seconds
        while job.process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.25)
        if job.process.poll() is None:
            try:
                os.killpg(job.process_group, signal.SIGKILL)
            except ProcessLookupError:
                pass
    try:
        return job.process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        return None


def _attempt_status(
    job: ActiveJob,
    *,
    outcome: str,
    return_code: int | None,
    eligible: bool,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "profile": job.profile,
        "attempt": job.attempt,
        "gpu_index": job.gpu_index,
        "gpu_uuid": job.gpu_uuid,
        "launcher_pid": job.process.pid,
        "process_group": job.process_group,
        "started_at_utc": job.started_at_utc,
        "finished_at_utc": _utc_now(),
        "outcome": outcome,
        "return_code": return_code,
        "eligible": eligible,
        "reason": reason,
        "external_compute_pids": job.preemption_external_pids or [],
        "resume_used": False,
        "relaunch_policy": (
            "from_scratch_in_new_attempt_directory"
            if outcome
            in {"preempted_by_external_gpu_process", "gpu_monitoring_lost"}
            else None
        ),
        "run_dir": str(job.run_dir),
        "replay_dir": str(job.replay_dir),
    }


def _campaign_state(
    *,
    status: str,
    pending: list[str],
    active: dict[int, ActiveJob],
    completed: dict[str, str],
    failed: dict[str, dict[str, Any]],
    attempt_counts: dict[str, int],
    gpu_snapshot: dict[int, dict[str, Any]],
    campaign_root: Path,
) -> dict[str, Any]:
    disk = shutil.disk_usage(campaign_root)
    memory = _memory_snapshot()
    return {
        "schema_version": 1,
        "protocol": ENV16_PROTOCOL,
        "status": status,
        "updated_at_utc": _utc_now(),
        "pending_profiles": list(pending),
        "active": {
            str(index): {
                "profile": job.profile,
                "attempt": job.attempt,
                "launcher_pid": job.process.pid,
                "process_group": job.process_group,
                "started_at_utc": job.started_at_utc,
                "run_dir": str(job.run_dir),
            }
            for index, job in sorted(active.items())
        },
        "completed": dict(completed),
        "failed": failed,
        "attempt_counts": dict(attempt_counts),
        "gpu_snapshot": gpu_snapshot,
        "host": {
            "load_average_1m_5m_15m": list(os.getloadavg()),
            "memory": memory,
            "campaign_filesystem": {
                "total_bytes": disk.total,
                "used_bytes": disk.used,
                "free_bytes": disk.free,
            },
        },
    }


def main() -> int:
    args = _parser().parse_args()
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")
    if args.poll_seconds < 1:
        raise ValueError("--poll-seconds must be at least one second")
    if args.idle_confirm_seconds < args.poll_seconds:
        raise ValueError("--idle-confirm-seconds must span at least one poll")
    if len(set(args.gpu)) != len(args.gpu):
        raise ValueError("--gpu indices must be unique")
    max_active = args.max_active or len(args.gpu)
    if not 1 <= max_active <= len(args.gpu):
        raise ValueError("--max-active must lie between one and the GPU count")
    python = args.python.expanduser().resolve()
    if not python.is_file():
        raise FileNotFoundError(f"Python interpreter does not exist: {python}")
    campaign_root = args.campaign_root.expanduser().resolve()
    if campaign_root.exists():
        raise FileExistsError(
            f"Refusing to reuse campaign root: {campaign_root}"
        )

    subprocess.run(["git", "fetch", "--prune"], cwd=ROOT, check=True)
    project_git = require_synced_training_git_state(ROOT)
    initial_gpu_snapshot = _gpu_snapshot()
    missing_gpus = sorted(set(args.gpu) - set(initial_gpu_snapshot))
    if missing_gpus:
        raise ValueError(f"Requested GPU indices do not exist: {missing_gpus}")
    campaign_root.mkdir(parents=True)
    events_path = campaign_root / "events.jsonl"
    state_path = campaign_root / "monitor" / "latest.json"
    _write_json_atomic(
        campaign_root / "campaign.json",
        {
            "schema_version": 1,
            "protocol": ENV16_PROTOCOL,
            "classification": "pilot",
            "created_at_utc": _utc_now(),
            "project_git": project_git,
            "profiles": list(PROFILES),
            "profile_set_matches_preregistration": set(PROFILES)
            == set(LR_EXPECTED_PROFILES),
            "gpus_allowed": args.gpu,
            "max_active": max_active,
            "python": str(python),
            "cpu_threads_per_candidate": args.cpu_threads,
            "poll_seconds": args.poll_seconds,
            "idle_confirm_seconds": args.idle_confirm_seconds,
            "idle_memory_mib": args.idle_memory_mib,
            "external_gpu_process_priority": True,
            "preemption_scope": "assigned_gpu_only",
            "preempted_attempt_eligible": False,
            "preempted_attempt_resume_used": False,
        },
    )
    _append_event(events_path, {"event": "supervisor_started"})

    pending = list(PROFILES)
    active: dict[int, ActiveJob] = {}
    completed: dict[str, str] = {}
    failed: dict[str, dict[str, Any]] = {}
    attempt_counts = {profile: 0 for profile in PROFILES}
    idle_since: dict[int, float | None] = {index: None for index in args.gpu}
    consecutive_gpu_poll_failures = 0
    stop_requested = False

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        _append_event(
            events_path,
            {"event": "supervisor_stop_requested", "signal": signum},
        )

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    while not stop_requested and (pending or active):
        try:
            gpu_snapshot = _gpu_snapshot()
        except Exception as exc:
            consecutive_gpu_poll_failures += 1
            _append_event(
                events_path,
                {
                    "event": "gpu_poll_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            if active and (
                consecutive_gpu_poll_failures * args.poll_seconds
                >= max(10.0, 2 * args.poll_seconds)
            ):
                requeued: list[str] = []
                for gpu_index, job in sorted(active.items()):
                    return_code = _terminate_job(
                        job, grace_seconds=args.termination_grace_seconds
                    )
                    job.launcher_log.close()
                    _write_json_atomic(
                        job.attempt_dir / "operator_attempt_status.json",
                        _attempt_status(
                            job,
                            outcome="gpu_monitoring_lost",
                            return_code=return_code,
                            eligible=False,
                            reason=(
                                "GPU process monitoring failed closed; the "
                                "candidate yielded rather than run unmonitored."
                            ),
                        ),
                    )
                    requeued.append(job.profile)
                    idle_since[gpu_index] = None
                pending = requeued + pending
                active.clear()
                _append_event(
                    events_path,
                    {
                        "event": "active_candidates_stopped_fail_closed",
                        "profiles_requeued": requeued,
                        "consecutive_gpu_poll_failures": (
                            consecutive_gpu_poll_failures
                        ),
                    },
                )
            time.sleep(args.poll_seconds)
            continue
        consecutive_gpu_poll_failures = 0

        for gpu_index, job in list(active.items()):
            external_pids = _external_compute_pids(
                gpu_snapshot[gpu_index],
                owned_process_group=job.process_group,
            )
            if external_pids:
                job.preemption_external_pids = external_pids
                _append_event(
                    events_path,
                    {
                        "event": "external_gpu_process_detected",
                        "gpu_index": gpu_index,
                        "profile": job.profile,
                        "attempt": job.attempt,
                        "external_compute_pids": external_pids,
                    },
                )
                return_code = _terminate_job(
                    job, grace_seconds=args.termination_grace_seconds
                )
                job.launcher_log.close()
                status = _attempt_status(
                    job,
                    outcome="preempted_by_external_gpu_process",
                    return_code=return_code,
                    eligible=False,
                    reason=(
                        "An external CUDA process appeared on the assigned GPU; "
                        "the campaign yielded without signaling that process."
                    ),
                )
                _write_json_atomic(
                    job.attempt_dir / "operator_attempt_status.json", status
                )
                pending.insert(0, job.profile)
                del active[gpu_index]
                idle_since[gpu_index] = None
                _append_event(
                    events_path,
                    {
                        "event": "candidate_preempted_and_requeued",
                        "gpu_index": gpu_index,
                        "profile": job.profile,
                        "attempt": job.attempt,
                        "return_code": return_code,
                    },
                )
                continue

            return_code = job.process.poll()
            if return_code is None:
                continue
            job.launcher_log.close()
            run_status_path = job.run_dir / "run_status.json"
            try:
                run_status = json.loads(run_status_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                run_status = {}
            eligible = return_code == 0 and run_status.get("complete") is True
            if eligible:
                outcome = "complete"
                reason = "Launcher and eligibility gate completed successfully."
                completed[job.profile] = str(job.run_dir)
                event = "candidate_completed"
            else:
                outcome = "failed"
                reason = "Candidate exited without a complete eligible run status."
                failed[job.profile] = {
                    "attempt": job.attempt,
                    "run_dir": str(job.run_dir),
                    "return_code": return_code,
                    "run_status": run_status,
                }
                event = "candidate_failed"
            _write_json_atomic(
                job.attempt_dir / "operator_attempt_status.json",
                _attempt_status(
                    job,
                    outcome=outcome,
                    return_code=return_code,
                    eligible=eligible,
                    reason=reason,
                ),
            )
            del active[gpu_index]
            idle_since[gpu_index] = None
            _append_event(
                events_path,
                {
                    "event": event,
                    "gpu_index": gpu_index,
                    "profile": job.profile,
                    "attempt": job.attempt,
                    "return_code": return_code,
                },
            )

        now = time.monotonic()
        for gpu_index in args.gpu:
            if gpu_index in active:
                idle_since[gpu_index] = None
                continue
            if _gpu_is_idle(
                gpu_snapshot[gpu_index], idle_memory_mib=args.idle_memory_mib
            ):
                if idle_since[gpu_index] is None:
                    idle_since[gpu_index] = now
            else:
                idle_since[gpu_index] = None

        while pending and len(active) < max_active:
            eligible_gpu = next(
                (
                    index
                    for index in args.gpu
                    if index not in active
                    and idle_since[index] is not None
                    and now - float(idle_since[index])
                    >= args.idle_confirm_seconds
                ),
                None,
            )
            if eligible_gpu is None:
                break
            profile = pending.pop(0)
            attempt_counts[profile] += 1
            attempt = attempt_counts[profile]
            attempt_dir = (
                campaign_root / "attempts" / profile / f"attempt_{attempt:03d}"
            )
            run_dir = attempt_dir / "run"
            replay_dir = attempt_dir / "replay"
            attempt_dir.mkdir(parents=True)
            gpu_uuid = str(gpu_snapshot[eligible_gpu]["uuid"])
            command = [
                str(python),
                str(ROOT / "scripts" / "run_evolving_task0_sweep.py"),
                "--profile",
                profile,
                "--collection-envs",
                "16",
                "--seed",
                "0",
                "--output-dir",
                str(run_dir),
                "--replay-mmap-root",
                str(replay_dir),
                "--python",
                str(python),
                "--cpu-threads",
                str(args.cpu_threads),
            ]
            environment = os.environ.copy()
            environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            environment["CUDA_VISIBLE_DEVICES"] = gpu_uuid
            launcher_log = (attempt_dir / "launcher.log").open("wb")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=launcher_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            process_group = os.getpgid(process.pid)
            started_at_utc = _utc_now()
            job = ActiveJob(
                profile=profile,
                gpu_index=eligible_gpu,
                gpu_uuid=gpu_uuid,
                attempt=attempt,
                attempt_dir=attempt_dir,
                run_dir=run_dir,
                replay_dir=replay_dir,
                process=process,
                process_group=process_group,
                launcher_log=launcher_log,
                started_at_utc=started_at_utc,
            )
            active[eligible_gpu] = job
            idle_since[eligible_gpu] = None
            _write_json_atomic(
                attempt_dir / "operator_attempt_launch.json",
                {
                    "schema_version": 1,
                    "protocol": ENV16_PROTOCOL,
                    "profile": profile,
                    "attempt": attempt,
                    "gpu_index": eligible_gpu,
                    "gpu_uuid": gpu_uuid,
                    "launcher_pid": process.pid,
                    "process_group": process_group,
                    "started_at_utc": started_at_utc,
                    "command": command,
                    "cuda_visible_devices": gpu_uuid,
                    "from_scratch": True,
                    "resume_used": False,
                },
            )
            _append_event(
                events_path,
                {
                    "event": "candidate_started",
                    "gpu_index": eligible_gpu,
                    "gpu_uuid": gpu_uuid,
                    "profile": profile,
                    "attempt": attempt,
                    "launcher_pid": process.pid,
                },
            )

        status = "running" if not failed else "running_with_failed_profiles"
        _write_json_atomic(
            state_path,
            _campaign_state(
                status=status,
                pending=pending,
                active=active,
                completed=completed,
                failed=failed,
                attempt_counts=attempt_counts,
                gpu_snapshot=gpu_snapshot,
                campaign_root=campaign_root,
            ),
        )
        time.sleep(args.poll_seconds)

    if stop_requested:
        for job in list(active.values()):
            return_code = _terminate_job(
                job, grace_seconds=args.termination_grace_seconds
            )
            job.launcher_log.close()
            _write_json_atomic(
                job.attempt_dir / "operator_attempt_status.json",
                _attempt_status(
                    job,
                    outcome="supervisor_stopped",
                    return_code=return_code,
                    eligible=False,
                    reason="The supervisor was stopped; no run is left unmonitored.",
                ),
            )
        final_status = "stopped"
        return_code = 130
    elif failed:
        final_status = "failed"
        return_code = 1
    elif set(completed) == set(PROFILES):
        selection_path = campaign_root / "task0_selection.json"
        selection_command = [
            str(python),
            str(ROOT / "scripts" / "select_evolving_task0_profile.py"),
        ]
        for profile in PROFILES:
            selection_command.extend(["--candidate-dir", completed[profile]])
        selection_command.extend(["--output", str(selection_path)])
        selection_log_path = campaign_root / "selection.log"
        with selection_log_path.open("wb") as selection_log:
            selection_result = subprocess.run(
                selection_command,
                cwd=ROOT,
                stdout=selection_log,
                stderr=subprocess.STDOUT,
            )
        if selection_result.returncode == 0 and selection_path.is_file():
            final_status = "complete"
            return_code = 0
        else:
            failed["selection"] = {
                "return_code": selection_result.returncode,
                "log": str(selection_log_path),
            }
            final_status = "selection_failed"
            return_code = 1
    else:
        final_status = "failed"
        return_code = 1

    try:
        final_gpu_snapshot = _gpu_snapshot()
    except Exception:
        final_gpu_snapshot = {}
    _write_json_atomic(
        state_path,
        _campaign_state(
            status=final_status,
            pending=pending,
            active={},
            completed=completed,
            failed=failed,
            attempt_counts=attempt_counts,
            gpu_snapshot=final_gpu_snapshot,
            campaign_root=campaign_root,
        ),
    )
    _append_event(
        events_path,
        {"event": "supervisor_finished", "status": final_status},
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
