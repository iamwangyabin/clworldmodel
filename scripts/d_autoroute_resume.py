# SPDX-License-Identifier: Apache-2.0
"""Failure-preserving D-AutoRoute boundary continuation orchestration.

No model loads, environment construction or optimizer updates. The trainer
independently validates the complete checkpoint before restoring any state.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_config(value: dict) -> dict:
    result = copy.deepcopy(value)
    result.setdefault("benchmark", "atari")
    result.setdefault("interaction_counter_mode", "legacy_trajectory_positions")
    for task in result["esc"]["env_configs"]:
        task.setdefault("adapter", "atari")
    return result


def inspect_resume(checkpoint: Path, config: dict, protocol: str) -> dict:
    checkpoint = checkpoint.expanduser().resolve()
    match = re.fullmatch(r"task_(\d{2})_post_consolidation\.pt", checkpoint.name)
    if match is None or checkpoint.parent.name != "evolving_core_checkpoints":
        raise ValueError("Resume needs a post-consolidation training checkpoint, not inference/pre weights")
    source = checkpoint.parent.parent
    launch = json.loads((source / "launch.json").read_text())
    original = json.loads((source / "resolved_training_config.json").read_text())
    status = json.loads((source / "run_status.json").read_text())
    if (config["continual_method"] != "evolving_atomic_rssm_adaptive_compression_shared_heads_autoroute_arrow"
            or launch["protocol"] != protocol or launch["seed"] != config["seed"]
            or _normalized_config(original) != _normalized_config(config)):
        raise ValueError("Resume cannot change D-AutoRoute seed, protocol, budget or resolved config")
    if status.get("complete") is not False or status.get("return_code") in (None, 0):
        raise ValueError("Resume source must be a confirmed failed attempt, never a running/completed run")
    task = int(match.group(1))
    schedule = config["esc"]
    duration = schedule["kwargs"]["swap_sched"]
    if (schedule["env_schedule_type"] != "SequentialEnvironments" or type(duration) is not int
            or not 0 <= task < len(schedule["env_configs"])):
        raise ValueError("Resume source must use the fixed sequential acquisition schedule")
    completed = (task + 1) * duration
    if not 0 < completed < config["epochs"]:
        raise ValueError("Checkpoint must have pending training epochs")
    expected = checkpoint.with_suffix(".pt.sha256").read_text().split()[0]
    actual = sha256(checkpoint)
    if actual != expected:
        raise ValueError("Resume source checkpoint checksum mismatch")
    text = (source / "train.log").read_text(errors="replace")
    cut = re.search(rf"^Starting Epoch\s+{completed}\s*$", text, re.M)
    prefix = text[:cut.start()] if cut else text
    stages = [int(n) + 1 for n in re.findall(r"\[stage-time\] epoch=(\d+)", text)]
    if max(stages, default=0) < completed:
        raise ValueError("Source log does not confirm the requested durable boundary")
    return {
        "schema_version": 1, "artifact_kind": "d_autoroute_resume_lineage",
        "source_run": str(source), "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": actual, "source_project_git": launch["project_git"],
        "source_run_status": status, "completed_epochs": completed, "completed_task_id": task,
        "parent_completed_epochs": max(stages),
        "discarded_completed_online_epochs": max(stages) - completed,
        "remaining_epochs": config["epochs"] - completed,
        "prefix_log_characters": len(prefix),
        "source_log_sha256": sha256(source / "train.log"),
        "caveat": "Boundary continuation with fresh environment resets; not bitwise mid-epoch recovery. Failed suffix compute remains overhead, never new result seeds or extra logical updates.",
        "failed_suffix_accounting": {
            "completed_online_wm_updates": max(0, max(stages) - completed) * config["steps_per_batch"],
            "completed_online_ac_updates": max(0, max(stages) - completed) * config["ac_train_steps"],
            "partial_collection_and_evaluation": "Preserved in original full log/diagnostics; incomplete steps may not have exact counters. Not claimed zero.",
        },
    }


def stage_resume_prefix(output: Path, lineage: dict) -> Path:
    """Copy only records before the restored boundary; never relabel old weights.

    The reporting log is an explicit concatenation of the immutable inherited
    prefix and the new attempt. Full failed logs/manifests are kept separately.
    Each commit keeps its own snapshot index; old inference weights stay in the
    parent run and are referenced with their original provenance.
    """
    source = Path(lineage["source_run"])
    completed, task_id = lineage["completed_epochs"], lineage["completed_task_id"]
    parent = output / "resume_parent"
    parent.mkdir()
    for name in ("launch.json", "resolved_training_config.json", "run_status.json", "train.log",
                 "resume_lineage.json", "deployment_provenance.json"):
        path = source / name
        if path.is_file():
            shutil.copy2(path, parent / name)
    prefix = output / "inherited_training_prefix.log"
    text = (parent / "train.log").read_text(errors="replace")
    if sha256(parent / "train.log") != lineage["source_log_sha256"]:
        raise ValueError("Parent log changed during resume staging")
    prefix.write_text(text[:lineage["prefix_log_characters"]])
    copied = []
    for folder in ("adaptive_qfp_compression", "evolving_core_consolidation", "task_routing"):
        for path in sorted((source / folder).glob("*.json")):
            epoch = re.search(r"epoch_(\d+)", path.name)
            task = re.search(r"task_(\d+)", path.name)
            if (folder == "task_routing" and epoch and int(epoch.group(1)) < completed
                    or folder != "task_routing" and task and int(task.group(1)) <= task_id):
                destination = output / folder / path.name
                destination.parent.mkdir(exist_ok=True)
                shutil.copy2(path, destination)
                copied.append({"path": str(destination.relative_to(output)), "sha256": sha256(destination),
                               "source": str(path)})
    # A consolidation that rolled back has its record under checkpoints instead.
    for path in (source / "evolving_core_checkpoints").glob("task_*_consolidation_failure.json"):
        if int(path.name.split("_")[1]) <= task_id:
            destination = output / "evolving_core_checkpoints" / path.name
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(path, destination)
            copied.append({"path": str(destination.relative_to(output)), "sha256": sha256(destination),
                           "source": str(path)})
    lineage.update(inherited_records=copied, inherited_prefix_sha256=sha256(prefix),
                   source_inference_snapshots=str(source / "task_boundary_snapshots"),
                   inference_snapshots_relabelled=False)
    (output / "resume_lineage.json").write_text(json.dumps(lineage, indent=2) + "\n")
    return prefix
