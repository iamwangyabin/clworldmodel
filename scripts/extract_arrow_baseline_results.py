#!/usr/bin/env python3
"""Extract the local ARROW pilot log into a browser-readable result bundle."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any


EVAL_RE = re.compile(
    r"Eval for epoch:\s+(?P<epoch>\d+)\s*\n"
    r"Eval means: (?P<means>\[[^\n]+\])\s*\n"
    r"Eval stds: (?P<stds>\[[^\n]+\])"
)
STAGE_RE = re.compile(
    r"\[stage-time\] epoch=(?P<epoch>\d+) "
    r"collect=(?P<collect>[\d.]+)s eval=(?P<eval>[\d.]+)s "
    r"world_model=(?P<world_model>[\d.]+)s actor=(?P<actor>[\d.]+)s "
    r"overhead=(?P<overhead>[\d.]+)s total=(?P<total>[\d.]+)s"
)
MEMORY_RE = re.compile(
    r"\[cuda-mem\] (?P<label>[^|]+) \| .*?"
    r"allocated=(?P<allocated>[\d.]+) GiB reserved=(?P<reserved>[\d.]+) GiB "
    r"peak_allocated=(?P<peak_allocated>[\d.]+) GiB "
    r"peak_reserved=(?P<peak_reserved>[\d.]+) GiB "
    r"suggested_slurm_gpu_mem=(?P<suggested>[\d.]+) GiB"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--format", choices=("json", "js"), default="json", dest="output_format"
    )
    return parser.parse_args()


def read_run_metadata(log_text: str) -> dict[str, Any]:
    metadata, _ = json.JSONDecoder().raw_decode(log_text.lstrip())
    return metadata


def build_results(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "tensorboard" / "config.json"
    log_path = run_dir / f"{run_dir.name}.log"
    config = json.loads(config_path.read_text())
    log_text = log_path.read_text(errors="replace")
    metadata = read_run_metadata(log_text)

    evaluations = []
    for match in EVAL_RE.finditer(log_text):
        means = ast.literal_eval(match.group("means"))
        stds = ast.literal_eval(match.group("stds"))
        evaluations.append(
            {"epoch": int(match.group("epoch")), "means": means, "stds": stds}
        )

    env_configs = config["esc"]["env_configs"]
    if any(len(row["means"]) != len(env_configs) for row in evaluations):
        raise ValueError("Evaluation task count does not match the configured curriculum")

    swap = int(config["esc"]["kwargs"]["swap_sched"])
    tasks = []
    for index, env_config in enumerate(env_configs):
        task_evaluations = [
            {
                "epoch": row["epoch"],
                "mean": row["means"][index],
                "std": row["stds"][index],
            }
            for row in evaluations
        ]
        phase_start = index * swap
        phase_end = (index + 1) * swap
        phase_points = [
            point
            for point in task_evaluations
            if phase_start <= point["epoch"] <= phase_end
        ]
        phase_peak = max(phase_points, key=lambda point: point["mean"])
        final = task_evaluations[-1]
        tasks.append(
            {
                "index": index,
                "name": env_config["name"].removeprefix("ALE/").removesuffix("-v5"),
                "environment": env_config["name"],
                "rewardScale": env_config["rew_scale"],
                "trainStart": phase_start,
                "trainEnd": phase_end,
                "phasePeak": phase_peak,
                "final": final,
                "finalRawDerived": final["mean"] / env_config["rew_scale"],
                "phasePeakForgetting": phase_peak["mean"] - final["mean"],
                "phasePeakRetention": final["mean"] / phase_peak["mean"],
                "evaluations": task_evaluations,
            }
        )

    stage_rows = [
        {
            "epoch": int(match.group("epoch")),
            **{
                key: float(match.group(key))
                for key in ("collect", "eval", "world_model", "actor", "overhead", "total")
            },
        }
        for match in STAGE_RE.finditer(log_text)
    ]
    stage_totals = {
        key: sum(row[key] for row in stage_rows)
        for key in ("collect", "eval", "world_model", "actor", "overhead", "total")
    }
    stage_breakdown = [
        {
            "name": key,
            "hours": stage_totals[key] / 3600,
            "fraction": stage_totals[key] / stage_totals["total"],
        }
        for key in ("world_model", "actor", "collect", "eval", "overhead")
    ]

    memory_rows = [
        {
            "label": match.group("label").strip(),
            **{
                key: float(match.group(key))
                for key in (
                    "allocated",
                    "reserved",
                    "peak_allocated",
                    "peak_reserved",
                    "suggested",
                )
            },
        }
        for match in MEMORY_RE.finditer(log_text)
    ]

    epochs = int(config["epochs"])
    total_env_frames = (
        epochs
        * int(config["n_sync"])
        * int(config["gen_seq_len"])
        * int(config["env_repeat"])
    )
    replay_slots = sum(int(item["n"]) for item in metadata.get("replay_buffers", []))
    if replay_slots == 0:
        replay_slots = int(metadata["fifo_slots"]) + int(metadata["ltdm_slots"])
    image_bytes = (
        replay_slots
        * int(metadata["sequence_length"])
        * int(config["img_size"])
        * int(config["img_size"])
        * 3
        * 4
    )

    return {
        "provenance": {
            "run": run_dir.name,
            "method": metadata["method"],
            "runtime": metadata["runtime"],
            "upstreamCommit": metadata["upstream_commit"],
            "curriculum": metadata["curriculum"],
            "seedId": metadata["seed_id"],
            "seed": metadata["seed"],
            "complete": "[cuda-mem] training_end" in log_text,
            "sourceLog": f"runs/{run_dir.name}/{log_path.name}",
            "sourceConfig": f"runs/{run_dir.name}/tensorboard/config.json",
        },
        "protocol": {
            "epochs": epochs,
            "taskSwapEpochs": swap,
            "evaluationEveryEpochs": 10,
            "evaluationRollouts": 16,
            "evaluationPolicy": "stochastic",
            "imageSize": int(config["img_size"]),
            "frameRepeat": int(config["env_repeat"]),
            "worldModelUpdates": epochs * int(config["steps_per_batch"]),
            "actorUpdates": epochs * int(config["ac_train_steps"]),
            "rawEnvironmentFrames": total_env_frames,
            "agentDecisions": total_env_frames // int(config["env_repeat"]),
            "evaluationCheckpoints": len(evaluations),
        },
        "replay": {
            "fifoSlots": int(metadata["fifo_slots"]),
            "longTermSlots": int(metadata["ltdm_slots"]),
            "sequenceLength": int(metadata["sequence_length"]),
            "observations": replay_slots * int(metadata["sequence_length"]),
            "rawFloat32ImageGiB": image_bytes / (1024**3),
        },
        "resources": {
            "loggedStageHours": stage_totals["total"] / 3600,
            "stageBreakdown": stage_breakdown,
            "peakAllocatedGiB": max(row["peak_allocated"] for row in memory_rows),
            "peakReservedGiB": max(row["peak_reserved"] for row in memory_rows),
            "suggestedGpuMemoryGiB": max(row["suggested"] for row in memory_rows),
        },
        "tasks": tasks,
    }


def main() -> None:
    args = parse_args()
    results = build_results(args.run_dir.resolve())
    payload = json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output_format == "js":
        payload = f"window.ARROW_BASELINE_RESULTS = {payload};\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)


if __name__ == "__main__":
    main()
