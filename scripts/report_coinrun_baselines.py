#!/usr/bin/env python3
"""Recompute the archived 2026-09-03 CoinRun baseline report, without run folders."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clworldmodel.evaluation.metrics import (  # noqa: E402
    median_iqr,
    normalize_return_matrix,
    single_pass_metrics,
)

RECORDS = ROOT / "docs/experiments/records"
REPORT = ROOT / "docs/experiments/coinrun_baselines_5seed_20260906.md"
SEEDS = [123456789, 1337, 31337, 42, 987654321]
RANDOM = [2.78, 2.45, 2.70, 2.62, 2.50, 2.69]
SINGLE = [6.09, 7.14, 6.89, 6.85, 7.89, 5.78]
TASK_END_ROWS = [9, 18, 27, 36, 45, 54]
LAUNCH_COMMIT = "7b3ebec334c54942a40c888d8972aac24761c5ad"


def read_run(path: Path) -> dict:
    """Check every stored episode against its curated taskwise mean/std/count."""
    record = json.loads(path.read_text(encoding="utf-8"))
    log_path = path.parent / "evaluation.log"
    raw_bytes = log_path.read_bytes()
    sources = {item["role"]: item for item in record["source_artifacts"]}
    if hashlib.sha256(raw_bytes).hexdigest() != sources["curated_raw_returns"]["sha256"]:
        raise ValueError(f"{path.parent.name}: curated raw-return checksum mismatch")
    lines = raw_bytes.decode("utf-8").splitlines()
    header = json.loads(lines[0])
    if header["source_sha256"] != sources["original_raw_returns"]["sha256"]:
        raise ValueError(f"{path.parent.name}: original raw-return checksum mismatch")
    rows = [json.loads(line) for line in lines[1:]]
    expected = [(epoch, task) for epoch in range(0, 541, 10) for task in range(6)]
    if [(row["epoch"], row["task_index"]) for row in rows] != expected:
        raise ValueError(f"{path.parent.name}: missing, duplicate or reordered evaluation")
    checkpoints = record["evaluation"]["checkpoints"]
    if len(checkpoints) != 55:
        raise ValueError(f"{path.parent.name}: expected 55 evaluation checkpoints")
    matrix, episodes = [], 0
    for index, checkpoint in enumerate(checkpoints):
        epoch = index * 10
        if (checkpoint["completed_epochs"], checkpoint["world_model_updates"],
                checkpoint["actor_critic_updates"]) != (epoch, epoch * 1000, epoch * 800):
            raise ValueError(f"{path.parent.name}: incorrect evaluation counters")
        if [task["task_index"] for task in checkpoint["tasks"]] != list(range(6)):
            raise ValueError(f"{path.parent.name}: incorrect task indices")
        means = []
        for task, row in zip(checkpoint["tasks"], rows[index * 6:index * 6 + 6]):
            values = row["returns"]
            if len(values) not in (256, 257) or any(
                type(value) is not int or value not in (0, 10) for value in values
            ):
                raise ValueError(f"{path.parent.name}: invalid raw returns/count")
            mean, std = statistics.mean(values), statistics.pstdev(values)
            if (task["episode_count"] != len(values)
                    or not math.isclose(mean, task["raw_return_mean"], abs_tol=1e-12)
                    or not math.isclose(std, task["raw_return_std"], abs_tol=1e-12)):
                raise ValueError(f"{path.parent.name}: raw mean/std/count mismatch")
            means.append(mean)
            episodes += len(values)
        matrix.append(means)
    normalized = normalize_return_matrix(matrix, RANDOM, SINGLE)
    metrics = single_pass_metrics(normalized, TASK_END_ROWS)
    stored = record["derived_metrics"]
    if stored["normalization"]["random"] != RANDOM or stored["normalization"]["single_task"] != SINGLE:
        raise ValueError(f"{path.parent.name}: normalization anchors changed")
    if stored["fixed_table_A16_diagnostic"] != metrics:
        raise ValueError(f"{path.parent.name}: stale derived metrics")
    return {"record": record, "matrix": matrix, "episodes": episodes, "metrics": metrics}


def load_campaign(records_root: Path = RECORDS) -> list[dict]:
    runs = []
    for method in ("arrow-ar50", "dv3-fifo"):
        for seed_id, seed_value in enumerate(SEEDS):
            path = records_root / f"coinrun-{method}-original-s{seed_id}-20260903" / "record.json"
            run = read_run(path)
            record = run["record"]
            if record["seed"] != {"id": seed_id, "value": seed_value}:
                raise ValueError(f"{path.parent.name}: seed mismatch")
            if record["project_git"]["commit"] != LAUNCH_COMMIT:
                raise ValueError(f"{path.parent.name}: launch provenance mismatch")
            runs.append(run)
    return runs


def render_report(runs: list[dict]) -> str:
    lines = [
        "# CoinRun baseline archive: ARROW-50 and DreamerV3/FIFO, five seeds",
        "",
        "Archived 2026-09-06. This is **10 runs / two methods / seeds 0–4**, not the",
        "earlier standalone seed-0 run and not the Atari or task-aware method campaigns.",
        f"All **3,300 task/checkpoint records and {sum(run['episodes'] for run in runs):,} individual episode returns**",
        "are preserved in Git as small text evidence. No seed is selected or discarded.",
        "",
        "## Recompute and verify",
        "",
        "```sh",
        "python3 scripts/report_coinrun_baselines.py check",
        "python3 scripts/experiment_registry.py check",
        "python3 -m unittest tests.test_coinrun_baseline_archive tests.test_experiment_registry",
        "```",
        "",
        "Use `write` instead of `check` to regenerate this report. Only the committed",
        "`record.json` and `evaluation.log` files are needed; no server, TensorBoard,",
        "downloaded run folder, GPU or third-party Python package is required.",
        "",
        "## Completion and limitations",
        "",
        "All ten logs reach epoch 540 and `training_end`, with all 55 × 6 scheduled",
        "evaluations and finite recorded core losses. **Training/evaluation is complete;",
        "launcher finalization is a separate check.** VirtAI seeds 0–3 retain stale",
        "`state=running`, no exit code and no per-job final checksum manifest after",
        "post-training filesystem I/O errors. These source states were NOT rewritten.",
        "Both 4090-2 seed-4 jobs have `completed`, exit code 0 and verified final",
        "checksum manifests. Record status `complete` refers to training/evaluation",
        "coverage; `strict_finalization_verified` distinguishes the eight incomplete",
        "launcher records. All ten are labelled **diagnostic evidence**, not a verified",
        "reproduction of the paper's numerical results.",
        "",
        "The metric is **raw episodic return, not classification accuracy**. Returns",
        "are exactly 0 or 10. Each evaluation has 256 or 257 completed episodes due",
        "to vectorized threshold overshoot; no extra episode was dropped. Within-task",
        "standard deviation uses `ddof=0`; across-seed SD below uses `ddof=1`.",
        "",
        "## Final scheduled evaluation (epoch 540)",
        "",
        "This evaluation is before epoch-540 optimizer updates: 540,000 world-model",
        "and 432,000 actor-critic updates. Training subsequently ends at 541,000 and",
        "432,800 updates. The last collection revisits task 0 but its extra gradients",
        "do not enter the reported epoch-540 evaluation. There is no separate held-out",
        "post-training evaluation and no saved model/resumable checkpoint in these runs.",
        "",
        "Raw columns are six variants of the **same CoinRun game and reward scale**.",
        "Their descriptive mean is not an average across unrelated Atari games.",
        "Taskwise episode counts and mean ± population SD at every checkpoint are",
        "in each linked record; ordered individual returns are in its `evaluation.log`.",
        "",
        "| Method / seed | CoinRun | +NB | +NB+RT | +NB+RT+GA | +NB+RT+GA+MA | +NB+RT+GA+MA+CA | Six-variant raw mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        record, final = run["record"], run["matrix"][-1]
        label = f"{record['method']} / {record['seed']['id']}"
        link = f"records/{record['record_id']}/record.json"
        lines.append(f"| [{label}]({link}) | " + " | ".join(f"{v:.6f}" for v in [*final, statistics.mean(final)]) + " |")
    lines += ["", "| Method | Across-seed mean ± sample SD | Median [Q25, Q75] |", "|---|---:|---:|"]
    for offset in (0, 5):
        group = runs[offset:offset + 5]
        values = [statistics.mean(run["matrix"][-1]) for run in group]
        q = median_iqr(values)
        lines.append(f"| {group[0]['record']['method']} | {statistics.mean(values):.6f} ± {statistics.stdev(values):.6f} | {q['median']:.6f} [{q['q25']:.6f}, {q['q75']:.6f}] |")
    lines += [
        "", "## Separate fixed-anchor normalization diagnostic", "",
        "Constants are the rounded median endpoints from [ARROW v3 Table A.16](https://arxiv.org/html/2603.11395v3):",
        f"random `{RANDOM}`; single-task `{SINGLE}`. Apply `(return − random) /",
        "(single-task − random)` per task, then average tasks, then aggregate seeds.",
        "Values are not clipped. These are **not the authors' per-seed/time-aligned",
        "normalization curves**, so they must not be labelled exact paper replication",
        "numbers. Forward transfer is unavailable without aligned single-task curves.",
        "F/min-ACC/WC-ACC use the same fixed anchors for the entire checkpoint matrix.",
        "", "| Method / seed | F ↓ | ACC ↑ | min-ACC ↑ | WC-ACC ↑ |",
        "|---|---:|---:|---:|---:|",
    ]
    for run in runs:
        record = run["record"]
        values = [run["metrics"][key] for key in ("forgetting", "acc", "min_acc", "wc_acc")]
        lines.append(f"| {record['method']} / {record['seed']['id']} | " + " | ".join(f"{v:.6f}" for v in values) + " |")
    lines += ["", "| Method | Median F | Median ACC | Median min-ACC | Median WC-ACC |", "|---|---:|---:|---:|---:|"]
    for offset in (0, 5):
        group = runs[offset:offset + 5]
        values = [statistics.median(run["metrics"][key] for run in group) for key in ("forgetting", "acc", "min_acc", "wc_acc")]
        lines.append(f"| {group[0]['record']['method']} | " + " | ".join(f"{v:.6f}" for v in values) + " |")
    lines += [
        "", "## Provenance and storage", "",
        f"Launch commit: [`{LAUNCH_COMMIT}`](https://github.com/iamwangyabin/clworldmodel/commit/{LAUNCH_COMMIT});",
        "upstream ARROW pin: `cb05e7d97ed83c3cf6e528960db0da6868e29232`.",
        f"Seed IDs 0–4 map to `{SEEDS}`. Seeds 0–3 ran on VirtAI's four reported",
        "`S2.gpu.xlarge` devices, two jobs per GPU; both seed-4 jobs ran on 4090-2.",
        "Do not infer identical accelerator/runtime behavior from identical seed values.",
        "",
        "The [CPU uint8 protocol](../protocols/arrow_coinrun_cpu_uint8_replay.md)",
        "retains 541 epochs, 90/task, 1,024 replay trajectory slots and 6,480,199,680",
        "allocated replay tensor bytes/job. ARROW: FIFO 512 + LTDM 512 with 50/50",
        "whole-minibatch selection; DV3: FIFO 1,024. Only decoded sampled minibatches",
        "enter CUDA. Explicit Python/Procgen seeding is a documented deviation from",
        "released defaults, not a promise of bitwise determinism. Resolved configs,",
        "published-config differences, counter meanings, package/hardware provenance,",
        "loss checks and original artifact SHA256s are embedded in every record.",
        "",
        "`evaluation.log` is compact JSONL: one provenance header, followed by epoch,",
        "task index and an ordered integer array of raw returns. Rendering original",
        "0.0/10.0 as 0/10 is numerically exact; no sampling or lossy compression is",
        "used. Means/std/counts are independently recomputed by the command above.",
        "Full training logs, TensorBoard events, runtime dumps and downloaded tarballs",
        "remain outside Git and are identified by hashes. This archive does not claim",
        "that large run packages or nonexistent model weights were uploaded to GitHub.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("write", "check"))
    args = parser.parse_args()
    report = render_report(load_campaign())
    if args.mode == "write":
        REPORT.write_text(report, encoding="utf-8")
    elif REPORT.read_text(encoding="utf-8") != report:
        raise SystemExit("Stale CoinRun report; run with mode 'write'.")
    print(f"CoinRun archive {args.mode}: 10 runs, all raw returns verified.")


if __name__ == "__main__":
    main()
