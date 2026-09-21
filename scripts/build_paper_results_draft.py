#!/usr/bin/env python3
"""Generate the current paper-results draft from preserved raw artifacts."""

from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from clworldmodel.evaluation.metrics import (  # noqa: E402
    median_iqr,
    normalize_return_matrix,
    single_pass_metrics,
)
from summarize_continual_metrics import build_run_report  # noqa: E402


OUTPUT = ROOT / "docs" / "experiments" / "AWM_AUTOROUTE_PAPER_RESULTS_DRAFT.md"
NORMALIZATION = (
    ROOT
    / "docs"
    / "protocols"
    / "references"
    / "arrow_v3_atari_normalization_v1.json"
)
S0_COMPARISON = (
    ROOT
    / "docs"
    / "protocols"
    / "references"
    / "local_s0_continual_metric_comparison_v1.json"
)
BASELINE_ARCHIVES = (
    "runs/cloud_result_backups/20260915/completed/h2/arrow_ar50_cpu_fp32_original_s1.tar",
    "runs/cloud_result_backups/20260915/completed/h4/arrow_ar50_cpu_fp32_original_s2.tar",
    "runs/cloud_result_backups/20260915/completed/h1/arrow_ar50_cpu_fp32_original_s3_attempt2.tar",
    "runs/cloud_result_backups/20260915/completed/h1/arrow_ar50_cpu_fp32_original_s4.tar",
    "runs/cloud_result_backups/20260915/completed/h3/dv3_s1.tar",
    "runs/cloud_result_backups/20260915/completed/h4/dv3_s2.tar",
    "runs/cloud_result_backups/20260915/completed/h2/dv3_s3.tar",
    "runs/cloud_result_backups/20260915/completed/h3/dv3_s4.tar",
)
AWM_ATARI_SEEDS = ("s0", "s7", "s11", "s12", "s30")
AWM_COINRUN_SEEDS = (
    "seed2026091701",
    "seed2026091704",
    "seed2026091705",
)
COINRUN_EXCLUDED_RECORD_IDS = {"coinrun-dv3-fifo-original-s2-20260903"}
TABLE9_SEEDS = ("s0", "s7", "s11", "s12")
METRICS = ("forgetting", "acc", "min_acc", "wc_acc")
COINRUN_RANDOM = (2.78, 2.45, 2.70, 2.62, 2.50, 2.69)
COINRUN_SINGLE = (6.09, 7.14, 6.89, 6.85, 7.89, 5.78)
CAPACITY_NAMES = {
    "Frozen shared core + residuals": "Frozen core + residuals",
    "Full per-task WMs + Private AC": "FullBank + private AC",
    "Independent residuals + plastic shared core": "Independent residuals",
    "Shared WM + Private AC": "Shared WM + private AC",
    "Wider Shared WM + Private AC": "Wider shared + private AC",
}
CAPACITY_WM_PARAMS = {
    "Shared WM + private AC": 19_498_853,
    "Wider shared + private AC": 34_742_117,
    "FullBank + private AC": 116_993_118,
    "Frozen core + residuals": 42_601_625,
    "Independent residuals": 42_601_625,
}
PRIVATE_AC_PARAMS = 10_295_910
CUDA_MEMORY_RE = re.compile(
    r"peak_allocated=([0-9.]+) GiB peak_reserved=([0-9.]+) GiB"
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _extract_run(archive: Path, destination: Path) -> Path:
    keep = {
        "train.log",
        "resolved_training_config.json",
        "config.json",
        "launch.json",
        "final_evaluation.json",
        "run_status.json",
        "model_parameter_accounting.json",
        "actor_critic_parameter_accounting.json",
    }
    with tarfile.open(archive) as source:
        members = [
            member
            for member in source.getmembers()
            if member.isfile()
            and len(Path(member.name).parts) == 2
            and Path(member.name).name in keep
        ]
        if not members:
            raise ValueError(f"No run metadata found in {archive}")
        run_dir = destination / Path(members[0].name).parts[0]
        run_dir.mkdir(parents=True)
        for member in members:
            stream = source.extractfile(member)
            if stream is None:
                raise ValueError(f"Cannot read {member.name} from {archive}")
            (run_dir / Path(member.name).name).write_bytes(stream.read())
    return run_dir


def _aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(reports),
        "seeds": [report.get("seed_id") for report in reports],
        "metrics": {
            metric: median_iqr(
                [float(report["metrics"][metric]) for report in reports]
            )
            for metric in METRICS
        },
    }


def _atari_results() -> tuple[
    dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]
]:
    groups: dict[str, list[dict[str, Any]]] = {
        "ARROW-50": [],
        "DreamerV3/FIFO": [],
    }
    for report in _json(S0_COMPARISON)["reports"]:
        if report["method"] in groups:
            groups[report["method"]].append(report)
    with tempfile.TemporaryDirectory() as temporary:
        destination = Path(temporary)
        for relative in BASELINE_ARCHIVES:
            report = build_run_report(_extract_run(ROOT / relative, destination))
            groups[report["method"]].append(report)
    if any(len(reports) != 5 for reports in groups.values()):
        raise ValueError("Atari baselines must each contain exactly five seeds")

    awm_reports = []
    for seed_dir in AWM_ATARI_SEEDS:
        report = _json(
            ROOT
            / "runs"
            / "main_results"
            / "atari"
            / "awm_autoroute"
            / seed_dir
            / "continual_metrics.json"
        )
        report["display_seed_id"] = seed_dir.upper()
        awm_reports.append(report)
    groups["AWM-AutoRoute"] = awm_reports
    return (
        {method: _aggregate(reports) for method, reports in groups.items()},
        awm_reports,
        groups,
    )


def _capacity_report(run_dir: Path) -> dict[str, Any]:
    config = _json(run_dir / "resolved_training_config.json")
    launch = _json(run_dir / "launch.json")
    task_names = [str(task["name"]) for task in config["esc"]["env_configs"]]
    references = {
        str(task["task_name"]): task for task in _json(NORMALIZATION)["tasks"]
    }
    random_returns = [float(references[name]["random_return"]) for name in task_names]
    single_returns = [
        float(references[name]["single_task_arrow_return"]) for name in task_names
    ]
    lines = (run_dir / "train.log").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    evaluations: list[tuple[int, list[float]]] = []
    for index, line in enumerate(lines):
        if not line.startswith("Eval for epoch:"):
            continue
        epoch = int(line.split(":", 1)[1])
        block = lines[index + 1 : index + 7]
        raw_line = next(
            candidate for candidate in block if candidate.startswith("Eval raw means:")
        )
        observed = [
            float(value)
            for value in ast.literal_eval(raw_line.split(":", 1)[1].strip())
        ]
        if len(observed) > len(task_names):
            raise ValueError(f"Too many task returns in {run_dir} at epoch {epoch}")
        evaluations.append(
            (epoch, observed + random_returns[len(observed) :])
        )

    final = _json(run_dir / "final_evaluation.json")
    final_epoch = int(final["evaluation_after_completed_epochs"])
    final_raw = [float(task["raw_return_mean"]) for task in final["tasks"]]
    evaluations = [row for row in evaluations if row[0] != final_epoch]
    evaluations.append((final_epoch, final_raw))
    evaluations.sort(key=lambda row: row[0])
    normalized = normalize_return_matrix(
        [row for _, row in evaluations], random_returns, single_returns
    )
    row_by_epoch = {epoch: index for index, (epoch, _) in enumerate(evaluations)}
    boundaries = [90, 180, 270, 360, 450, 540]
    if any(epoch not in row_by_epoch for epoch in boundaries):
        raise ValueError(f"Missing task boundary evaluation in {run_dir}")
    metrics = single_pass_metrics(
        normalized, [row_by_epoch[epoch] for epoch in boundaries]
    )
    return {
        "method": CAPACITY_NAMES[str(launch["method"])],
        "seed_id": int(launch["seed_id"]),
        "metrics": metrics,
    }


def _capacity_results() -> dict[str, Any]:
    archives = sorted(
        (ROOT / "runs" / "cloud_result_backups").glob(
            "*/completed/*/capacity_*.tar"
        )
    )
    groups: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, int]] = set()
    with tempfile.TemporaryDirectory() as temporary:
        destination = Path(temporary)
        for archive in archives:
            report = _capacity_report(_extract_run(archive, destination))
            key = (report["method"], report["seed_id"])
            if key in seen:
                raise ValueError(f"Duplicate capacity result: {key}")
            seen.add(key)
            groups.setdefault(report["method"], []).append(report)
    return {
        method: _aggregate(sorted(reports, key=lambda row: row["seed_id"]))
        for method, reports in groups.items()
    }


def _coinrun_results() -> tuple[
    dict[str, Any], dict[str, Any], list[dict[str, Any]]
]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(
        (ROOT / "docs" / "experiments" / "records").glob(
            "coinrun-*/record.json"
        )
    ):
        record = _json(path)
        if record["record_id"] in COINRUN_EXCLUDED_RECORD_IDS:
            continue
        final = next(
            checkpoint
            for checkpoint in record["evaluation"]["checkpoints"]
            if int(checkpoint["completed_epochs"]) == 540
        )
        raw = [float(task["raw_return_mean"]) for task in final["tasks"]]
        groups.setdefault(str(record["method"]), []).append(
            {
                "raw_final_average": statistics.mean(raw),
                "raw_final_vector": raw,
                "metrics": record["derived_metrics"]["fixed_table_A16_diagnostic"],
            }
        )
    baseline = {}
    for method, rows in groups.items():
        raw = [float(row["raw_final_average"]) for row in rows]
        baseline[method] = {
            "n": len(rows),
            "raw": median_iqr(raw),
            "raw_mean": statistics.mean(raw),
            "raw_sample_sd": statistics.stdev(raw),
            "raw_tasks": [
                median_iqr([float(row["raw_final_vector"][index]) for row in rows])
                for index in range(6)
            ],
            "metrics": {
                metric: median_iqr(
                    [float(row["metrics"][metric]) for row in rows]
                )
                for metric in METRICS
            },
        }
    awm_rows = []
    for seed_dir in AWM_COINRUN_SEEDS:
        run_dir = (
            ROOT
            / "runs"
            / "main_results"
            / "coinrun"
            / "awm_autoroute"
            / seed_dir
        )
        report = _json(run_dir / "continual_metrics.json")
        final = _json(run_dir / "final_evaluation.json")
        vector = [float(task["raw_return_mean"]) for task in final["tasks"]]
        normalized = normalize_return_matrix(
            [
                [float(value) for value in checkpoint["raw_return_mean"]]
                for checkpoint in report["evaluation_checkpoints"]
            ],
            COINRUN_RANDOM,
            COINRUN_SINGLE,
        )
        metrics = single_pass_metrics(
            normalized,
            [int(index) for index in report["metrics"]["acquisition_rows"]],
        )
        awm_rows.append(
            {
                "seed": int(report["seed"]),
                "raw_final_average": statistics.mean(vector),
                "raw_final_vector": vector,
                "raw_forgetting": float(report["metrics"]["mean_raw_forgetting"]),
                "metrics": metrics,
            }
        )
    raw = [float(row["raw_final_average"]) for row in awm_rows]
    awm = {
        "n": len(awm_rows),
        "raw": median_iqr(raw),
        "raw_mean": statistics.mean(raw),
        "raw_sample_sd": statistics.stdev(raw),
        "raw_forgetting": median_iqr(
            [float(row["raw_forgetting"]) for row in awm_rows]
        ),
        "raw_tasks": [
            median_iqr(
                [float(row["raw_final_vector"][index]) for row in awm_rows]
            )
            for index in range(6)
        ],
        "metrics": {
            metric: median_iqr(
                [float(row["metrics"][metric]) for row in awm_rows]
            )
            for metric in METRICS
        },
    }
    if awm["n"] != len(AWM_COINRUN_SEEDS):
        raise ValueError("CoinRun AWM cohort size does not match selected seeds")
    return baseline, awm, awm_rows


def _metric(value: dict[str, float]) -> str:
    return f'{value["median"]:.4f} [{value["q25"]:.4f}, {value["q75"]:.4f}]'


def _interval(value: dict[str, float], digits: int) -> str:
    return (
        f'{value["median"]:.{digits}f} '
        f'[{value["q25"]:.{digits}f}, {value["q75"]:.{digits}f}]'
    )


def _final_raw_task_summary(
    reports: list[dict[str, Any]],
) -> list[dict[str, float]]:
    vectors = [
        [float(value) for value in report["evaluation_checkpoints"][-1]["raw_return_mean"]]
        for report in reports
    ]
    if any(len(vector) != 6 for vector in vectors):
        raise ValueError("Expected six final raw task returns")
    return [median_iqr([vector[index] for vector in vectors]) for index in range(6)]


def _ablation_counts() -> list[tuple[str, int, int, int]]:
    variants = (
        ("NoReuse", "no_reuse"),
        ("NoFunctionalProtection", "no_functional_protection"),
        ("NoConflictProjection", "no_conflict_projection"),
        ("NoRCC", "no_rcc"),
    )
    root = ROOT / "runs" / "cloud_result_backups"
    rows = []
    for label, key in variants:
        completed = {
            int(match.group(1))
            for path in root.glob(f"*/completed/*/{key}_s*.tar")
            if (match := re.search(r"_s(\d+)", path.name))
        }
        staged = {
            int(match.group(1))
            for path in root.glob(
                f"*/completed/**/{key}_s*.partial/continual_metrics.json"
            )
            if (match := re.search(r"_s(\d+)", str(path)))
        }
        failed = {
            int(match.group(1))
            for path in root.glob(
                f"*/metadata_snapshots/**/failed/{key}_s*/run_status.json"
            )
            if (match := re.search(r"_s(\d+)", str(path)))
        }
        rows.append((label, len(completed), len(staged), len(failed)))
    return rows


def _run_hours(launch: dict[str, Any], status: dict[str, Any]) -> float:
    started = datetime.fromisoformat(str(launch["started_at_utc"]))
    finished = datetime.fromisoformat(str(status["finished_at_utc"]))
    return (finished - started).total_seconds() / 3600


def _cuda_memory(log_path: Path) -> tuple[float, float]:
    values = [
        (float(allocated), float(reserved))
        for allocated, reserved in CUDA_MEMORY_RE.findall(
            log_path.read_text(encoding="utf-8", errors="replace")
        )
    ]
    if not values:
        raise ValueError(f"No CUDA memory profile in {log_path}")
    return max(row[0] for row in values), max(row[1] for row in values)


def _baseline_resource_results() -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, float]]] = {}
    with tempfile.TemporaryDirectory() as temporary:
        destination = Path(temporary)
        for relative in BASELINE_ARCHIVES:
            run = _extract_run(ROOT / relative, destination)
            launch = _json(run / "launch.json")
            status = _json(run / "run_status.json")
            config = _json(run / "resolved_training_config.json")
            wm = _json(run / "model_parameter_accounting.json")
            ac = _json(run / "actor_critic_parameter_accounting.json")
            slots = (
                2 * int(config["data_n_max"])
                if config["algorithm"] == "arrow"
                else int(config["sac_dv3_data_n_max"])
            )
            dtype_bytes = {"float32": 4, "uint8": 1}[
                str(config["replay_observation_dtype"])
            ]
            replay_bytes = (
                slots
                * int(config["data_t"])
                * 3
                * int(config["img_size"])
                * int(config["img_size"])
                * dtype_bytes
            )
            allocated, reserved = _cuda_memory(run / "train.log")
            groups.setdefault(str(launch["method"]), []).append(
                {
                    "world_model_parameters": float(wm["world_model"]["parameters"]),
                    "actor_critic_parameters": float(ac["actor_critic"]["parameters"]),
                    "parameters": float(wm["world_model"]["parameters"])
                    + float(ac["actor_critic"]["parameters"]),
                    "parameter_state_bytes": float(
                        wm["world_model_parameter_and_buffer_state"][
                            "parameter_and_buffer_bytes"
                        ]
                    )
                    + float(ac["actor_critic"]["parameter_and_buffer_bytes"]),
                    "replay_bytes": float(replay_bytes),
                    "peak_allocated_gib": allocated,
                    "peak_reserved_gib": reserved,
                    "gpu_hours": _run_hours(launch, status),
                }
            )
    return {
        method: {
            "n": len(rows),
            **{
                key: median_iqr([row[key] for row in rows])
                for key in rows[0]
            },
        }
        for method, rows in groups.items()
    }


def _awm_resource_summary(
    seeds: tuple[str, ...] = AWM_ATARI_SEEDS,
) -> dict[str, Any]:
    world_model_parameters: list[float] = []
    actor_critic_parameters: list[float] = []
    online_parameters: list[float] = []
    parameter_state_bytes: list[float] = []
    widths = {
        name: [[] for _ in range(6)]
        for name in ("representation", "recurrent", "transition")
    }
    replay_payload_bytes = set()
    peak_allocated_gib: list[float] = []
    peak_reserved_gib: list[float] = []
    gpu_hours: list[float] = []
    for seed_dir in seeds:
        run = ROOT / "runs" / "main_results" / "atari" / "awm_autoroute" / seed_dir
        wm = _json(run / "model_parameter_accounting.json")
        ac = _json(run / "actor_critic_parameter_accounting.json")
        replay = _json(run / "replay_mmap_storage_accounting.json")
        world_model_parameters.append(float(wm["world_model"]["parameters"]))
        actor_critic_parameters.append(float(ac["aggregate_actor_critic_parameters"]))
        online_parameters.append(
            float(wm["world_model"]["parameters"])
            + float(ac["aggregate_actor_critic_parameters"])
        )
        parameter_state_bytes.append(
            float(
                wm["world_model_parameter_and_buffer_state"][
                    "parameter_and_buffer_bytes"
                ]
            )
            + sum(
                float(row["actor_critic"]["parameter_and_buffer_bytes"])
                for row in ac["per_task"].values()
            )
        )
        banks = wm["rssm_task_mechanism_banks"]
        for task_index in range(6):
            for name, bank in banks.items():
                widths[name][task_index].append(
                    float(bank["mechanism_hidden_features_per_task"][task_index])
                )
        replay_payload_bytes.add(
            sum(int(buffer["logical_storage_bytes"]) for buffer in replay["buffers"])
        )
        allocated, reserved = _cuda_memory(run / "train.log")
        peak_allocated_gib.append(allocated)
        peak_reserved_gib.append(reserved)
        gpu_hours.append(
            _run_hours(_json(run / "launch.json"), _json(run / "run_status.json"))
        )
    if len(replay_payload_bytes) != 1:
        raise ValueError("AWM replay payload differs across selected seeds")
    return {
        "n": len(seeds),
        "world_model_parameters": median_iqr(world_model_parameters),
        "actor_critic_parameters": median_iqr(actor_critic_parameters),
        "parameters": median_iqr(online_parameters),
        "parameter_state_bytes": median_iqr(parameter_state_bytes),
        "widths": {
            name: [median_iqr(values) for values in rows]
            for name, rows in widths.items()
        },
        "replay_bytes": median_iqr([float(replay_payload_bytes.pop())]),
        "peak_allocated_gib": median_iqr(peak_allocated_gib),
        "peak_reserved_gib": median_iqr(peak_reserved_gib),
        "gpu_hours": median_iqr(gpu_hours),
    }


def _routing_summary(seeds: tuple[str, ...] = AWM_ATARI_SEEDS) -> dict[str, Any]:
    confusion = [[0 for _ in range(6)] for _ in range(6)]
    seed_accuracies = []
    observation_correct = {1: 0.0, 2: 0.0}
    observation_counts = {1: 0, 2: 0}
    for seed_dir in seeds:
        final = _json(
            ROOT
            / "runs"
            / "main_results"
            / "atari"
            / "awm_autoroute"
            / seed_dir
            / "final_evaluation.json"
        )
        seed_confusion = [[0 for _ in range(6)] for _ in range(6)]
        for task in final["routing"]:
            audit = task["audit"]
            for true_task in range(6):
                for selected_route in range(6):
                    count = int(audit["confusion_matrix"][true_task][selected_route])
                    confusion[true_task][selected_route] += count
                    seed_confusion[true_task][selected_route] += count
            for observation in (1, 2):
                row = audit[f"observation_{observation}"]
                count = int(row["count"])
                observation_counts[observation] += count
                observation_correct[observation] += count * float(row["accuracy"])
        total = sum(sum(row) for row in seed_confusion)
        seed_accuracies.append(
            sum(seed_confusion[index][index] for index in range(6)) / total
        )
    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[index][index] for index in range(6))
    return {
        "confusion": confusion,
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "seed_accuracy": median_iqr(seed_accuracies),
        "observation_accuracy": {
            observation: observation_correct[observation]
            / observation_counts[observation]
            for observation in (1, 2)
        },
    }


def render() -> str:
    atari, awm_seeds, atari_reports = _atari_results()
    coinrun, coinrun_awm, coinrun_awm_rows = _coinrun_results()
    capacity = _capacity_results()
    baseline_resources = _baseline_resource_results()
    awm_resources = _awm_resource_summary()
    table9_resources = _awm_resource_summary(TABLE9_SEEDS)
    routing = _routing_summary(TABLE9_SEEDS)
    resource_lines = []
    for method in ("DreamerV3/FIFO", "ARROW-50"):
        row = baseline_resources[method]
        world_model = row["world_model_parameters"]
        actor_critic = row["actor_critic_parameters"]
        parameters = row["parameters"]
        state_bytes = row["parameter_state_bytes"]
        allocated = row["peak_allocated_gib"]
        reserved = row["peak_reserved_gib"]
        hours = row["gpu_hours"]
        resource_lines.append(
            f'| {method} | {world_model["median"] / 1e6:.2f}M | '
            f'{actor_critic["median"] / 1e6:.2f}M | '
            f'{parameters["median"] / 1e6:.2f}M '
            f'({state_bytes["median"] / 2**20:.2f} MiB) | '
            f'{row["replay_bytes"]["median"] / 2**30:.1f} GiB float32 | '
            f'{allocated["median"]:.2f} [{allocated["q25"]:.2f}, {allocated["q75"]:.2f}] / '
            f'{reserved["median"]:.2f} [{reserved["q25"]:.2f}, {reserved["q75"]:.2f}] GiB | '
            f'{hours["median"]:.2f} [{hours["q25"]:.2f}, {hours["q75"]:.2f}] (n={row["n"]}) |'
        )
    awm_parameters = awm_resources["parameters"]
    awm_world_model = awm_resources["world_model_parameters"]
    awm_actor_critic = awm_resources["actor_critic_parameters"]
    awm_state = awm_resources["parameter_state_bytes"]
    awm_allocated = awm_resources["peak_allocated_gib"]
    awm_reserved = awm_resources["peak_reserved_gib"]
    awm_hours = awm_resources["gpu_hours"]
    resource_lines.append(
        f'| AWM-AutoRoute pilot | {awm_world_model["median"] / 1e6:.2f}M '
        f'[{awm_world_model["q25"] / 1e6:.2f}, {awm_world_model["q75"] / 1e6:.2f}] | '
        f'{awm_actor_critic["median"] / 1e6:.2f}M | '
        f'{awm_parameters["median"] / 1e6:.2f}M '
        f'[{awm_parameters["q25"] / 1e6:.2f}, {awm_parameters["q75"] / 1e6:.2f}] '
        f'({awm_state["median"] / 2**20:.2f} MiB '
        f'[{awm_state["q25"] / 2**20:.2f}, {awm_state["q75"] / 2**20:.2f}]) | '
        f'{awm_resources["replay_bytes"]["median"] / 2**30:.1f} GiB uint8 | '
        f'{awm_allocated["median"]:.2f} [{awm_allocated["q25"]:.2f}, {awm_allocated["q75"]:.2f}] / '
        f'{awm_reserved["median"]:.2f} [{awm_reserved["q25"]:.2f}, {awm_reserved["q75"]:.2f}] GiB | '
        f'{awm_hours["median"]:.2f} [{awm_hours["q25"]:.2f}, {awm_hours["q75"]:.2f}] '
        f'(n={awm_resources["n"]}) |'
    )
    confusion_diagonal = ", ".join(
        str(routing["confusion"][index][index]) for index in range(6)
    )
    table9_parameters = table9_resources["parameters"]
    table9_world_model = table9_resources["world_model_parameters"]
    width_lines = []
    task_names = ("MsPacman", "Boxing", "CrazyClimber", "Frostbite", "Seaquest", "Enduro")
    for task_index, task_name in enumerate(task_names):
        q_width = table9_resources["widths"]["representation"][task_index]
        f_width = table9_resources["widths"]["recurrent"][task_index]
        p_width = table9_resources["widths"]["transition"][task_index]
        width_lines.append(
            f'| T{task_index}: {task_name} | '
            f'{q_width["median"]:.0f} [{q_width["q25"]:.0f}, {q_width["q75"]:.0f}] | '
            f'{f_width["median"]:.0f} [{f_width["q25"]:.0f}, {f_width["q75"]:.0f}] | '
            f'{p_width["median"]:.0f} [{p_width["q25"]:.0f}, {p_width["q75"]:.0f}] |'
        )
    lines = [
        "# 4 实验设置（Experimental Setup）",
        "",
        "> **作者工作稿。** 本章已按正式论文结构排版；标记为 *pilot* 或 *pending* 的内容必须在投稿前由预声明正式结果替换，不能作为无条件优越性结论。",
        "",
        "> **占位规则。** `PENDING` 图槽只固定最终版式、坐标与应报告统计量，不包含虚构、插值或由未完成运行推算的数据。partial/staged artifacts 只有通过完整性与协议审计后才会自动替换占位。",
        "",
        "我们从五个方面评估 AWM-AutoRoute：（1）在异构 Atari 任务序列上的获取与保持；（2）在 CoinRun 共享结构序列上的跨 benchmark 泛化；（3）性能是否可由模型容量或完整任务隔离解释；（4）各机制的独立贡献；以及（5）自动路由相对 oracle route 的损失。所有主实验采用固定顺序、单轮六任务设置，以隔离单次任务迁移中的灾难性遗忘。第 4 节只定义协议、预算与指标；数值结果和解释统一放在第 5 节，以避免把实验设计与结果主张混在一起。",
        "",
        "## 4.1 Benchmarks 与持续学习协议",
        "",
        "**表 1：主实验课程。**",
        "",
        "| Benchmark | 任务顺序 | 训练长度 | 评估目的 |",
        "|---|---|---|---|",
        "| Atari | MsPacman → Boxing → CrazyClimber → Frostbite → Seaquest → Enduro | 90 epochs/任务，单轮六任务 | 跨游戏获取、保持与灾难性遗忘 |",
        "| CoinRun | CoinRun → +NB → +RT → +GA → +MA → +CA | 90 epochs/变体，单轮六任务 | 随视觉和动力学复杂度递增时的迁移与保持 |",
        "",
        "Atari 使用 64×64 RGB 观测、完整动作空间和 frame repeat 4。每个 epoch 包含 16,384 次 agent decisions；六任务边界对应 8,847,360 次 decisions 和 35,389,440 个 raw environment frames。图中横轴统一报告 agent decisions。CoinRun 中 NB 表示移除背景，RT 表示限制主题，GA 表示生成式素材，MA 表示单色素材，CA 表示相机不再始终居中于 agent。六个 CoinRun 变体共享奖励尺度，因此可同时报告逐变体回报和六变体 raw average；Atari 各游戏回报量纲不同，只作同游戏逐列比较。按照当前研究范围，我们不报告反向顺序或 two-cycle 结果。",
        "",
        "## 4.2 比较方法",
        "",
        "**DreamerV3/FIFO** 使用单一世界模型与 1,024 条 trajectory 的 FIFO replay；**ARROW-50** 在相同 trajectory 容量下配置 512 FIFO 和 512 LTDM，并以 0.5/0.5 选择子缓冲区；**AWM-AutoRoute** 在训练阶段利用任务边界组织累积世界模型，在交互与评估时通过两帧概率重建自动选择 route，不向策略提供 task ID。TES-SAC 未在本项目中重新运行，因此不进入本地主表；这意味着当前本地 baseline 覆盖窄于 ARROW 论文，不能把“未比较”写成“已经胜过”。",
        "",
        "## 4.3 训练预算与可比性",
        "",
        "**表 2：Atari 六任务边界处的预算和任务信息。**",
        "",
        "| 方法 | 在线 WM 更新 | 额外 WM 更新 | AC 更新 | Replay | Task information | 评估协议 |",
        "|---|---:|---:|---:|---|---|---|",
        "| DreamerV3/FIFO | 540,000 | 0 | 432,000 | FIFO 1,024×512，float32 | 不提供给 agent | stochastic，advancing cohort |",
        "| ARROW-50 | 540,000 | 0 | 432,000 | FIFO 512×512 + LTDM 512×512，float32 | 不提供给 agent | stochastic，advancing cohort |",
        "| AWM-AutoRoute | 540,000 | 12,000 | 432,000 | 总计 1,024×512，uint8 | 训练/replay task-aware；推理 task-ID-free | auto-route，fixed validation + held-out final |",
        "",
        "**表 2a：Atari 资源核算，median `[Q25, Q75]`。** 主参数对比采用完整 world model 的保存参数量，而不是单条 route 的瞬时 active 参数。AC 和总在线参数另列，避免隐藏 AWM 保留六个 private actor–critic 的成本。参数来自保存的 runtime accounting，Replay payload 由 resolved runtime config/accounting 的实际 shape 与 dtype 计算，显存来自逐 epoch CUDA profiler；GPU-hours 为启动到完成的单卡 wall-clock 小时。",
        "",
        "| 方法 | World-model parameters | Actor–critic parameters | Total online parameters (state) | Replay observation payload | Peak CUDA allocated / reserved | GPU-hours |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *resource_lines,
        "",
        "AWM 的 world-model 参数增长不是复制六套完整 DreamerV3。五-seed中位数可拆为：19.50M 共享基础 world model，加上 13.76M representation、2.50M recurrent、1.88M transition 的任务残差机制库，以及 0.21M 六任务 observation projectors，共 37.85M。也就是说，新增的 18.35M 参数几乎全部来自按任务累积的 Q/F/P 残差机制；共享 encoder、decoder、reward/continue heads 仍只保留一份。各 seed 经自适应压缩后保留的机制宽度不同，因此 AWM 总量存在 35.00M–38.80M 的 IQR。",
        "",
        "AWM 的 `task-aware training` 表示训练调度器和 replay 知道任务边界及样本归属，用于创建和保护机制；它不表示部署时直接把 task ID 输入策略。部署和报告评估使用自动路由，但这一训练假设仍比完全 task-agnostic baseline 更强，必须单独披露。",
        "",
        "AWM 的额外 12,000 次 WM optimizer steps 包括 6,000 次 boundary consolidation 和 6,000 次 adaptive compression；压缩过程另使用 480 个 validation rollouts。因而当前 AWM pilot 与两项基线并非严格 compute-matched，评估 cohort（用于考试的一组环境随机 seeds/episodes）也不同。不同 cohort 可能具有不同难度，因此同一数值差不能全部归因于方法。表 2 的差异属于方法成本，不能在主结果中隐去。表 2a 已从现有 accounting、CUDA profiler 和运行时间戳直接补齐；baseline 资源统计只有带 profiler 的 S1–S4（n=4），而 AWM 为五个 seed。GPU-hours 跨 S2/S4 云卡、RTX 4090 与 RTX 3090，属于实测 wall-clock 成本而非硬件归一化吞吐比较；Replay 列只统计 observation payload，不含 Python 索引和文件系统稀疏分配差异。",
        "",
        "## 4.4 指标与统计",
        "",
        "对任务 \(\\tau\)，归一化性能为",
        "",
        "$$q_\\tau(n)=\\frac{p_\\tau(n)-p_{\\mathrm{ST}_\\tau}(0)}{p_{\\mathrm{ST}_\\tau}(n)-p_{\\mathrm{ST}_\\tau}(0)}.$$",
        "",
        "其中 \(q=0\) 对应随机策略，\(q=1\) 对应匹配预算的单任务 ARROW；数值不截断。我们报告最终平均性能 ACC、平均最小性能 min-ACC、最坏情况综合性能 WC-ACC，以及从任务获取边界到最终边界的 forgetting（F）。ACC、min-ACC 和 WC-ACC 越高越好，F 越低越好；负 F 表示 backward improvement。低 F 必须与 ACC 和 raw return 联合解释，因为未能获取任务也可能产生低遗忘。",
        "",
        "正式主结果使用预声明 seed 的 seed-level median `[Q25, Q75]`。偶数样本的 median 是中间两值的平均；Q25–Q75 覆盖中间一半结果，区间越宽表示跨 seed 波动越大。Atari 基线已有五个完整 seed；当前 AWM Atari 表使用五个完整 pilot（S0、S7、S11、S12、S30），并非冻结的正式 cohort。CoinRun DreamerV3/FIFO 排除有问题的 S2 记录后为四个 seed；AWM 使用原 post-hoc top-five 子集中保留的三个 seed（排除 2026091703、2026091801），不是预声明 cohort。FT 需要时间对齐的单任务学习曲线；当前仍不报告。",
        "",
        "### 4.4.1 结果判读约定",
        "",
        "| 指标 | 回答的问题 | 正确解释 |",
        "|---|---|---|",
        "| `q` | 单个任务学到了什么水平？ | `q=0` 是随机参考，`q=1` 是单任务 ARROW 参考；`q>1` 合法，表示超过该参考，而不是“准确率超过 100%” |",
        "| ACC | 六任务结束时总体表现如何？ | 高 ACC 表示最终平均水平高，但可能掩盖个别任务失败 |",
        "| F | 任务学完后到最终时下降多少？ | 越低越好；负值表示后续训练使该任务变好，但低 F 不能单独证明性能好 |",
        "| min-ACC | 后续训练期间旧任务最差掉到哪里？ | 越高越稳定；它能发现“最后恢复了、途中却崩过”的情况 |",
        "| WC-ACC | 当前任务学习与旧任务最差保持是否兼顾？ | 越高越好，用于观察 stability–plasticity balance |",
        "| Raw return | 提升是否存在于环境原始回报中？ | 同一任务内可直接比较；不同 Atari 游戏量纲不同，不能横向平均 |",
        "",
        "学习曲线中的粗线段表示该任务正在训练，细线段表示训练前或训练后的评估；竖直虚线表示任务边界。一个旧任务在粗线段结束后持续下滑，表示遗忘；维持水平表示保持；继续上升则表示 backward improvement。五 seed 曲线显示中位数和 IQR，单 seed 曲线不具有跨 seed 不确定性信息。",
        "",
        "最简判读顺序是：先用 raw return 和 ACC 判断任务是否真正学会，再用 F 与 min-ACC 判断学会后是否遗忘，最后用 WC-ACC 检查保持旧任务是否以牺牲新任务学习为代价。只有当这些指标方向一致、逐任务结果不由单个游戏驱动、并且跨 seed 分布稳定时，才能形成较强结论。",
        "",
        "# 5 结果（Results）",
        "",
        "![Atari 与 CoinRun 主指标汇总](figures/main_metric_summary_with_pending.png)",
        "",
        "**图 1：Atari 与 CoinRun 的持续学习指标总览。** 柱高为 seed-level median，误差线为 IQR。CoinRun DreamerV3/FIFO 的 S2 记录已排除（n=4）；AWM 使用原 post-hoc top-five 中保留的三个 pilot seed（排除 2026091703、2026091801），不能作为无偏或预声明 cohort 估计。",
        "",
        "## 5.1 无共享结构任务：Atari",
        "",
        "![Atari 固定顺序单轮学习曲线](figures/atari_main_learning_curves.png)",
        "",
        "**图 2：Atari 固定顺序、单轮六任务学习曲线。** 每种颜色对应一个游戏，粗线段表示当前训练任务，阴影表示跨 seed IQR。主图采用固定纵轴；未裁剪结果见[完整范围图](figures/atari_main_learning_curves_full_range.png)。AWM 曲线来自当前五个完整 pilot。",
        "",
        "每个任务占 1,474,560 次 agent decisions。阅读图 2 时，应先看粗线段能否快速上升，以判断当前任务是否被获取；再沿同一颜色观察任务边界后的细线段，以判断旧任务是否被保持。例如，DreamerV3/FIFO 的多条旧任务曲线在切换后回到零附近，而 ARROW-50 的旧任务曲线整体维持在更高水平。AWM 多条曲线超过固定纵轴上限，因此主图顶部的水平段不代表真实饱和，必须结合完整范围图和表 4。",
        "",
        "**表 3：Atari 持续学习指标，median `[Q25, Q75]`。**",
        "",
        "| 方法 | n | 证据等级 | F ↓ | ACC ↑ | min-ACC ↑ | WC-ACC ↑ |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    statuses = {
        "DreamerV3/FIFO": "5-seed baseline",
        "ARROW-50": "5-seed baseline",
        "AWM-AutoRoute": "5-seed pilot；非正式 cohort",
    }
    for method in ("DreamerV3/FIFO", "ARROW-50", "AWM-AutoRoute"):
        row = atari[method]
        metrics = row["metrics"]
        lines.append(
            f'| {method} | {row["n"]} | {statuses[method]} | '
            f'{_metric(metrics["forgetting"])} | {_metric(metrics["acc"])} | '
            f'{_metric(metrics["min_acc"])} | {_metric(metrics["wc_acc"])} |'
        )
    lines += [
        "",
        "**Baseline sanity check.** ARROW-50 相对 DreamerV3/FIFO 将 F 中位数从 1.9894 降至 0.3326，并将 ACC 从 0.1099 提高到 0.6187；min-ACC 和 WC-ACC 也由负值提升至 0.6226 和 0.5429。学习曲线显示，FIFO 在任务切换后旧任务显著回落，而 ARROW-50 保持了更高的历史任务性能。这一趋势与 ARROW 的已发表观察一致，但局部峰值和最终幅度并非精确数值复现。",
        "",
        "**Retention and plasticity.** 四项指标给出了相互补充的证据。ACC 的提升说明 ARROW-50 在六任务结束时总体更强；F 的下降说明这种提升不只是更快学习新任务，也包含更少遗忘；min-ACC 从负值转为正值，说明旧任务在整个训练过程中不再频繁跌到随机参考以下；WC-ACC 的提升则表明保持旧任务并未完全牺牲当前任务学习。四项指标方向一致，比只比较最终平均值更有说服力。",
        "",
        "**AWM pilot signal.** AWM-AutoRoute pilot 的 ACC、min-ACC 和 WC-ACC 中位数分别为 1.9321、1.5927 和 1.3775，F 为 0.0078。描述性地看，它们分别比本地 ARROW-50 高 1.3134、0.9701 和 0.8346，F 低 0.3248。然而，AWM 的 ACC IQR `[1.5687, 2.9629]` 明显宽于 ARROW-50 的 `[0.6058, 0.6259]`，表明 seed 敏感性或评估协议差异较大。由于 AWM 具有额外更新、不同评估 cohort 和非正式 seed cohort，表 3 只能作为 pilot evidence，不能构成严格公平预算下的 superiority claim。",
        "",
        "**Across-seed variability.** AWM 的四项 aggregate 指标方向同样一致：最终水平更高、获取后下降更小、训练途中的最低点更高。然而，五个 pilot 的 ACC 从 1.3236 到 3.2175，最高值约为最低值的 2.43 倍。这意味着当前结果同时包含“较高性能潜力”和“较大 seed/评估敏感性”两个信息；不能只引用中位数而忽略离散程度。",
        "",
        "**Distribution separation.** 从已记录分布看，AWM 的 ACC、min-ACC 和 WC-ACC 的 Q25 均高于 ARROW-50 的 Q75，AWM 的 F 上四分位数 0.0907 也低于 ARROW-50 的下四分位数 0.2774。也就是说，两组中间 50% seed 在四项指标上均未重叠；当前差异并非只由一个最佳 seed 造成。这加强了“信号值得认真验证”的判断，但不能消除表 2 的协议混杂：分布分离说明观测差异大，不等于已经证明差异完全由算法本身造成。",
        "",
        "**表 4：Atari 最终逐任务 raw return，median `[Q25, Q75]`。** 不同游戏的 raw return 不横向平均。",
        "",
        "| 方法 | MsPacman | Boxing | CrazyClimber | Frostbite | Seaquest | Enduro |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("DreamerV3/FIFO", "ARROW-50", "AWM-AutoRoute"):
        values = _final_raw_task_summary(atari_reports[method])
        lines.append(
            f'| {method} | ' + " | ".join(_interval(value, 1) for value in values) + " |"
        )
    lines += [
        "",
        "**Task-level consistency.** AWM 的最终 raw-return 中位数在六个游戏上均高于本地 ARROW-50，说明其高 aggregate 并非由单一游戏独占。ARROW-50 则在前五个游戏上高于 FIFO，但 Enduro 低于 FIFO，表明其优势并非逐游戏支配。",
        "",
        "**Where the gains arise.** 逐任务差异并不均匀。AWM 相对 ARROW-50 在 Boxing 上仅由 82.7 提高到 87.8，而在 Frostbite 上由 260.6 提高到 1,871.9、Seaquest 上由 263.8 提高到 865.0。最大的 aggregate 优势主要来自 Frostbite、Seaquest、MsPacman 和 CrazyClimber，而不是所有任务等比例改善。Frostbite 约 7.2 倍的差距尤其需要在同 evaluator、同 cohort 重评后确认。",
        "",
        "进一步看四分位区间，AWM 与 ARROW-50 在 MsPacman、CrazyClimber、Frostbite、Seaquest 和 Enduro 上的中间 50% seed 完全分离，只有 Boxing 明显重叠。这个模式比“六项中位数都更高”更具体：AWM 的主要不确定性不是优势是否只出现在一个任务，而是这些大幅优势在统一评估协议下会保留多少。",
        "",
        "## 5.2 共享结构任务：Procgen CoinRun",
        "",
        "![CoinRun 固定难度顺序单轮学习曲线](figures/coinrun_main_learning_curves.png)",
        "",
        "**图 3a：CoinRun 固定难度顺序、单轮六变体学习曲线。** DreamerV3/FIFO 的 S2 记录已排除（n=4）；ARROW-50 为五个 seed；AWM 为原 post-hoc top-five 中保留的三个 seed（排除 2026091703、2026091801）。曲线显示各组 median 和 IQR。完整纵轴范围见[完整范围图](figures/coinrun_main_learning_curves_full_range.png)。",
        "",
        "![AWM 在 CoinRun 上随任务累积的参数组成](figures/coinrun_awm_parameter_growth.png)",
        "",
        "**图 3b：AWM 在 CoinRun 上随任务增加的 world-model 参数组成。** 统计口径与图 4b 相同，使用当前归档的三个 post-hoc seed。已完成任务的逻辑保留容量依次为 20.02M、21.97M、23.92M、24.92M、26.87M 和 27.39M，明显低于 Atari 最终的 37.85M；主要原因是 CoinRun 各阶段被自适应压缩为更窄的 Q/F/P 机制。灰色 checkpoint 分配线从 39.27M 下降并在最后一个边界与 27.39M 汇合，因为未来任务槽位同样是在启动时预分配。",
        "",
        "**表 5：CoinRun 最终表现与持续学习指标，median `[Q25, Q75]`。** DreamerV3/FIFO 的 S2 因记录问题从本稿计算中排除（n=4），原始记录仍保留；AWM 的 seed 2026091703、2026091801 也从本稿计算中排除，表内为原 post-hoc top-five 的剩余三条记录。AWM 归一化指标使用与 baseline 相同的 Table A.16 固定锚点，但 evaluator/cohort 仍不匹配。",
        "",
        "| 方法 | n | Final raw average | F ↓ | ACC ↑ | min-ACC ↑ | WC-ACC ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("DreamerV3/FIFO", "ARROW-50"):
        row = coinrun[method]
        lines.append(
            f'| {method} | {row["n"]} | {_metric(row["raw"])}; '
            f'mean±SD {row["raw_mean"]:.4f}±{row["raw_sample_sd"]:.4f} | '
            f'{_metric(row["metrics"]["forgetting"])} | '
            f'{_metric(row["metrics"]["acc"])} | '
            f'{_metric(row["metrics"]["min_acc"])} | '
            f'{_metric(row["metrics"]["wc_acc"])} |'
        )
    lines.append(
        f'| AWM-AutoRoute (post-hoc pilot) | {coinrun_awm["n"]} | '
        f'{_metric(coinrun_awm["raw"])}; '
        f'mean±SD {coinrun_awm["raw_mean"]:.4f}±{coinrun_awm["raw_sample_sd"]:.4f} | '
        f'{_metric(coinrun_awm["metrics"]["forgetting"])} | '
        f'{_metric(coinrun_awm["metrics"]["acc"])} | '
        f'{_metric(coinrun_awm["metrics"]["min_acc"])} | '
        f'{_metric(coinrun_awm["metrics"]["wc_acc"])} |'
    )
    lines += [
        "",
        "**Overall result.** ARROW-50 的 final raw average 中位数为 6.9727，高于排除 S2 后 DreamerV3/FIFO 的 5.5452；其固定锚点 ACC 高 0.2870、F 低 0.4110，min-ACC 和 WC-ACC 分别高 1.2099 和 0.9598。逐任务结果显示，该优势集中在中间四个累积变体，而基础 CoinRun 和最终 +CA 低于 FIFO。",
        "",
        "**Retention rather than a single final score.** 这里 min-ACC 与 final raw average 传达了不同信息。FIFO 的 final raw average 中位数为 5.5452，但 min-ACC 为 -0.3287，说明部分旧变体在训练过程中曾明显跌到随机参考以下；ARROW-50 的 min-ACC 为 0.8812，表明其历史任务低谷仍接近单任务参考。ARROW-50 的 F 略为负值（-0.0076），表示最终性能相对获取边界有轻微 backward improvement，而不是“完全没有变化”。",
        "",
        "CoinRun 上的证据强度也应分指标判断。ARROW-50 与排除 S2 后 FIFO 的 final raw average IQR 不重叠，F、min-ACC 和 WC-ACC 的 IQR 也不重叠；但 ACC 区间存在小幅重叠。因此，现有四个 FIFO 与五个 ARROW seed 对“ARROW 改善保持和训练中稳定性”的支持强于对“所有意义上的最终性能都稳定领先”的支持。",
        "",
        "**AWM CoinRun pilot.** 当前三条保留记录的 final raw average 中位数为 6.9401，略低于 ARROW-50 的 6.9727、高于排除 S2 后 FIFO 的 5.5452。它的 ACC、min-ACC 和 WC-ACC 中位数分别为 1.1804、1.0017 和 1.0268，F 为 -0.0101；描述性地均优于本地 ARROW-50。两组证据并不矛盾：raw average 强调最终原始得分，而归一化持续学习指标还利用获取边界和训练中低谷。该三-seed集合只是此前 post-hoc top-five 的剩余部分。",
        "",
        "AWM 相对 ARROW-50 的逐变体中位数为四胜、一平、一负：CoinRun、+RT、+MA 和 +CA 更高，+NB 相同，+GA 更低。当前三条记录仍来自观察结果后选出的 top-five 子集，不能将其 IQR 解释为预声明 cohort 的稳定性。加之 AWM 与 baseline 的 evaluator/cohort 不同，表 5 支持的是描述性 pilot signal，而不是正式 superiority。",
        "",
        "**表 6：CoinRun 最终逐变体 raw return。**",
        "",
        "| 方法 | CoinRun | +NB | +RT | +GA | +MA | +CA |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("DreamerV3/FIFO", "ARROW-50"):
        lines.append(
            f'| {method} | '
            + " | ".join(_interval(value, 4) for value in coinrun[method]["raw_tasks"])
            + " |"
        )
    lines += [
        "| AWM-AutoRoute (post-hoc pilot) | "
        + " | ".join(_interval(value, 4) for value in coinrun_awm["raw_tasks"])
        + " |",
        "",
        "**The last-task exception.** 逐变体表揭示了可能的 stability–plasticity trade-off：ARROW-50 和 AWM 在多个旧变体上优于 FIFO，但在当前最后任务 +CA 上的中位数分别为 4.6875 和 5.1172，低于排除 S2 后 FIFO 的 5.9180。由于 +CA 改变相机是否围绕 agent 居中，它也比前几个主要视觉扰动更像一次动力学/观测机制转移。因此，低 +CA 既可能是保持旧任务带来的 plasticity cost，也可能是该特定任务更难获取；需要同 seed、同 evaluator 以及任务内学习曲线区分。",
        "",
        "表 6 是六任务全部训练结束后对六个不同变体的横向评估，不能把各列直接当成同一任务随时间下降。AWM 从 +RT 到 +GA 的中位数由 7.8125 降至 7.4609，随后 +MA 回升至 8.2422，呈现中间波动而非持续下降。最后任务 +CA 的中位数为 5.1172，仍低于 FIFO。对当前保留的三个 seed，T6 fixed-validation checkpoint 的 +CA 中位数为 6.25，高于独立 held-out final evaluation 的 5.1172；这提示跨环境 seed 泛化或 evaluator 方差可能参与其中，不能仅归因于训练没有学会 +CA。",
        "",
        "相似现象也出现在 Atari：ARROW-50 在前五个游戏上高于 FIFO，却在最后训练的 Enduro 上为 86.8，低于 FIFO 的 163.7。两个 benchmark 都出现“旧任务保持更好、最后任务不一定更好”的模式，因此 ARROW-50 的主要收益更像 retention，而不是无条件提升每个任务的获取能力。AWM 在 Atari-Enduro 上没有出现同样下降，但在 CoinRun-+CA 上仍然较低；这提示 AWM 是否真正改善 stability–plasticity balance 仍是需要正式实验回答的问题。",
        "",
        "## 5.3 容量组织对照",
        "",
        "为检验性能是否仅由参数规模、private actor 或完整任务隔离造成，我们在 Atari 上比较五种 task-aware 组织方式。各组匹配交互、Replay 容量、在线更新和评估预算，但不匹配参数量或 FLOPs，并统一使用 oracle task route。因此该实验是结构诊断，而不是与 task-ID-free AWM-AutoRoute 的主排名。",
        "",
        "**表 7：Atari 容量组织对照，median `[Q25, Q75]`。**",
        "",
        "| 对照 | 完成 seed | Retained WM 参数 | 含 6 个 private AC | F ↓ | ACC ↑ | min-ACC ↑ | WC-ACC ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = (
        "Shared WM + private AC",
        "Wider shared + private AC",
        "FullBank + private AC",
        "Frozen core + residuals",
        "Independent residuals",
    )
    for method in order:
        row = capacity[method]
        metrics = row["metrics"]
        wm = CAPACITY_WM_PARAMS[method]
        lines.append(
            f'| {method} | {row["n"]}/3 ({", ".join(f"S{s}" for s in row["seeds"])}) | '
            f'{wm:,} | {wm + PRIVATE_AC_PARAMS:,} | '
            f'{_metric(metrics["forgetting"])} | {_metric(metrics["acc"])} | '
            f'{_metric(metrics["min_acc"])} | {_metric(metrics["wc_acc"])} |'
        )
    lines += [
        "",
        "![Atari 容量与性能 Pareto 诊断](figures/capacity_performance_pareto.png)",
        "",
        "**图 4a：Atari 容量—性能诊断。** 横轴只比较最终保留的完整 world-model 参数量，纵轴为 ACC；纵向误差线为 ACC IQR，AWM 还显示跨 seed world-model 参数量 IQR。六个 private actor–critic 的额外成本在表 7 和表 2a 单列，不混入该横轴。该图使用已有真实参数清单和完整指标，不是占位数据；Replay observation payload、峰值 CUDA 显存和 wall-clock GPU-hours 已在表 2a 补齐。",
        "",
        "![AWM 随任务累积的参数组成](figures/awm_parameter_growth.png)",
        "",
        "**图 4b：AWM 随任务增加的 world-model 参数组成。** 堆叠柱只统计已完成任务真正保留的组件：共享基础模型、累计 representation Q、recurrent F、transition P 机制和 observation projectors；柱顶数字与黑色误差线分别为五 seed median 和 IQR。逻辑保留容量从完成 T1 后的 23.35M 增至 T6 后的 37.85M，六个边界依次为 23.35M、26.25M、30.10M、32.05M、34.00M 和 37.85M。增长主要来自橙色 Q 机制。灰线是实际 checkpoint 分配量：当前实现从启动时就预分配六个任务槽位，因此它会随边界压缩从 42.60M 下降并最终与 37.85M 汇合；这条线不能误读成方法在学习过程中删除共享知识。",
        "",
        "参数规模与性能没有呈现单调关系。参数最少的 Shared WM + private AC（19.50M WM 参数）取得最高 ACC 中位数 1.6204，Independent residuals 以 42.60M 参数取得 1.5870，而 116.99M 参数的 FullBank 为 1.1866。Frozen core + residuals 的 ACC 仅 0.1891，表明完全冻结共享表征可能显著限制后续任务可塑性；但该组只有两个完整 seed。该结果部分排除了“参数越多自然越好”的解释，尚不能排除 task-aware routing、private AC 或 FLOP 差异。",
        "",
        "Frozen core 是理解指标组合的重要反例：其 F 为 -0.0161，看似几乎没有遗忘甚至略有提升，但 ACC 只有 0.1891，min-ACC 只有 0.2074。这说明低 forgetting 可能来自任务获取不足，而不是成功保持。相反，Shared WM 和 Independent residuals 同时取得较高 ACC 与较高 min-ACC，更符合“既学会又保持”的目标。",
        "",
        "FullBank 的 retained WM 参数约为 Shared WM 的 6 倍，但 ACC 更低；Wider shared 也未稳定超过标准 shared。因而现有结果不支持“完整隔离”或“增加宽度”本身足以解释高性能。不过这些控制都使用 oracle task route 与 private AC，不能直接证明 AWM 的自动路由或保护机制有效。",
        "",
        "还不能据此给五种结构做严格名次排序：Shared、Independent residuals 与 FullBank 的 ACC IQR 大幅重叠，Frozen 又少一个 seed。表 7 最可靠的结论是“参数量与表现不单调”和“冻结共享核心存在明显获取风险”，而不是“Shared 一定优于 Independent”或“FullBank 一定无效”。",
        "",
        "**A strong alternative explanation.** 表 7 也暴露了一个不能回避的替代解释：使用 oracle route 和 private AC 的 Shared control 已达到 ACC 1.6204、min-ACC 1.7380 和 WC-ACC 1.4745。与之相比，AWM pilot 的 ACC 更高（1.9321），但 F、min-ACC 和 WC-ACC 并未全面更好。由于两组 seed、route 与评估协议并不匹配，不能直接排名；但这些数字说明，task-aware 组织和 private AC 本身可能解释相当一部分增益。AWM 最有价值的潜在主张应当是“在推理时无需 oracle task ID，仍接近或超过 task-aware 上界”，而这一主张必须由同 checkpoint 的 auto-route/oracle gap 和匹配 cohort 实验支持。",
        "",
        "## 5.4 机制消融",
        "",
        "**表 8：AWM-AutoRoute 单变量消融完成状态。**",
        "",
        "| 消融 | 要回答的问题 | 准入完整结果 | Staged metric artifacts | Failed attempts | 可填正式结果 |",
        "|---|---|---:|---:|---:|---|",
    ]
    ablation_questions = {
        "NoReuse": "历史机制复用是否带来正迁移或减少重复学习？",
        "NoFunctionalProtection": "旧任务功能保持目标是否直接降低遗忘？",
        "NoConflictProjection": "冲突梯度投影是否缓解新旧任务更新干扰？",
        "NoRCC": "return-gated structural compaction 是否改善容量—性能权衡？",
    }
    for label, completed, staged, failed in _ablation_counts():
        lines.append(
            f'| {label} | {ablation_questions[label]} | {completed}/3 | '
            f'{staged}/3 | {failed}/3 | '
            f'{"否" if completed < 3 else "是"} |'
        )
    lines += [
        "",
        "`Staged metric artifacts` 表示本地已经看到完整形状的指标文件，但其目录仍标记为 partial，尚未完成全量文件、最终评估和完整性审计；它们不计入准入完整结果，也不进入数值汇总。四项消融当前均不能正式归因 reuse、functional protection、conflict projection 或 RCC 的贡献。",
        "",
        "![AWM 单变量消融占位图](figures/ablation_results_placeholder.png)",
        "",
        "**图 5：AWM-AutoRoute 单变量消融的最终版式占位。** 橙色柱仅为当前完整 AWM pilot；其余槽位显示本地 staged/failed 状态，不绘制任何 partial 数值。每项达到三条准入完整 seed 后，生成器将用 median `[Q25,Q75]` 替换 `PENDING`。",
        "",
        "消融的判读也不能只看“删掉后 ACC 是否下降”。若 NoFunctionalProtection 主要使 F 和 min-ACC 变差而当前任务 raw return 不变，才支持其作用是保持；若 NoReuse 主要降低新任务初期学习速度，则更接近 transfer 机制；若关闭某模块同时减少更新量或参数量，还需要 compute-matched control，否则性能下降可能只是预算下降。",
        "",
        "## 5.5 路由与机制诊断",
        "",
        "**表 9：冻结 checkpoint 上的补充诊断。**",
        "",
        "Table 9 使用预先固定的四-seed artifact-available cohort：S0、S7、S11、S12（n=4）。S30 仍保留在五-seed主结果中，但因最终 checkpoint 已缺失而不进入需要重新前向计算的诊断；排除规则与性能无关。完整定义见 `docs/protocols/awm_table9_frozen_diagnostics_v1.md`。",
        "",
        "| 诊断 | 主要测量 | 如何解释 | 当前状态 |",
        "|---|---|---|---|",
        f'| Auto-route vs oracle-route | 同 checkpoint、同 cohort 的逐任务 return gap | final held-out trace 中首帧和第二帧 route accuracy 均为 {routing["observation_accuracy"][1]:.1%}；但没有单独保存 paired oracle returns | **Running（4 seeds，eval-only）** |',
        f'| Routing confusion | route accuracy、task×route confusion、reset 后切换延迟 | {routing["correct"]}/{routing["total"]} 个 held-out episode starts 正确；aggregate confusion 对角线为 ({confusion_diagonal})，off-diagonal 全为 0；首个 actionable observation 已达 {routing["observation_accuracy"][1]:.1%} | **Complete（4 seeds）** |',
        "| Historical reuse intervention | 同一冻结 checkpoint 禁用历史 atom 路由前后的 return 差异 | 现有 NoReuse 是另行训练的 ablation，不能替代 same-checkpoint causal intervention | **Running（4 seeds，eval-only）** |",
        f'| Retained capacity | 每任务 Q/F/P 宽度、完整 world model 与总在线参数 | world model 为 {table9_world_model["median"] / 1e6:.2f}M [{table9_world_model["q25"] / 1e6:.2f}, {table9_world_model["q75"] / 1e6:.2f}]；含六个 private AC 的总在线参数为 {table9_parameters["median"] / 1e6:.2f}M [{table9_parameters["q25"] / 1e6:.2f}, {table9_parameters["q75"] / 1e6:.2f}] | **Complete（4 seeds）** |',
        "| Predictive retention | H=1,2,4,8,16 open-loop 图像/奖励误差 | 冻结 checkpoint 上的独立 held-out rollout；不需要重新训练 | **Running（4 seeds，frozen audit）** |",
        "",
        "**表 9a：AWM 每个 route 的压缩后 Q/F/P 宽度，median `[Q25,Q75]`。**",
        "",
        "| Route | Posterior Q width | Recurrent F width | Prior P width |",
        "|---|---:|---:|---:|",
        *width_lines,
        "",
        "![路由与机制诊断占位图](figures/diagnostic_panels_placeholder.png)",
        "",
        "**图 6：路由与机制诊断的预注册版式。** 表 9 已从现有 artifacts 填入 routing confusion、首帧识别和 retained capacity；图中对应 panel 仍保留版式占位，待与剩余三项 eval-only 诊断一起统一重绘。占位不含假数据。",
        "",
        "所有诊断均使用冻结 checkpoint 和独立评估数据，不进入 replay，也不更新参数。Oracle route 仅作为同一 AWM-AutoRoute 模型的诊断上界，不定义为第二个方法。",
        "",
        f'现有 held-out trace 的 routing 结果很强：四个 Table 9 seed 合计 {routing["total"]} 个 episode starts 均在第一个 actionable observation 选择正确 route，第二帧也没有改错。这排除了当前 cohort 上的显式 route confusion，但不能用分类正确率替代 paired return gap；v4 协议本身也规定，多 route 情况下必须验证实际 action/return path。因而 auto/oracle 只剩同 checkpoint、同 seed 的 oracle 重评，不需要重新训练。',
        "",
        "另外两项未填诊断也不是新训练：reuse intervention 是冻结权重后关闭历史 route contribution 的配对评估；predictive retention 在冻结 checkpoint 上使用独立 held-out 轨迹做 open-loop forward，不读取或改写训练 Replay。Table 9 已固定为 S0/S7/S11/S12（n=4），S30 因最终 checkpoint 缺失而排除；该排除依据是 artifact 可用性，不是性能。四个诊断已从干净且已推送的提交启动，完成前仍保持 pending。",
        "",
        "# 6 讨论（Discussion）",
        "",
        "## 6.1 主要发现与证据层级",
        "",
        "现有结果首先验证了本地评估能够测出 replay retention 的作用：ARROW-50 在 Atari 和 CoinRun 上均表现出比 FIFO 更好的历史任务保持。其次，AWM-AutoRoute 在 Atari pilot 中产生了强烈正向信号，其 aggregate 指标和六个游戏的最终 raw-return 中位数均高于本地 ARROW-50。CoinRun 当前保留的三条 post-hoc pilot 记录呈现混合模式：归一化持续学习指标较强，但 final raw average 中位数略低于 ARROW-50，逐变体中位数四胜、一平、一负。容量组织实验进一步表明性能不随参数量单调增长，但尚不能排除 private AC、task-aware training 或额外计算的贡献。",
        "",
        "按证据强度，当前结果可分为三层。第一层是较稳固的本地事实：在相同 baseline 协议下，ARROW-50 比 FIFO 更能保持旧任务。第二层是强但暂定的信号：AWM 在 Atari 的五个完整 pilot 上与 ARROW-50 有大幅、跨任务且分布分离的差异。第三层仍是未知：AWM 是否跨 benchmark 稳定领先、自动路由是否接近 oracle、以及每个机制分别贡献多少。论文表述必须保持这三层边界。",
        "",
        "**表 10：研究问题与当前证据。**",
        "",
        "| 研究问题 | 当前观察 | 当前能够支持的结论 |",
        "|---|---|---|",
        "| RQ1：Atari 获取与保持 | AWM pilot 的四项 aggregate 指标及六任务 raw-return 中位数均高于本地 ARROW-50 | 强正向 pilot signal；尚非公平预算下的正式 superiority |",
        "| RQ2：跨 benchmark 泛化 | CoinRun AWM 当前保留的三条 post-hoc 记录归一化指标较强，raw average 略低于 ARROW，逐变体四胜、一平、一负 | 记录排除、事后选择和协议差异阻止正式跨 benchmark 优势结论 |",
        "| RQ3：是否只是容量更多 | 19.50M Shared 的 ACC 高于 116.99M FullBank，性能不随参数单调上升 | 部分排除纯参数量解释；尚未排除 task-aware/private AC/FLOPs |",
        "| RQ4：各机制贡献 | 四项消融均无完整结果 | 不能归因任何单一机制 |",
        f'| RQ5：自动路由有效性 | final held-out routing 为 {routing["correct"]}/{routing["total"]}，首帧与第二帧均 100%；paired oracle return 尚未评估 | 可声称当前 cohort 上未观察到 route confusion；尚不能声称 return 与 oracle 等价 |',
        "",
        "## 6.2 稳定性—可塑性与机制解释",
        "",
        "两个 benchmark 都提示不能把 retention 与 acquisition 混成一个“性能”概念。ARROW-50 相对 FIFO 的主要优势是任务切换后的保持，而最后任务 Enduro 和 +CA 并未同步改善。AWM 在 Atari 上同时取得高 ACC 与低 F，表面上更接近理想的稳定性—可塑性平衡；但 CoinRun-+CA 和强 task-aware controls 又表明这种平衡尚未跨设置确认。",
        "",
        "AWM 的结果模式与其设计动机一致：route-specific capacity 可减少参数干扰，functional protection 和 conflict projection 可能保护旧任务，历史复用可能加速新任务获取。然而，“与设计一致”不是机制证据。缺少 NoReuse、NoFunctionalProtection、NoConflictProjection、NoRCC 以及 auto/oracle 诊断时，任何把性能提升归因于某个组件的说法都仍是 hypothesis。",
        "",
        "## 6.3 局限与可支持主张",
        "",
        "从性能角度看，当前最积极的证据来自 Atari：AWM 不仅最终平均性能更高，而且历史任务低谷和最终保持也同时改善。CoinRun 当前只纳入原 post-hoc top-five 中的三个 seed，并排除 seed 2026091703 和 2026091801；该子集不能替代预声明 cohort。两个 benchmark 还都存在额外更新、不同 evaluator/cohort 与非正式 seed cohort。换言之，现有结果更适合支持“方法值得继续验证”，而不是支持“方法已经被证明全面优于 ARROW”。",
        "",
        "如果只回答“方法目前看起来好不好”，答案是：**Atari 上看起来很强，CoinRun 上有竞争力但并未全面领先，整体还不能下正式胜出结论。** Atari 的好信号不是单一数字：最终 ACC、遗忘、训练中最低点、最坏情况综合指标和六项 raw return 同时改善；风险则来自 seed 波动较大和评估协议不完全匹配。CoinRun 的归一化保持指标较强，但三-seed保留子集的 final raw average 略低于 ARROW，且该子集来自事后选择并排除了两条记录，因此只支持继续验证而非正式胜出。",
        "",
        "当前结论受到四项主要限制。第一，AWM Atari 使用完整 pilot seed，而非冻结正式 cohort。第二，AWM 与基线的 evaluator 和 cohort 不一致。第三，AWM 包含额外 12,000 次 WM 更新和 480 个压缩 validation rollouts。第四，CoinRun 当前三-seed子集来自事后筛选，且又排除了两条记录，并非预声明 cohort。因而本章当前支持的表述是：**AWM-AutoRoute 在 Atari 上显示出值得进一步确认的获取与保持优势，并在 CoinRun 上呈现有竞争力但混合的 pilot signal；正式 superiority、跨 benchmark 泛化和机制归因仍待匹配评估与预声明实验验证。**",
        "",
        "# 7 复现与结果来源（Reproducibility）",
        "",
        "- Atari baseline S0：`docs/protocols/references/local_s0_continual_metric_comparison_v1.json`。",
        "- Atari baseline S1–S4：`runs/cloud_result_backups/20260915/completed/` 下 8 个 canonical tar。",
        "- Atari AWM：`runs/main_results/atari/awm_autoroute/{s0,s7,s11,s12,s30}/continual_metrics.json`。",
        "- Table 9 frozen diagnostics：`docs/protocols/awm_table9_frozen_diagnostics_v1.md`；固定 cohort 为 `{s0,s7,s11,s12}`（n=4）。",
        "- 资源表：同一批 run 的 `model_parameter_accounting.json`、`actor_critic_parameter_accounting.json`、`resolved_training_config.json`、`replay_mmap_storage_accounting.json`、`train.log`、`launch.json` 与 `run_status.json`。",
        "- 路由/容量诊断：五个 AWM run 的 `final_evaluation.json` 与 `model_parameter_accounting.json`。",
        "- CoinRun baseline：`docs/experiments/records/coinrun-*/record.json`。",
        "- CoinRun AWM post-hoc pilot（本稿保留）：`runs/main_results/coinrun/awm_autoroute/{seed2026091701,seed2026091704,seed2026091705}/`；排除 seed2026091703 和 seed2026091801，原 top-five 选择与 rsync 校验记录见同目录 `selected5_archive_manifest.json`。",
        "- 容量对照：`runs/cloud_result_backups/*/completed/*/capacity_*.tar`。",
        "",
        "```bash",
        "python3 scripts/build_paper_curve_figures.py",
        "python3 scripts/build_paper_curve_figures.py --check",
        "python3 scripts/build_paper_results_draft.py --write",
        "python3 scripts/build_paper_results_draft.py --check",
        "```",
        "",
        "# 附录 A：逐 seed 与内部一致性检查",
        "",
        "## A.1 AWM-AutoRoute Atari pilot seeds",
        "",
        "**表 A.1：当前五个完整 Atari pilot 的 seed-level 指标。**",
        "",
        "| Seed ID | 实际 seed | F ↓ | ACC ↑ | min-ACC ↑ | WC-ACC ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for report in awm_seeds:
        metrics = report["metrics"]
        lines.append(
            f'| {report["display_seed_id"]} | {report["seed"]} | '
            f'{metrics["forgetting"]:.4f} | {metrics["acc"]:.4f} | '
            f'{metrics["min_acc"]:.4f} | {metrics["wc_acc"]:.4f} |'
        )
    lines += [
        "",
        "S7 与 S12 的 F 为负，表示最终性能高于其任务获取边界；S30 的 ACC 最低。该跨度说明正文必须保留完整 seed 分布，不能只展示最佳运行。",
        "",
        "## A.2 AWM-AutoRoute CoinRun post-hoc pilot seeds",
        "",
        "**表 A.2：当前纳入本稿的三个 CoinRun pilot 的 seed-level 指标。**",
        "",
        "| 实际 seed | Final raw average | F ↓ | ACC ↑ | min-ACC ↑ | WC-ACC ↑ |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for report in coinrun_awm_rows:
        metrics = report["metrics"]
        lines.append(
            f'| {report["seed"]} | {report["raw_final_average"]:.4f} | '
            f'{metrics["forgetting"]:.4f} | {metrics["acc"]:.4f} | '
            f'{metrics["min_acc"]:.4f} | {metrics["wc_acc"]:.4f} |'
        )
    lines += [
        "",
        "这三个 seed 是原先观察八个完成运行后选出的 top-five 子集在排除 2026091703 和 2026091801 后的剩余记录；该子集不能用于无偏估计总体 seed 分布。",
        "",
        "## A.3 与 ARROW 原图的内部视觉核对（不作为投稿主图）",
        "",
        "![ARROW Figure 3A 与本地 Atari 结果对照](figures/atari_arrow_paper_vs_local.png)",
        "",
        "![ARROW Figure 5A 与本地 CoinRun 结果对照](figures/coinrun_arrow_paper_vs_local.png)",
        "",
        "以上拼图仅用于内部检查版式、曲线趋势与纵轴范围；正式稿主图使用图 1 和图 2，不直接复用他人论文图形。ARROW 原图来源为 `https://arxiv.org/html/2603.11395v3`。",
        "",
        "## A.4 投稿前必须替换的证据",
        "",
        "- 用冻结正式 Atari cohort 替换当前 AWM pilot cohort。",
        "- 用同一 evaluator、policy mode 和环境 seeds 重评 AWM、ARROW-50 与 FIFO checkpoint。",
        "- 用预声明 CoinRun cohort 替换当前从 post-hoc top-five 中排除两条记录后剩余的三-seed 子集，或完整报告全部八个完成 seed。",
        "- 完成 Frozen S1、四项三 seed 消融，以及 paired oracle、reuse intervention 和 predictive-retention 三项 eval-only 诊断；表 2a 资源成本已由现有 artifacts 补齐。",
        "- 补齐时间对齐的单任务曲线后再报告 FT 和 sample efficiency。",
        "",
    ]
    return "\n".join(lines).replace("AWM-AutoRoute", "AWM")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="write the draft")
    mode.add_argument("--check", action="store_true", help="verify the draft")
    args = parser.parse_args()
    content = render()
    if args.write:
        OUTPUT.write_text(content, encoding="utf-8")
        print(f"wrote {OUTPUT}")
    elif args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != content:
            print(f"out of date: {OUTPUT}", file=sys.stderr)
            return 1
        print(f"up to date: {OUTPUT}")
    else:
        print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
