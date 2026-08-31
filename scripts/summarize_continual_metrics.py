#!/usr/bin/env python3
"""Build versioned ARROW-style metric records from completed local runs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clworldmodel.evaluation.metrics import (  # noqa: E402
    METRIC_SCHEMA_VERSION,
    median_iqr,
    normalize_return_matrix,
    single_pass_metrics,
)


DEFAULT_NORMALIZATION = (
    ROOT
    / "docs"
    / "protocols"
    / "references"
    / "arrow_v3_atari_normalization_v1.json"
)
DEFAULT_PAPER_METRICS = (
    ROOT
    / "docs"
    / "protocols"
    / "references"
    / "arrow_v3_atari_reported_metrics_v1.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _config_path(run_dir: Path) -> Path:
    candidates = (
        run_dir / "resolved_training_config.json",
        run_dir / "config.json",
        run_dir / "tensorboard" / "config.json",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No resolved_training_config.json or config.json found under {run_dir}"
    )


def _number_list(line: str, prefix: str) -> list[float]:
    if not line.startswith(prefix):
        raise ValueError(f"Expected {prefix!r}, got {line!r}")
    value = ast.literal_eval(line[len(prefix) :].strip())
    if not isinstance(value, list) or not value:
        raise ValueError(f"{prefix} must contain a non-empty list")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{prefix} contains a non-finite value")
    return result


def _parse_periodic_evaluations(
    log_path: Path, reward_scales: list[float]
) -> list[dict[str, Any]]:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    evaluations: list[dict[str, Any]] = []
    for line_index, line in enumerate(lines):
        if not line.startswith("Eval for epoch:"):
            continue
        epoch = int(line.split(":", 1)[1].strip())
        block = lines[line_index + 1 : line_index + 7]
        by_prefix: dict[str, str] = {}
        for candidate in block:
            for prefix in (
                "Eval means:",
                "Eval stds:",
                "Eval raw means:",
                "Eval raw stds:",
            ):
                if candidate.startswith(prefix):
                    by_prefix[prefix] = candidate
        means = _number_list(by_prefix["Eval means:"], "Eval means:")
        stds = _number_list(by_prefix["Eval stds:"], "Eval stds:")
        if len(means) != len(reward_scales) or len(stds) != len(reward_scales):
            raise ValueError(
                f"Evaluation at epoch {epoch} does not match configured task count"
            )
        if "Eval raw means:" in by_prefix:
            raw_means = _number_list(
                by_prefix["Eval raw means:"], "Eval raw means:"
            )
            raw_stds = _number_list(by_prefix["Eval raw stds:"], "Eval raw stds:")
        else:
            if any(scale == 0 for scale in reward_scales):
                raise ValueError("Cannot recover raw returns from a zero reward scale")
            raw_means = [
                mean / scale for mean, scale in zip(means, reward_scales)
            ]
            raw_stds = [
                std / abs(scale) for std, scale in zip(stds, reward_scales)
            ]
        evaluations.append(
            {
                "completed_epochs": epoch,
                "cohort": "periodic_evaluation",
                "scaled_return_mean": means,
                "scaled_return_std": stds,
                "raw_return_mean": raw_means,
                "raw_return_std": raw_stds,
            }
        )
    if not evaluations:
        raise ValueError(f"No periodic evaluation blocks found in {log_path}")
    epochs = [row["completed_epochs"] for row in evaluations]
    if epochs != sorted(set(epochs)):
        raise ValueError("Periodic evaluation epochs must be unique and increasing")
    return evaluations


def _append_final_evaluation(
    evaluations: list[dict[str, Any]], final_path: Path, task_count: int
) -> None:
    if not final_path.is_file():
        return
    final = _json(final_path)
    tasks = final.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != task_count:
        raise ValueError("final_evaluation.json task count does not match the config")
    completed_epochs = int(final["evaluation_after_completed_epochs"])
    row = {
        "completed_epochs": completed_epochs,
        "cohort": str(final.get("seed_cohort", "final_evaluation")),
        "scaled_return_mean": [float(task["scaled_return_mean"]) for task in tasks],
        "scaled_return_std": [float(task["scaled_return_std"]) for task in tasks],
        "raw_return_mean": [float(task["raw_return_mean"]) for task in tasks],
        "raw_return_std": [float(task["raw_return_std"]) for task in tasks],
    }
    existing = [
        index
        for index, evaluation in enumerate(evaluations)
        if evaluation["completed_epochs"] == completed_epochs
    ]
    if existing:
        evaluations[existing[0]] = row
    else:
        evaluations.append(row)
        evaluations.sort(key=lambda evaluation: evaluation["completed_epochs"])


def _task_durations(config: dict[str, Any], task_count: int) -> list[int]:
    kwargs = config["esc"]["kwargs"]
    durations = kwargs.get("task_durations")
    if durations is not None:
        result = [int(value) for value in durations]
        if len(result) != task_count or any(value <= 0 for value in result):
            raise ValueError("task_durations must match the positive task count")
        return result
    swap = int(kwargs["swap_sched"])
    if swap <= 0:
        raise ValueError("swap_sched must be positive")
    return [swap] * task_count


def _normalization_for_tasks(
    normalization: dict[str, Any], task_names: list[str]
) -> tuple[list[float], list[float], list[dict[str, Any]]]:
    references = {
        str(task["task_name"]): task for task in normalization["tasks"]
    }
    missing = [name for name in task_names if name not in references]
    if missing:
        raise ValueError(f"Normalization reference is missing tasks: {missing}")
    selected = [references[name] for name in task_names]
    return (
        [float(task["random_return"]) for task in selected],
        [float(task["single_task_arrow_return"]) for task in selected],
        selected,
    )


def _policy(config: dict[str, Any], final_path: Path) -> str:
    if final_path.is_file():
        final = _json(final_path)
        policy = final.get("policy")
        if isinstance(policy, str):
            return policy
    task_expert_methods = {
        "moe_arrow",
        "cnn_fullbank_arrow",
        "cnn_projector_lora_arrow",
        "cnn_compact_shared_actor_arrow",
        "cnn_mechanism_bank_arrow",
        "rec_rssm_arrow",
        "evolving_atomic_rssm_arrow",
        "evolving_atomic_rssm_shared_fastkan_arrow",
        "dino_fullbank_arrow",
        "dino_patchbank_arrow",
        "dino_convbank_arrow",
    }
    if config.get("continual_method", "none") in task_expert_methods:
        return "deterministic_argmax_and_latent_mode"
    return "stochastic"


def _budget_signature(
    config: dict[str, Any],
    durations: list[int],
    launch: dict[str, Any],
) -> dict[str, Any]:
    fifo_slots = launch.get("fifo_slots")
    ltdm_slots = launch.get("ltdm_slots")
    if fifo_slots is not None and ltdm_slots is not None:
        replay_sequence_slots = int(fifo_slots) + int(ltdm_slots)
    else:
        replay_sequence_slots = int(config["data_n_max"]) * len(
            config.get("replay_buffers", [])
        )
    sequence_length = int(launch.get("sequence_length", config["data_t"]))
    return {
        "task_duration_epochs": durations,
        "n_sync": int(config["n_sync"]),
        "collection_sequence_length": int(config["gen_seq_len"]),
        "frame_repeat": int(config["env_repeat"]),
        "world_model_updates_per_epoch": int(config["steps_per_batch"]),
        "actor_critic_updates_per_epoch": int(config["ac_train_steps"]),
        "world_model_batch_time": int(config["mb_t_size"]),
        "world_model_batch_sequences": int(config["mb_n_size"]),
        "replay_sequence_slots": replay_sequence_slots,
        "replay_sequence_length": sequence_length,
        "replay_transition_capacity": replay_sequence_slots * sequence_length,
    }


def _resource_accounting(run_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in (
        "model_parameter_accounting.json",
        "actor_critic_parameter_accounting.json",
    ):
        path = run_dir / name
        if path.is_file():
            result[name.removesuffix(".json")] = {
                "path": _portable_path(path),
                "sha256": _sha256(path),
                "data": _json(path),
            }
    return result


def build_run_report(
    run_dir: Path,
    normalization_path: Path = DEFAULT_NORMALIZATION,
) -> dict[str, Any]:
    """Build one self-contained per-seed metric record."""

    run_dir = run_dir.resolve()
    config_path = _config_path(run_dir)
    log_path = run_dir / "train.log"
    launch_path = run_dir / "launch.json"
    final_path = run_dir / "final_evaluation.json"
    if not log_path.is_file():
        raise FileNotFoundError(f"Missing training log: {log_path}")
    config = _json(config_path)
    launch = _json(launch_path) if launch_path.is_file() else {}
    normalization = _json(normalization_path)
    env_configs = config["esc"]["env_configs"]
    task_names = [str(task["name"]) for task in env_configs]
    reward_scales = [float(task["rew_scale"]) for task in env_configs]
    durations = _task_durations(config, len(task_names))
    evaluations = _parse_periodic_evaluations(log_path, reward_scales)
    _append_final_evaluation(evaluations, final_path, len(task_names))

    random_returns, single_task_returns, selected_references = (
        _normalization_for_tasks(normalization, task_names)
    )
    raw_matrix = [row["raw_return_mean"] for row in evaluations]
    normalized_matrix = normalize_return_matrix(
        raw_matrix, random_returns, single_task_returns
    )
    for row, normalized in zip(evaluations, normalized_matrix):
        row["normalized_score"] = normalized
        completed_epochs = int(row["completed_epochs"])
        decisions = (
            completed_epochs * int(config["n_sync"]) * int(config["gen_seq_len"])
        )
        row["agent_decisions"] = decisions
        row["raw_environment_frames"] = decisions * int(config["env_repeat"])

    boundary_epochs: list[int] = []
    cumulative = 0
    for duration in durations:
        cumulative += duration
        boundary_epochs.append(cumulative)
    row_by_epoch = {
        int(row["completed_epochs"]): index for index, row in enumerate(evaluations)
    }
    missing_boundaries = [
        epoch for epoch in boundary_epochs if epoch not in row_by_epoch
    ]
    if missing_boundaries:
        raise ValueError(
            "No evaluation exists at task completion epochs "
            f"{missing_boundaries}; ACC/forgetting would be ambiguous"
        )
    task_end_rows = [row_by_epoch[epoch] for epoch in boundary_epochs]
    metrics = single_pass_metrics(normalized_matrix, task_end_rows)
    for boundary, epoch in zip(metrics["boundaries"], boundary_epochs):
        boundary["completed_epochs"] = epoch

    final_row = task_end_rows[-1]
    per_task = []
    for task_index, task_name in enumerate(task_names):
        acquisition_row = task_end_rows[task_index]
        later_values = [
            normalized_matrix[row][task_index]
            for row in range(acquisition_row + 1, final_row + 1)
        ]
        per_task.append(
            {
                "task_index": task_index,
                "task_name": task_name,
                "reward_scale": reward_scales[task_index],
                "random_return": random_returns[task_index],
                "single_task_arrow_return": single_task_returns[task_index],
                "acquisition_completed_epochs": boundary_epochs[task_index],
                "acquisition_raw_return_mean": raw_matrix[acquisition_row][task_index],
                "acquisition_normalized_score": normalized_matrix[acquisition_row][
                    task_index
                ],
                "final_raw_return_mean": raw_matrix[final_row][task_index],
                "final_raw_return_std": evaluations[final_row]["raw_return_std"][
                    task_index
                ],
                "final_normalized_score": normalized_matrix[final_row][task_index],
                "forgetting": metrics["per_task_forgetting"][task_index],
                "minimum_normalized_score_after_acquisition": (
                    min(later_values)
                    if later_values
                    else normalized_matrix[final_row][task_index]
                ),
            }
        )

    policy = _policy(config, final_path)
    evaluation_epochs = [int(row["completed_epochs"]) for row in evaluations]
    expected_interval = int(
        normalization["paper_evaluation_protocol"]["evaluation_every_epochs"]
    )
    periodic_differences = [
        right - left
        for left, right in zip(evaluation_epochs, evaluation_epochs[1:])
        if right <= boundary_epochs[-1]
    ]
    mismatches = [
        "single-seed record; the paper table reports median [IQR] across five seeds"
    ]
    if policy != "stochastic":
        mismatches.append(
            f"evaluation policy is {policy!r}, while the paper uses a stochastic policy"
        )
    if durations != [90] * len(task_names):
        mismatches.append("task durations differ from the paper's 90 epochs per task")
    if any(difference != expected_interval for difference in periodic_differences):
        mismatches.append("evaluation checkpoints are not uniformly 10 epochs apart")
    if len(task_names) != 6:
        mismatches.append(
            f"partial {len(task_names)}-task curriculum; paper headline uses six tasks"
        )
    evaluation_protocol = config.get("evaluation_seed_protocol", "advancing")
    if evaluation_protocol != "advancing":
        mismatches.append(
            "evaluation cohort protocol differs from the paper's advancing stochastic evaluations"
        )

    method = str(launch.get("method", config.get("algorithm", run_dir.name)))
    return {
        "schema_version": 1,
        "artifact_kind": "continual_metric_report",
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "method": method,
        "protocol": launch.get("protocol"),
        "classification": launch.get("classification", launch.get("role", "unknown")),
        "seed_id": launch.get("seed_id", launch.get("seed_index")),
        "seed": int(config["seed"]),
        "task_order": task_names,
        "source": {
            "run_dir": _portable_path(run_dir),
            "config": _portable_path(config_path),
            "config_sha256": _sha256(config_path),
            "train_log": _portable_path(log_path),
            "train_log_sha256": _sha256(log_path),
            "launch": _portable_path(launch_path) if launch_path.is_file() else None,
            "launch_sha256": _sha256(launch_path) if launch_path.is_file() else None,
            "final_evaluation": _portable_path(final_path) if final_path.is_file() else None,
            "final_evaluation_sha256": _sha256(final_path)
            if final_path.is_file()
            else None,
        },
        "normalization": {
            "reference_id": normalization["reference_id"],
            "reference_path": _portable_path(normalization_path),
            "reference_sha256": _sha256(normalization_path),
            "formula": "q=(raw_return-random)/(single_task_arrow-random)",
            "scores_clipped": False,
            "reference_tasks": selected_references,
        },
        "evaluation_protocol": {
            "policy": policy,
            "cohort_protocol": evaluation_protocol,
            "rollouts_per_task": 16,
            "evaluation_scope": "all configured tasks",
            "evaluation_completed_epochs": evaluation_epochs,
            "task_completion_epochs": boundary_epochs,
            "raw_returns_preserved": True,
            "evaluation_transitions_enter_replay": False,
        },
        "budget_signature": _budget_signature(config, durations, launch),
        "paper_comparability": {
            "status": "diagnostic_per_seed",
            "direct_published_table_comparison": False,
            "mismatches": mismatches,
            "requirements_for_official_aggregate": [
                "five predeclared seeds",
                "same curriculum and per-task budget",
                "stochastic 16-rollout evaluation of every task every 10 epochs",
                "time-aligned single-task curves for forward transfer",
                "median and IQR aggregation across seeds"
            ]
        },
        "metrics": {
            "forgetting": metrics["forgetting"],
            "forward_transfer": None,
            "acc": metrics["acc"],
            "min_acc": metrics["min_acc"],
            "wc_acc": metrics["wc_acc"],
            "max_forgetting": None,
            "recovery": None,
            "named_summary": {
                f"F_{len(task_names)}": metrics["forgetting"],
                f"ACC_{len(task_names)}": metrics["acc"],
                f"min-ACC_{len(task_names)}": metrics["min_acc"],
                f"WC-ACC_{len(task_names)}": metrics["wc_acc"],
            },
            "unavailable": {
                "forward_transfer": (
                    "aligned single-task acquisition curves are not present "
                    "in this run artifact"
                ),
                "max_forgetting": "requires a two-cycle curriculum",
                "recovery": "requires a two-cycle curriculum",
                "sample_efficiency": (
                    "requires five-seed median curves and a shared "
                    "cross-method maximum"
                ),
            },
            "boundaries": metrics["boundaries"],
        },
        "per_task": per_task,
        "evaluation_checkpoints": evaluations,
        "resource_accounting": _resource_accounting(run_dir),
    }


def _comparison_signature(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_order": report["task_order"],
        "policy": report["evaluation_protocol"]["policy"],
        "cohort_protocol": report["evaluation_protocol"]["cohort_protocol"],
        "evaluation_completed_epochs": report["evaluation_protocol"][
            "evaluation_completed_epochs"
        ],
        "budget_signature": report["budget_signature"],
        "normalization_reference_id": report["normalization"]["reference_id"],
    }


def build_comparison(
    reports: list[dict[str, Any]], paper_metrics_path: Path = DEFAULT_PAPER_METRICS
) -> dict[str, Any]:
    if not reports:
        raise ValueError("At least one report is required")
    first_signature = _comparison_signature(reports[0])
    differing = []
    for report in reports[1:]:
        signature = _comparison_signature(report)
        for key in first_signature:
            if signature[key] != first_signature[key]:
                differing.append({"method": report["method"], "field": key})
    seed_sets: dict[str, set[int]] = {}
    for report in reports:
        seed_sets.setdefault(report["method"], set()).add(int(report["seed"]))
    distinct_seed_sets = {tuple(sorted(values)) for values in seed_sets.values()}
    matched_seed_sets = len(distinct_seed_sets) == 1

    paper = _json(paper_metrics_path)
    headline = paper["default_order"]["ARROW-50"]
    table = []
    for report in reports:
        metrics = report["metrics"]
        row = {
            "method": report["method"],
            "seed_id": report["seed_id"],
            "seed": report["seed"],
            "completed_task_count": len(report["task_order"]),
            "forgetting": metrics["forgetting"],
            "forward_transfer": metrics["forward_transfer"],
            "acc": metrics["acc"],
            "min_acc": metrics["min_acc"],
            "wc_acc": metrics["wc_acc"],
            "difference_from_published_arrow_median": {
                metric: (
                    None
                    if metrics[metric] is None
                    else metrics[metric] - float(headline[metric][0])
                )
                for metric in (
                    "forgetting",
                    "forward_transfer",
                    "acc",
                    "min_acc",
                    "wc_acc",
                )
            },
        }
        table.append(row)

    metric_names = (
        "forgetting",
        "forward_transfer",
        "acc",
        "min_acc",
        "wc_acc",
    )
    method_aggregates = []
    for method in sorted(seed_sets):
        method_rows = [row for row in table if row["method"] == method]
        aggregate_metrics: dict[str, Any] = {}
        for metric in metric_names:
            values = [row[metric] for row in method_rows]
            aggregate_metrics[metric] = (
                None
                if any(value is None for value in values)
                else median_iqr([float(value) for value in values])
            )
        seed_count = len(method_rows)
        method_aggregates.append(
            {
                "method": method,
                "seed_count": seed_count,
                "seeds": sorted(int(row["seed"]) for row in method_rows),
                "status": "official_five_seed_candidate"
                if seed_count == 5
                else "diagnostic_insufficient_seeds",
                "metrics": aggregate_metrics,
            }
        )
    return {
        "schema_version": 1,
        "artifact_kind": "continual_metric_comparison",
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "comparison_scope": "local matched per-seed diagnostic",
        "direct_local_comparison_valid": not differing and matched_seed_sets,
        "differing_comparison_fields": differing,
        "matched_seed_sets_across_methods": matched_seed_sets,
        "seed_sets_by_method": {
            method: sorted(values) for method, values in seed_sets.items()
        },
        "directions": {
            "forgetting": "lower",
            "forward_transfer": "higher",
            "acc": "higher",
            "min_acc": "higher",
            "wc_acc": "higher",
        },
        "paper_reference": {
            "path": _portable_path(paper_metrics_path),
            "sha256": _sha256(paper_metrics_path),
            "published_arrow_default_order": headline,
            "warning": paper["comparison_warning"],
        },
        "table": table,
        "method_aggregates": method_aggregates,
        "reports": reports,
    }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--normalization", type=Path, default=DEFAULT_NORMALIZATION)
    parser.add_argument("--paper-metrics", type=Path, default=DEFAULT_PAPER_METRICS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reports = [
        build_run_report(run_dir, args.normalization) for run_dir in args.run_dirs
    ]
    result = build_comparison(reports, args.paper_metrics)
    if args.output is not None:
        _write_json_atomic(args.output.resolve(), result)
        print(f"wrote {args.output.resolve()}")
    print("method\tF↓\tACC↑\tmin-ACC↑\tWC-ACC↑\tFT↑")
    for row in result["table"]:
        values = [
            row["forgetting"],
            row["acc"],
            row["min_acc"],
            row["wc_acc"],
            row["forward_transfer"],
        ]
        rendered = ["NA" if value is None else f"{value:.6f}" for value in values]
        print("\t".join((row["method"], *rendered)))
    print(
        "local matched comparison:",
        "valid" if result["direct_local_comparison_valid"] else "NOT valid",
    )
    print("published-table comparison: diagnostic only until five seeds are aggregated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
