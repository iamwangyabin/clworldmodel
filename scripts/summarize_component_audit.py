#!/usr/bin/env python3
"""Create paired, C6-relative conclusion data from a frozen component audit.

This reporting pass only reads an already-complete held-out collection and its
checkpoint evaluator output.  It does not interact with an environment, write
to replay, or update any model parameter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from component_audit_metrics import paired_episode_bootstrap_difference
from git_provenance import git_state


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
BOUNDARY_LABEL_RE = re.compile(r"^C(?P<task>\d+)_e\d+$")


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    direction: str


HEADLINE_METRICS = (
    MetricSpec("representation.linear_cka", "representation linear CKA", "higher_is_better"),
    MetricSpec("teacher_forced.reconstruction_mse", "teacher-forced reconstruction MSE", "lower_is_better"),
    MetricSpec("teacher_forced.posterior_prior_kl", "posterior-prior KL gap", "diagnostic"),
    MetricSpec("teacher_forced.reward_symlog_mse", "teacher-forced reward symlog MSE", "lower_is_better"),
    MetricSpec("critic.anchored_return_mae", "critic anchored-return MAE", "lower_is_better"),
    MetricSpec("actor.symmetric_kl", "actor symmetric KL", "lower_is_better"),
    MetricSpec("actor.top1_agreement", "actor top-1 agreement", "higher_is_better"),
    MetricSpec("open_loop.h1.visual_mse", "open-loop visual MSE (H=1)", "lower_is_better"),
    MetricSpec("open_loop.h16.visual_mse", "open-loop visual MSE (H=16)", "lower_is_better"),
    MetricSpec("open_loop.h1.reward_symlog_mse", "open-loop reward symlog MSE (H=1)", "lower_is_better"),
    MetricSpec("open_loop.h16.reward_symlog_mse", "open-loop reward symlog MSE (H=16)", "lower_is_better"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Missing SHA-256 sidecar for {path}")
    declared, declared_name = sidecar.read_text(encoding="ascii").split()
    if declared_name != path.name:
        raise ValueError(f"Malformed SHA-256 sidecar for {path}")
    actual = _sha256(path)
    if actual != declared:
        raise ValueError(f"SHA-256 mismatch for {path}")
    return actual


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        temporary.write(text)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _write_sha256_sidecar(path: Path) -> str:
    digest = _sha256(path)
    _write_text_atomic(path.with_suffix(path.suffix + ".sha256"), f"{digest}  {path.name}\n")
    return digest


def _load_raw_metrics(path: Path, expected_sha256: str) -> dict[tuple[int, str, str, str], np.ndarray]:
    if _sha256(path) != expected_sha256:
        raise ValueError(f"Raw metric SHA-256 mismatch for {path}")
    with np.load(path, allow_pickle=False) as archive:
        records = [json.loads(value) for value in archive["records"]]
        offsets = np.asarray(archive["offsets"], dtype=np.int64)
        values = np.asarray(archive["values"], dtype=np.float64)
    if offsets.shape != (len(records) + 1,) or offsets[0] != 0 or offsets[-1] != len(values):
        raise ValueError("Malformed raw metric offsets")
    result: dict[tuple[int, str, str, str], np.ndarray] = {}
    for index, record in enumerate(records):
        key = (
            int(record["task_index"]),
            str(record["dataset_role"]),
            str(record["checkpoint"]),
            str(record["metric"]),
        )
        if key in result:
            raise ValueError(f"Duplicate raw metric record {key}")
        result[key] = values[offsets[index] : offsets[index + 1]]
    return result


def _summary_lookup(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str, str], Mapping[str, Any]]:
    result: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for row in rows:
        if row["dataset_role"] != "natural":
            continue
        key = (int(row["task_index"]), str(row["checkpoint"]), str(row["metric"]))
        if key in result:
            raise ValueError(f"Duplicate summary record {key}")
        result[key] = row
    return result


def _boundary_labels(snapshot_sequence: Sequence[Mapping[str, Any]]) -> dict[int, str]:
    labels: dict[int, str] = {}
    for snapshot in snapshot_sequence:
        match = BOUNDARY_LABEL_RE.match(str(snapshot["label"]))
        if match is None:
            continue
        task_index = int(match.group("task")) - 1
        if task_index in labels:
            raise ValueError(f"Duplicate task-boundary label for task {task_index}")
        labels[task_index] = str(snapshot["label"])
    if not labels or sorted(labels) != list(range(len(labels))):
        raise ValueError("Snapshot sequence does not contain contiguous C1..Cn boundaries")
    return labels


def _relative_ratio(baseline: float, comparison: float) -> float | None:
    if abs(baseline) <= 1e-12:
        return None
    return comparison / baseline


def _paired_row(
    spec: MetricSpec,
    *,
    task_index: int,
    baseline_label: str,
    comparison_label: str,
    episode_ids: np.ndarray,
    raw_metrics: Mapping[tuple[int, str, str, str], np.ndarray],
    summary_rows: Mapping[tuple[int, str, str], Mapping[str, Any]],
    bootstrap_seed: int,
) -> dict[str, Any]:
    summary_key_baseline = (task_index, baseline_label, spec.key)
    summary_key_comparison = (task_index, comparison_label, spec.key)
    if spec.key == "representation.linear_cka":
        baseline = float(summary_rows[summary_key_baseline]["mean"])
        comparison = float(summary_rows[summary_key_comparison]["mean"])
        raw_delta = comparison - baseline
        return {
            "metric": spec.key,
            "label": spec.label,
            "direction": spec.direction,
            "baseline_mean": baseline,
            "comparison_mean": comparison,
            "comparison_minus_baseline": raw_delta,
            "comparison_minus_baseline_ci_low": None,
            "comparison_minus_baseline_ci_high": None,
            "relative_ratio": _relative_ratio(baseline, comparison),
            "degradation_score": baseline - comparison,
            "degradation_ci_low": None,
            "degradation_ci_high": None,
            "n_chunks": None,
            "n_episodes": None,
            "ci_kind": "not available for global CKA",
        }

    raw_key_baseline = (task_index, "natural", baseline_label, spec.key)
    raw_key_comparison = (task_index, "natural", comparison_label, spec.key)
    baseline_values = raw_metrics[raw_key_baseline]
    comparison_values = raw_metrics[raw_key_comparison]
    paired = paired_episode_bootstrap_difference(
        baseline_values,
        comparison_values,
        episode_ids,
        seed=bootstrap_seed,
    )
    raw_delta = float(paired["comparison_minus_baseline"])
    if spec.direction == "higher_is_better":
        degradation = -raw_delta
        ci_low = -float(paired["ci_high"])
        ci_high = -float(paired["ci_low"])
    elif spec.direction == "lower_is_better":
        degradation = raw_delta
        ci_low = float(paired["ci_low"])
        ci_high = float(paired["ci_high"])
    else:
        degradation = None
        ci_low = None
        ci_high = None
    return {
        "metric": spec.key,
        "label": spec.label,
        "direction": spec.direction,
        **paired,
        "comparison_minus_baseline_ci_low": float(paired["ci_low"]),
        "comparison_minus_baseline_ci_high": float(paired["ci_high"]),
        "relative_ratio": _relative_ratio(
            float(paired["baseline_mean"]), float(paired["comparison_mean"])
        ),
        "degradation_score": degradation,
        "degradation_ci_low": ci_low,
        "degradation_ci_high": ci_high,
        "ci_kind": "paired episode-cluster bootstrap",
    }


def _render_number(value: float | None, *, precision: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{precision}g}"


def _render_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3g}x"


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Component Forgetting Conclusion Data (Pilot P1)",
        "",
        "This is a paired, frozen-diagnostic comparison from each acquisition boundary to C6, the actual sixth-task completion checkpoint. It is a single-seed pilot, not an official baseline or causal intervention result.",
        "",
        "## Compact C6 Comparison",
        "",
        "| Old task | C_i to C6 raw return | Rep. CKA | Decoder MSE ratio | Visual H=1 ratio | Visual H=16 ratio | Actor top-1 agreement | Actor symmetric KL |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task in payload["tasks"]:
        metrics = {row["metric"]: row for row in task["headline_metrics"]}
        returns = task["end_to_end_return"]
        lines.append(
            "| {task} | {acq:.4g} to {final:.4g} | {cka} | {decoder} | {h1} | {h16} | {agreement} | {kl} |".format(
                task=task["task_name"],
                acq=returns["acquisition_raw_return"],
                final=returns["c6_raw_return"],
                cka=_render_number(metrics["representation.linear_cka"]["comparison_mean"]),
                decoder=_render_ratio(metrics["teacher_forced.reconstruction_mse"]["relative_ratio"]),
                h1=_render_ratio(metrics["open_loop.h1.visual_mse"]["relative_ratio"]),
                h16=_render_ratio(metrics["open_loop.h16.visual_mse"]["relative_ratio"]),
                agreement=_render_number(metrics["actor.top1_agreement"]["comparison_mean"]),
                kl=_render_number(metrics["actor.symmetric_kl"]["comparison_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Paired Component Data",
            "",
            "`C6 - C_i` is the raw signed change. `Degradation` is positive only for metrics with a declared better direction; the posterior-prior KL is shown as a diagnostic gap rather than automatically called better or worse.",
        ]
    )
    for task in payload["tasks"]:
        lines.extend(
            [
                "",
                f"### {task['task_name']} ({task['baseline_checkpoint']} to {task['comparison_checkpoint']})",
                "",
                "| Metric | C_i | C6 | C6 - C_i | 95% CI for C6 - C_i | Degradation |",
                "| --- | ---: | ---: | ---: | --- | ---: |",
            ]
        )
        for row in task["headline_metrics"]:
            ci = (
                "n/a"
                if row["comparison_minus_baseline_ci_low"] is None
                else "[{low}, {high}]".format(
                    low=_render_number(row["comparison_minus_baseline_ci_low"]),
                    high=_render_number(row["comparison_minus_baseline_ci_high"]),
                )
            )
            lines.append(
                "| {label} | {baseline} | {comparison} | {delta} | {ci} | {degradation} |".format(
                    label=row["label"],
                    baseline=_render_number(row["baseline_mean"]),
                    comparison=_render_number(row["comparison_mean"]),
                    delta=_render_number(row["comparison_minus_baseline"]),
                    ci=ci,
                    degradation=_render_number(row["degradation_score"]),
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation Boundaries",
            "",
            "- The natural headline set has no event-balanced subset in this pass, so terminal/continue-head conclusions are not supported.",
            "- A large actor change can arise from actor parameters, latent-to-actor interface drift, or both; it is not actor-only causal evidence.",
            "- C6 is used for continual retention. `Cfinal_e540` is intentionally excluded because it follows one additional Task 1 update.",
            "- The source training worktree was dirty at launch. These outputs remain a pilot; collector/evaluator and reporter provenance are recorded separately in the JSON artifact.",
        ]
    )
    return "\n".join(lines) + "\n"


def summarize(args: argparse.Namespace) -> None:
    audit_dir = args.audit_dir.resolve()
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing summary output {output_dir}")
    report_git = git_state(ROOT)
    collection_path = audit_dir / "collection_manifest.json"
    results_path = results_dir / "results.json"
    collection_digest = _validate_sha256_sidecar(collection_path)
    results_digest = _validate_sha256_sidecar(results_path)
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if not collection.get("complete") or not results.get("complete"):
        raise ValueError("Both source artifacts must be complete")
    if collection.get("role") != results.get("role"):
        raise ValueError("Collection/result role mismatch")
    if collection.get("role") != "pilot":
        raise ValueError("This reporter only accepts explicitly labelled pilot artifacts")

    raw_path = results_dir / str(results["raw_metrics"]["path"])
    raw_sidecar_digest = _validate_sha256_sidecar(raw_path)
    if raw_sidecar_digest != str(results["raw_metrics"]["sha256"]):
        raise ValueError("Raw metric sidecar does not match results.json")
    raw_metrics = _load_raw_metrics(raw_path, str(results["raw_metrics"]["sha256"]))
    boundary_labels = _boundary_labels(results["snapshot_sequence"])
    comparison_task_index = max(boundary_labels)
    comparison_label = boundary_labels[comparison_task_index]
    summary_rows = _summary_lookup(results["summary_records"])
    returns = {int(row["task_index"]): row for row in results["end_to_end_returns"]["rows"]}
    datasets = collection["datasets"]
    if len(datasets) != len(boundary_labels):
        raise ValueError("Collection task count does not match snapshot boundaries")

    tasks = []
    for task_index in range(comparison_task_index):
        entry = datasets[task_index]
        baseline_label = boundary_labels[task_index]
        natural = entry["natural"]
        natural_path = audit_dir / str(natural["path"])
        if _sha256(natural_path) != str(natural["sha256"]):
            raise ValueError(f"Natural dataset SHA-256 mismatch for task {task_index}")
        with np.load(natural_path, allow_pickle=False) as archive:
            episode_ids = np.asarray(archive["episode_ids"])
        task_metrics = [
            _paired_row(
                spec,
                task_index=task_index,
                baseline_label=baseline_label,
                comparison_label=comparison_label,
                episode_ids=episode_ids,
                raw_metrics=raw_metrics,
                summary_rows=summary_rows,
                bootstrap_seed=args.bootstrap_seed + task_index * 10_000 + metric_index,
            )
            for metric_index, spec in enumerate(HEADLINE_METRICS)
        ]
        tasks.append(
            {
                "task_index": task_index,
                "task_name": entry["task"]["name"],
                "baseline_checkpoint": baseline_label,
                "comparison_checkpoint": comparison_label,
                "end_to_end_return": returns[task_index],
                "headline_metrics": task_metrics,
            }
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "component_forgetting_conclusion_data",
        "complete": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "pilot",
        "report_git": report_git,
        "source": {
            "collection_manifest": {
                "path": str(collection_path),
                "sha256": collection_digest,
            },
            "results": {"path": str(results_path), "sha256": results_digest},
            "raw_metrics": {
                "path": str(raw_path),
                "sha256": str(results["raw_metrics"]["sha256"]),
            },
            "evaluator_git": results["project_git"],
        },
        "comparison": {
            "baseline": "each task's acquisition boundary C_i",
            "checkpoint": comparison_label,
            "excludes": "Cfinal_e540 because it includes one additional Task 1 update",
            "bootstrap": "1,000 paired episode-cluster resamples",
        },
        "tasks": tasks,
        "limitations": {
            "seed_count": 1,
            "source_training_label": "pilot; original training worktree was dirty at launch",
            "event_subset": "not collected in this natural-headline pass",
            "causal_intervention": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "component_conclusion_data.json"
    _write_json_atomic(data_path, payload)
    data_digest = _write_sha256_sidecar(data_path)
    markdown_path = output_dir / "COMPONENT_CONCLUSIONS.md"
    _write_text_atomic(markdown_path, _markdown(payload))
    markdown_digest = _write_sha256_sidecar(markdown_path)
    manifest_path = output_dir / "summary_manifest.json"
    _write_json_atomic(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "component_forgetting_conclusion_summary",
            "complete": True,
            "files": {
                data_path.name: data_digest,
                markdown_path.name: markdown_digest,
            },
            "source": payload["source"],
        },
    )
    _write_sha256_sidecar(manifest_path)
    print(f"[audit-summary] complete output={output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=910_000)
    args = parser.parse_args()
    summarize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
