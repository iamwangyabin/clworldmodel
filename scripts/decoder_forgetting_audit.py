#!/usr/bin/env python3
"""Measure decoder-output drift on fixed old DreamerV3 model states.

This supplementary offline audit isolates the decoder itself.  For each old
task-boundary checkpoint C_i, it first creates the old model-state trace
u_i(t) = zh_transform(z_i(t), h_i(t)).  Both the old decoder and every later
decoder receive that exact trace.  Consequently this is not a reconstruction
score for the full current image-to-latent-to-image path, and it cannot be
used to attribute planning or control changes to the decoder.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from artifact_io import (
    sha256_file as _sha256,
    write_json_atomic as _write_json_atomic,
    write_sha256_sidecar as _write_sha256_sidecar,
    write_text_atomic as _write_text_atomic,
)
from component_forgetting_audit import (
    DEFAULT_BOOTSTRAP_REPETITIONS,
    ROOT,
    _load_dataset,
    _model_bundle,
    _write_metrics_npz,
    load_snapshot_specs,
)
from git_provenance import git_state
from input_fixed_module_forgetting_audit import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_BURN_IN,
    MetricSpec,
    _batch_from_dataset,
    _reference_trace,
    _summary_row,
    _torch,
    _validate_model_interface,
    _write_full_checksum_manifest,
)


SCHEMA_VERSION = 1

METRICS = (
    MetricSpec(
        "decoder.output_normalized_rmse",
        "decoder",
        "lower_is_better",
        "Decoder-output RMSE on fixed old model states, normalized by the old decoder output scale.",
    ),
    MetricSpec(
        "decoder.output_pixel_mse",
        "decoder",
        "lower_is_better",
        "Raw decoder-output pixel MSE on fixed old model states.",
    ),
    MetricSpec(
        "decoder.target_pixel_mse",
        "decoder",
        "lower_is_better",
        "Decoder reconstruction pixel MSE against fixed old observations while the decoder input remains frozen.",
    ),
)
METRIC_BY_NAME = {metric.name: metric for metric in METRICS}


def _decode_on_old_states(model: Any, old_model_states: Any, observations: Any) -> Any:
    """Apply exactly one decoder to C_i states and recover the image tensor shape."""

    time_steps, chunk_count, state_dim = old_model_states.shape
    decoded = model.world_model.decoder(old_model_states.reshape(-1, state_dim))
    if tuple(decoded.shape[1:]) != tuple(observations.shape[2:]):
        raise ValueError(
            "Decoder output shape does not match the frozen observation target: "
            f"decoder={tuple(decoded.shape[1:])}, observations={tuple(observations.shape[2:])}"
        )
    return decoded.view(time_steps, chunk_count, *observations.shape[2:])


def _decoder_metrics(
    reference_model: Any,
    comparison_model: Any,
    reference: Mapping[str, Any],
    *,
    burn_in: int,
) -> dict[str, np.ndarray]:
    """Return one decoder-only metric value per frozen diagnostic chunk."""

    old_model_states = reference["zhs"]
    observations = reference["observations"]
    if old_model_states.shape[0] <= burn_in:
        raise ValueError("Burn-in must leave at least one decoder evaluation timestep")

    reference_decoded = _decode_on_old_states(reference_model, old_model_states, observations)
    comparison_decoded = _decode_on_old_states(comparison_model, old_model_states, observations)
    reference_decoded = reference_decoded[burn_in:]
    comparison_decoded = comparison_decoded[burn_in:]
    targets = observations[burn_in:]

    output_error = comparison_decoded - reference_decoded
    output_mse = output_error.square().mean(dim=(0, 2, 3, 4))
    output_rmse = output_mse.sqrt()
    reference_centered = reference_decoded - reference_decoded.mean(
        dim=(0, 2, 3, 4), keepdim=True
    )
    reference_scale = reference_centered.square().mean(dim=(0, 2, 3, 4)).sqrt()
    output_normalized_rmse = output_rmse / reference_scale.clamp_min(1e-8)
    target_mse = (comparison_decoded - targets).square().mean(dim=(0, 2, 3, 4))

    metrics = {
        "decoder.output_normalized_rmse": output_normalized_rmse,
        "decoder.output_pixel_mse": output_mse,
        "decoder.target_pixel_mse": target_mse,
    }
    output: dict[str, np.ndarray] = {}
    for name, values in metrics.items():
        array = values.detach().cpu().numpy().astype(np.float64, copy=False)
        if array.ndim != 1 or not np.isfinite(array).all():
            raise ValueError(f"Malformed {name} values: shape={array.shape}")
        output[name] = array
    return output


def _evaluate_pair(
    reference_model: Any,
    comparison_model: Any,
    dataset: Mapping[str, np.ndarray],
    *,
    burn_in: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Evaluate one C_i/C_j pair without invoking the current encoder or RSSM."""

    torch = _torch()
    _validate_model_interface(reference_model, comparison_model)
    chunk_count = int(dataset["actions"].shape[0])
    all_values: dict[str, list[np.ndarray]] = {metric.name: [] for metric in METRICS}
    with torch.no_grad():
        for start in range(0, chunk_count, batch_size):
            stop = min(start + batch_size, chunk_count)
            batch = _batch_from_dataset(dataset, start, stop, reference_model.device)
            reference = _reference_trace(reference_model, batch)
            metrics = _decoder_metrics(
                reference_model, comparison_model, reference, burn_in=burn_in
            )
            for name, values in metrics.items():
                all_values[name].append(values)
    output = {name: np.concatenate(values) for name, values in all_values.items()}
    for name, values in output.items():
        if values.shape != (chunk_count,) or not np.isfinite(values).all():
            raise ValueError(f"Malformed {name} bundle: shape={values.shape}")
    return output


def _assert_baseline_invariance(values: Mapping[str, np.ndarray]) -> None:
    """A decoder must be exactly unchanged when it is compared to itself."""

    for name in ("decoder.output_normalized_rmse", "decoder.output_pixel_mse"):
        if not np.allclose(values[name], 0.0, atol=1e-8, rtol=1e-8):
            raise RuntimeError(f"Baseline decoder drift is not zero for {name}")


def _render_report(payload: Mapping[str, Any]) -> str:
    """Render a compact supplementary report with no causal-control claim."""

    rows = payload["summary_rows"]
    lines = [
        "# Fixed-Input Decoder Forgetting Audit",
        "",
        "This is a supplementary, offline, single-seed pilot. For every row, both decoders receive the exact C_i model-state trace; encoder and RSSM changes are held out of the measurement.",
        "",
        "The decoder is not part of the planning or action path. These numbers quantify decoder-readout drift only; they do not imply that restoring a decoder would restore return.",
        "",
    ]
    for task in payload["tasks"]:
        task_rows = [row for row in rows if row["task_index"] == task["task_index"]]
        by_checkpoint_metric = {
            (row["comparison_checkpoint"], row["metric"]): row for row in task_rows
        }
        lines.extend(
            [
                f"## {task['task_name']}",
                "",
                "| Later checkpoint | Decoder output nRMSE | Decoder output pixel MSE | Reconstruction-target MSE delta |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for label in task["comparison_checkpoints"]:
            def value(metric_name: str) -> float:
                return float(by_checkpoint_metric[(label, metric_name)]["boundary_relative_forgetting"])

            lines.append(
                "| {label} | {nrmse:.4g} | {output_mse:.4g} | {target_mse:.4g} |".format(
                    label=label,
                    nrmse=value("decoder.output_normalized_rmse"),
                    output_mse=value("decoder.output_pixel_mse"),
                    target_mse=value("decoder.target_pixel_mse"),
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation Limits",
            "",
            "- Decoder nRMSE and pixel MSE are native decoder-readout metrics, not task-return metrics.",
            "- `Cfinal_e540` is excluded because it follows an extra Task 1 update after C6.",
            "- The source run was a dirty-worktree, single-seed pilot; this audit is hypothesis-generating only.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--burn-in", type=int, default=DEFAULT_BURN_IN)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--bootstrap-repetitions", type=int, default=DEFAULT_BOOTSTRAP_REPETITIONS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--include-final", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.burn_in < 1 or args.batch_size < 1 or args.bootstrap_repetitions < 1:
        raise ValueError("burn-in, batch size, and bootstrap repetitions must be positive")
    run_dir = args.run_dir.resolve()
    audit_dir = args.audit_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit output: {output_dir}")

    collection_path = audit_dir / "collection_manifest.json"
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    if not collection.get("complete") or collection.get("artifact_kind") != "component_forgetting_audit_collection":
        raise ValueError("Audit collection is incomplete or has the wrong artifact kind")
    all_specs = load_snapshot_specs(run_dir / "analysis_snapshots")
    boundary_specs = [spec for spec in all_specs if spec.reason == "task_boundary"]
    evaluation_specs = all_specs if args.include_final else boundary_specs
    if len(collection.get("datasets", [])) != len(boundary_specs):
        raise ValueError("Collection datasets do not match task-boundary snapshots")

    source_script = Path(__file__).resolve()
    shared_helper = ROOT / "scripts" / "input_fixed_module_forgetting_audit.py"
    protocol_path = ROOT / "docs" / "protocols" / "decoder_forgetting_audit.md"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "fixed_input_decoder_forgetting_audit",
        "complete": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "pilot_supplementary",
        "project_git": git_state(ROOT),
        "source_script": {"path": str(source_script), "sha256": _sha256(source_script)},
        "shared_input_fixed_helper": {"path": str(shared_helper), "sha256": _sha256(shared_helper)},
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "source_run": str(run_dir),
        "collection_manifest": {"path": str(collection_path), "sha256": _sha256(collection_path)},
        "input_contract": "Each decoder receives the exact C_i zh model-state trace; this audit excludes encoder and RSSM drift.",
        "excluded_checkpoint": None if args.include_final else "Cfinal_e540",
        "arguments": {
            "device": args.device,
            "burn_in": args.burn_in,
            "batch_size": args.batch_size,
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "metrics": [metric.__dict__ for metric in METRICS],
        "input_snapshots": [
            {"label": spec.label, "epoch": spec.epoch, "path": str(spec.path), "sha256": spec.sha256}
            for spec in evaluation_specs
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json_atomic(output_dir / "manifest.json", manifest)

    records: list[dict[str, Any]] = []
    arrays: list[np.ndarray] = []
    summary_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    torch = _torch()
    for task_index, dataset_entry in enumerate(collection["datasets"]):
        task = dataset_entry["task"]
        if int(task["index"]) != task_index:
            raise ValueError("Collection task order is not contiguous")
        reference_spec = boundary_specs[task_index]
        natural = dataset_entry["natural"]
        dataset = _load_dataset(audit_dir / natural["path"], natural["sha256"])
        if int(dataset["actions"].shape[1]) <= args.burn_in:
            raise ValueError("Diagnostic chunk is too short for decoder audit")
        reference_model = _model_bundle(reference_spec, args.device)
        comparison_specs = [spec for spec in evaluation_specs if spec.epoch >= reference_spec.epoch]
        if not comparison_specs or comparison_specs[0].label != reference_spec.label:
            raise ValueError("Each task must begin at its own boundary snapshot")
        task_rows.append(
            {
                "task_index": task_index,
                "task_name": task["name"],
                "reference_checkpoint": reference_spec.label,
                "comparison_checkpoints": [spec.label for spec in comparison_specs],
                "dataset": {"path": natural["path"], "sha256": natural["sha256"]},
            }
        )
        baseline_values: dict[str, np.ndarray] | None = None
        for comparison_spec in comparison_specs:
            comparison_model = (
                reference_model
                if comparison_spec.label == reference_spec.label
                else _model_bundle(comparison_spec, args.device)
            )
            values = _evaluate_pair(
                reference_model,
                comparison_model,
                dataset,
                burn_in=args.burn_in,
                batch_size=args.batch_size,
            )
            if comparison_model is reference_model:
                _assert_baseline_invariance(values)
                baseline_values = values
            elif baseline_values is None:
                raise RuntimeError("Reference decoder metrics must be evaluated first")
            assert baseline_values is not None
            for metric in METRICS:
                current_values = values[metric.name]
                records.append(
                    {
                        "task_index": task_index,
                        "task_name": task["name"],
                        "reference_checkpoint": reference_spec.label,
                        "comparison_checkpoint": comparison_spec.label,
                        "metric": metric.name,
                    }
                )
                arrays.append(current_values)
                summary_rows.append(
                    _summary_row(
                        task_index=task_index,
                        task_name=str(task["name"]),
                        reference_checkpoint=reference_spec,
                        comparison_checkpoint=comparison_spec,
                        metric=metric,
                        reference_values=baseline_values[metric.name],
                        comparison_values=current_values,
                        episode_ids=dataset["episode_ids"],
                        bootstrap_seed=args.bootstrap_seed + task_index * 1000 + comparison_spec.epoch,
                        repetitions=args.bootstrap_repetitions,
                    )
                )
            if comparison_model is not reference_model:
                del comparison_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        del reference_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    raw_metrics_path = output_dir / "per_chunk_decoder_metrics.npz"
    _write_metrics_npz(raw_metrics_path, records, arrays)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "fixed_input_decoder_forgetting_results",
        "complete": True,
        "tasks": task_rows,
        "summary_rows": summary_rows,
        "metric_contract": [metric.__dict__ for metric in METRICS],
        "interpretation": {
            "primary_question": "How much does decoder output change when its old model-state input is held fixed?",
            "causal_limit": "Decoder output drift does not establish a planning or return cause because decoder output is not consumed by the actor.",
        },
    }
    results_path = output_dir / "results.json"
    report_path = output_dir / "DECODER_FORGETTING_REPORT.md"
    _write_json_atomic(results_path, payload)
    _write_text_atomic(report_path, _render_report(payload))
    for path in (output_dir / "manifest.json", raw_metrics_path, results_path, report_path):
        _write_sha256_sidecar(path)
    manifest["complete"] = True
    manifest["outputs"] = {
        "per_chunk_metrics": {"path": raw_metrics_path.name, "sha256": _sha256(raw_metrics_path)},
        "results": {"path": results_path.name, "sha256": _sha256(results_path)},
        "report": {"path": report_path.name, "sha256": _sha256(report_path)},
    }
    _write_json_atomic(output_dir / "manifest.json", manifest)
    _write_sha256_sidecar(output_dir / "manifest.json")
    _write_full_checksum_manifest(output_dir)
    print(f"[fixed-input-decoder-audit] complete output={output_dir}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
