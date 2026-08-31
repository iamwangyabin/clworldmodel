#!/usr/bin/env python3
"""Render a single, detailed dossier from the completed P1 audit artifacts.

This reporting command only reads completed local result artifacts. It does not
interact with an environment, modify parameters, or regenerate any measurement.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from artifact_io import (
    sha256_file as _sha256,
    write_json_atomic as _write_json_atomic,
    write_text_atomic as _write_text_atomic,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
SCHEMA_VERSION = 1


def _format(value: Any, *, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    if number == 0:
        return "0"
    return f"{number:.{precision}g}"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _require_complete(value: Mapping[str, Any], path: Path) -> None:
    if value.get("complete") is not True:
        raise ValueError(f"Result artifact is incomplete: {path}")


def _row_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str, str], Mapping[str, Any]]:
    index: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (int(row["task_index"]), str(row["comparison_checkpoint"]), str(row["metric"]))
        if key in index:
            raise ValueError(f"Duplicate summary row {key}")
        index[key] = row
    return index


def _result_trajectory(
    result: Mapping[str, Any],
    metric: str,
    *,
    task_index: int,
    field: str = "comparison_mean",
) -> tuple[str, Mapping[str, Any]]:
    index = _row_index(result["summary_rows"])
    task = result["tasks"][task_index]
    pieces: list[str] = []
    final_row: Mapping[str, Any] | None = None
    for checkpoint in task["comparison_checkpoints"]:
        row = index[(task_index, checkpoint, metric)]
        pieces.append(f"{checkpoint}={_format(row[field])}")
        final_row = row
    if final_row is None:
        raise ValueError(f"No rows for task={task_index}, metric={metric}")
    return " -> ".join(pieces), final_row


def _render_source_header(
    launch: Mapping[str, Any], config: Mapping[str, Any], collection: Mapping[str, Any]
) -> list[str]:
    protocol = collection["protocol"]
    return [
        "# DreamerV3/FIFO P1 Forgetting Audit Dossier",
        "",
        "## Status",
        "",
        "This is a complete synthesis of the finished offline audits for one single-seed pilot. It is diagnostic evidence, not an official multi-seed result or a causal intervention result.",
        "",
        "- Source run: `DreamerV3/FIFO` matched control, original six-task order, seed `123456789`.",
        f"- Training snapshots: task boundaries `C1_e89` through `C6_e539`; `Cfinal_e540` is excluded because it follows an extra Task 1 update.",
        f"- FIFO replay: `{launch['fifo_slots']}` slots x sequence length `{launch['sequence_length']}`; LTDM slots: `{launch['ltdm_slots']}`.",
        f"- Observation/action protocol: `{config['img_size']}x{config['img_size']}` RGB Atari, action space `{config['action_space']}`, environment repeat `{config['env_repeat']}`.",
        f"- Audit set: `{protocol['natural_chunks_per_task']}` held-out natural chunks per task, length `{protocol['chunk_length']}`, burn-in `{protocol['burn_in']}`, episode-cluster bootstrap `1000` repetitions.",
        "- Audit forwards are offline: no evaluation transition entered replay and no audit changed model parameters.",
        "- Source launch worktree was dirty. Preserve the pilot label in every use of these results.",
        "",
        "## Task Order",
        "",
        "| Boundary | Task | Reward scale |",
        "| --- | --- | ---: |",
        *[
            f"| C{index + 1}_e{89 + 90 * index} | {entry['name']} | {_format(entry['rew_scale'])} |"
            for index, entry in enumerate(config["esc"]["env_configs"])
        ],
        "",
    ]


def _render_return_history(component: Mapping[str, Any]) -> list[str]:
    lines = [
        "## End-to-End Continual Returns",
        "",
        "All values are raw environment returns. `C6` is the actual sixth-task completion boundary.",
        "",
        "| Old task | Acquisition | C6 | C6 / acquisition | Boundary forgetting | Full evaluation history |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in component["end_to_end_returns"]["rows"]:
        history = " -> ".join(
            f"e{point['epoch']}={_format(point['raw_return'])}" for point in row["history"]
        )
        retention = row["c6_raw_return"] / row["acquisition_raw_return"]
        lines.append(
            "| {task} | {acq} | {final} | {retention:.2%} | {forgetting} | {history} |".format(
                task=row["task_name"],
                acq=_format(row["acquisition_raw_return"]),
                final=_format(row["c6_raw_return"]),
                retention=retention,
                forgetting=_format(row["boundary_forgetting"]),
                history=history,
            )
        )
    lines.extend(["", "Enduro is the current sixth task at C6, so it has no later-task retention interval.", ""])
    return lines


def _render_metric_group(
    title: str,
    result: Mapping[str, Any],
    metrics: Sequence[str],
    *,
    final_label: str = "C6_e539",
) -> list[str]:
    contracts = {metric["name"]: metric for metric in result["metric_contract"]}
    lines = [f"## {title}", ""]
    for metric in metrics:
        contract = contracts[metric]
        lines.extend(
            [
                f"### `{metric}`",
                "",
                contract["description"],
                "",
                "| Old task | Full boundary trajectory | Final value | Final 95% CI | Chunks / episodes |",
                "| --- | --- | ---: | --- | ---: |",
            ]
        )
        for task_index, task in enumerate(result["tasks"][:-1]):
            trajectory, final_row = _result_trajectory(result, metric, task_index=task_index)
            if final_row["comparison_checkpoint"] != final_label:
                raise ValueError(f"Expected final boundary {final_label} for {metric}")
            lines.append(
                "| {task} | {trajectory} | {value} | [{low}, {high}] | {chunks} / {episodes} |".format(
                    task=task["task_name"],
                    trajectory=trajectory,
                    value=_format(final_row["comparison_mean"]),
                    low=_format(final_row["comparison_ci_low"]),
                    high=_format(final_row["comparison_ci_high"]),
                    chunks=final_row["n_chunks"],
                    episodes=final_row["n_episodes"],
                )
            )
        lines.append("")
    return lines


def _render_swap_summary(swap: Mapping[str, Any]) -> list[str]:
    lines = [
        "## Parameter-Swap Restoration Probe",
        "",
        "Each condition replaces selected `C6` parameters with the old task-boundary parameters and evaluates the same natural chunks. This tests interface dependence, not a train-time causal intervention. Visual metrics are teacher-forced reconstruction MSE; actor metrics compare the resulting policy with the old policy.",
        "",
    ]
    for task in swap["tasks"]:
        lines.extend(
            [
                f"### {task['task_name']} ({task['baseline_checkpoint']} -> {task['comparison_checkpoint']})",
                "",
                "| Condition | Reconstruction MSE | H=1 visual MSE | H=16 visual MSE | Actor top-1 agreement | Actor KL |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for condition in task["conditions"]:
            metrics = condition["metrics"]
            lines.append(
                "| {name} | {recon} | {h1} | {h16} | {agree} | {actor_kl} |".format(
                    name=condition["name"],
                    recon=_format(metrics["teacher_forced.reconstruction_mse"]["comparison_mean"]),
                    h1=_format(metrics["open_loop.h1.visual_mse"]["comparison_mean"]),
                    h16=_format(metrics["open_loop.h16.visual_mse"]["comparison_mean"]),
                    agree=_format(metrics["actor.top1_agreement"]["comparison_mean"]),
                    actor_kl=_format(metrics["actor.symmetric_kl"]["comparison_mean"]),
                )
            )
        lines.append("")
    return lines


def _render_interpretation() -> list[str]:
    return [
        "## What This Evidence Supports",
        "",
        "1. Encoder direct coordinates drift strongly on old images. A high CKA geometry score does not negate this: CKA intentionally ignores some coordinate changes.",
        "2. The posterior, recurrent transition, and prior all drift on their frozen old inputs. This is a latent-interface problem, not just a decoder problem.",
        "3. The actor and critic also change materially on frozen old states; world-model retention alone will not guarantee policy retention.",
        "4. Decoder restoration repairs visual readout but leaves actor outputs unchanged, so decoder drift is an auxiliary world-model retention signal rather than the direct control cause.",
        "5. The continue-head natural-data result is not sufficient evidence because terminal events are sparse or absent in several held-out collections.",
        "",
        "## Next Causal Tests",
        "",
        "- Fit scalar, per-channel affine, orthogonal, and small linear adapters from later encoder features to old features on disjoint calibration chunks; test their ability to restore direct feature RMS, posterior KL, and old-world-model reconstruction on held-out chunks.",
        "- In retraining, compare matched-budget replay-based functional anchors: encoder only; encoder plus RSSM posterior/prior/recurrent; encoder plus RSSM plus actor-critic; and the same method with decoder/reward auxiliary terms.",
        "- Keep replay capacity, bytes, sampling, update count, task order, and seeds matched; evaluate returns and rerun this dossier per seed.",
        "",
    ]


def _render_artifact_inventory(paths: Mapping[str, Path]) -> list[str]:
    lines = [
        "## Source Artifact Inventory",
        "",
        "Every table above is rendered from the structured JSON below. The native reports and per-chunk NPZ arrays remain the authoritative detailed records.",
        "",
        "| Artifact | Relative path | SHA-256 |",
        "| --- | --- | --- |",
    ]
    for name, path in paths.items():
        lines.append(f"| {name} | `{path.relative_to(ROOT)}` | `{_sha256(path)}` |")
    lines.append("")
    return lines


def _write_checksum_manifest(output_dir: Path) -> None:
    paths = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    content = "".join(f"{_sha256(path)}  {path.name}\n" for path in paths)
    _write_text_atomic(output_dir / "SHA256SUMS", content)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUNS / "dv3_fifo_original_s0_analysis_full_audit_dossier_p1",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing dossier: {output_dir}")

    source_paths = {
        "launch manifest": RUNS / "dv3_fifo_original_s0_analysis" / "launch.json",
        "resolved config": RUNS / "dv3_fifo_original_s0_analysis" / "config.json",
        "run status": RUNS / "dv3_fifo_original_s0_analysis" / "run_status.json",
        "held-out collection manifest": RUNS / "dv3_fifo_original_s0_analysis_component_audit_p1_natural" / "collection_manifest.json",
        "end-to-end component audit": RUNS / "dv3_fifo_original_s0_analysis_component_audit_p1_natural_results" / "results.json",
        "input-fixed module audit": RUNS / "dv3_fifo_original_s0_analysis_input_fixed_module_audit_v2_p1_r2" / "results.json",
        "direct encoder audit": RUNS / "dv3_fifo_original_s0_analysis_encoder_feature_forgetting_p1" / "results.json",
        "decoder audit": RUNS / "dv3_fifo_original_s0_analysis_decoder_forgetting_p1" / "results.json",
        "parameter-swap audit": RUNS / "dv3_fifo_original_s0_analysis_component_swap_audit_p1" / "swap_results.json",
        "dossier renderer": Path(__file__).resolve(),
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing source artifacts:\n" + "\n".join(missing))

    launch = _read_json(source_paths["launch manifest"])
    config = _read_json(source_paths["resolved config"])
    status = _read_json(source_paths["run status"])
    collection = _read_json(source_paths["held-out collection manifest"])
    component = _read_json(source_paths["end-to-end component audit"])
    module = _read_json(source_paths["input-fixed module audit"])
    encoder = _read_json(source_paths["direct encoder audit"])
    decoder = _read_json(source_paths["decoder audit"])
    swap = _read_json(source_paths["parameter-swap audit"])
    for value, path in (
        (collection, source_paths["held-out collection manifest"]),
        (component, source_paths["end-to-end component audit"]),
        (module, source_paths["input-fixed module audit"]),
        (encoder, source_paths["direct encoder audit"]),
        (decoder, source_paths["decoder audit"]),
        (swap, source_paths["parameter-swap audit"]),
    ):
        _require_complete(value, path)
    if status.get("complete") is not True or status.get("return_code") != 0:
        raise ValueError("Source training run is not recorded as successfully complete")

    lines = _render_source_header(launch, config, collection)
    lines.extend(_render_return_history(component))
    lines.extend(
        _render_metric_group(
            "Direct Encoder-Feature Perturbation (Primary Encoder Measurement)",
            encoder,
            [
                "encoder.feature_relative_rms_perturbation",
                "encoder.feature_rms_perturbation",
            ],
        )
    )
    lines.extend(
        _render_metric_group(
            "Input-Fixed Encoder Geometry and Latent Core",
            module,
            [
                "encoder.linear_cka",
                "encoder.procrustes_residual",
                "posterior.symmetric_kl",
                "rssm.recurrent_normalized_rmse",
                "rssm.prior_symmetric_kl",
            ],
        )
    )
    lines.extend(
        _render_metric_group(
            "Input-Fixed Reward, Continue, Actor, and Critic Heads",
            module,
            [
                "reward_head.output_symlog_mae",
                "reward_head.target_symlog_mse",
                "continue_head.output_probability_mae",
                "continue_head.target_bce",
                "actor_head.symmetric_kl",
                "actor_head.top1_agreement",
                "critic_head.distribution_symmetric_kl",
                "critic_head.value_reference_mae",
                "critic_head.anchored_return_mae",
            ],
        )
    )
    lines.extend(
        _render_metric_group(
            "Fixed-Input Decoder Readout Drift",
            decoder,
            [
                "decoder.output_normalized_rmse",
                "decoder.output_pixel_mse",
                "decoder.target_pixel_mse",
            ],
        )
    )
    lines.extend(_render_swap_summary(swap))
    lines.extend(_render_interpretation())
    lines.extend(_render_artifact_inventory(source_paths))

    report_path = output_dir / "FULL_AUDIT_DOSSIER_P1.md"
    manifest_path = output_dir / "manifest.json"
    _write_text_atomic(report_path, "\n".join(lines))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "p1_full_forgetting_audit_dossier",
        "complete": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "pilot_synthesis",
        "source_artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)} for name, path in source_paths.items()
        },
        "outputs": {"report": {"path": report_path.name, "sha256": _sha256(report_path)}},
    }
    _write_json_atomic(manifest_path, manifest)
    _write_checksum_manifest(output_dir)
    print(f"[p1-full-audit-dossier] complete output={output_dir}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
