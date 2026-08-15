#!/usr/bin/env python3
"""Probe component-level retention with frozen checkpoint parameter swaps.

Each condition starts from the final continual checkpoint C6 and restores a
named parameter group from the relevant acquisition checkpoint C_i.  The
script only evaluates fixed held-out audit chunks; it never collects new data,
updates parameters, or modifies the original snapshots.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from component_audit_metrics import paired_episode_bootstrap_difference
from component_forgetting_audit import (
    ROOT,
    _append_actor_comparisons,
    _evaluate_one_checkpoint,
    _load_dataset,
    _model_bundle,
    _sha256,
    _write_json_atomic,
    _write_npz_atomic,
    _write_sha256_sidecar,
    _write_text_atomic,
    load_snapshot_specs,
)
from git_provenance import git_state


SCHEMA_VERSION = 1
KEY_METRICS = (
    "teacher_forced.reconstruction_mse",
    "teacher_forced.posterior_prior_kl",
    "teacher_forced.reward_symlog_mse",
    "critic.anchored_return_mae",
    "actor.symmetric_kl",
    "actor.top1_agreement",
    "open_loop.h1.visual_mse",
    "open_loop.h16.visual_mse",
    "open_loop.h1.reward_symlog_mse",
    "open_loop.h16.reward_symlog_mse",
)
HIGHER_IS_BETTER = frozenset({"actor.top1_agreement", "representation.linear_cka"})
DIAGNOSTIC_ONLY = frozenset({"teacher_forced.posterior_prior_kl"})


@dataclass(frozen=True)
class SwapCondition:
    name: str
    description: str
    world_model_mode: str
    world_model_prefixes: tuple[str, ...] = ()
    actor_mode: str = "c6"


CONDITIONS = (
    SwapCondition("Ci_reference", "Acquisition checkpoint reference.", "ci", actor_mode="ci"),
    SwapCondition("C6_retention", "Final continual checkpoint without restoration.", "c6"),
    SwapCondition(
        "C6_old_decoder",
        "C6 with decoder parameters restored from C_i.",
        "restore",
        ("decoder.",),
    ),
    SwapCondition(
        "C6_old_encoder_posterior",
        "C6 with RSSM image encoder and posterior restored from C_i.",
        "restore",
        ("rssm.image_embedder.", "rssm.representation."),
    ),
    SwapCondition(
        "C6_old_dynamics",
        "C6 with RSSM recurrent and transition parameters restored from C_i.",
        "restore",
        ("rssm.recurrent.", "rssm.transition."),
    ),
    SwapCondition(
        "C6_old_rssm",
        "C6 with the complete RSSM restored from C_i.",
        "restore",
        ("rssm.",),
    ),
    SwapCondition(
        "C6_old_rssm_decoder",
        "C6 with the co-adapted RSSM and decoder restored from C_i.",
        "restore",
        ("rssm.", "decoder."),
    ),
    SwapCondition(
        "C6_old_reward_continue",
        "C6 with reward and continue heads restored from C_i.",
        "restore",
        ("reward_fc.", "continue_fc."),
    ),
    SwapCondition(
        "C6_old_actor",
        "C6 world model with acquisition actor/critic parameters.",
        "c6",
        actor_mode="ci",
    ),
    SwapCondition(
        "Ci_world_model_C6_actor",
        "Acquisition world model with final actor/critic parameters.",
        "ci",
        actor_mode="c6",
    ),
)


def _validate_sidecar(path: Path, expected_sha256: str | None = None) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Missing SHA-256 sidecar for {path}")
    declared, name = sidecar.read_text(encoding="ascii").split()
    if name != path.name:
        raise ValueError(f"Malformed SHA-256 sidecar for {path}")
    actual = _sha256(path)
    if actual != declared or (expected_sha256 is not None and actual != expected_sha256):
        raise ValueError(f"SHA-256 mismatch for {path}")
    return actual


def _merged_state_dict(
    base: Mapping[str, Any], source: Mapping[str, Any], prefixes: Sequence[str]
) -> dict[str, Any]:
    """Return a strict, non-mutating parameter-group restoration from ``source``."""
    if set(base) != set(source):
        raise ValueError("Swap source and target state dictionaries have different keys")
    result = dict(base)
    changed = 0
    for key in result:
        if any(key.startswith(prefix) for prefix in prefixes):
            result[key] = source[key]
            changed += 1
    if changed == 0:
        raise ValueError(f"No state-dict keys matched restoration prefixes {tuple(prefixes)}")
    return result


def _condition_model(
    condition: SwapCondition,
    ci: Any,
    c6: Any,
    *,
    device: str,
) -> Any:
    """Instantiate C6 architecture and replace only the named parameter groups."""
    model = _model_bundle(c6, device)
    ci_world_model = ci.payload["world_model_state_dict"]
    c6_world_model = c6.payload["world_model_state_dict"]
    if condition.world_model_mode == "ci":
        model.world_model.load_state_dict(ci_world_model, strict=True)
    elif condition.world_model_mode == "restore":
        model.world_model.load_state_dict(
            _merged_state_dict(c6_world_model, ci_world_model, condition.world_model_prefixes),
            strict=True,
        )
    elif condition.world_model_mode != "c6":
        raise ValueError(f"Unknown world-model mode {condition.world_model_mode!r}")

    if condition.actor_mode == "ci":
        model.actor_critic.load_state_dict(ci.payload["actor_critic_state_dict"], strict=True)
    elif condition.actor_mode != "c6":
        raise ValueError(f"Unknown actor mode {condition.actor_mode!r}")
    model.world_model.eval()
    model.actor_critic.eval()
    return model


def _direction(metric: str) -> str:
    if metric in DIAGNOSTIC_ONLY:
        return "diagnostic"
    return "higher_is_better" if metric in HIGHER_IS_BETTER else "lower_is_better"


def _paired_condition_summary(
    baseline_values: np.ndarray,
    comparison_values: np.ndarray,
    episode_ids: np.ndarray,
    *,
    metric: str,
    seed: int,
) -> dict[str, Any]:
    paired = paired_episode_bootstrap_difference(
        baseline_values, comparison_values, episode_ids, seed=seed
    )
    direction = _direction(metric)
    raw_delta = float(paired["comparison_minus_baseline"])
    if direction == "higher_is_better":
        improvement = raw_delta
        improvement_ci_low = float(paired["ci_low"])
        improvement_ci_high = float(paired["ci_high"])
    elif direction == "lower_is_better":
        improvement = -raw_delta
        improvement_ci_low = -float(paired["ci_high"])
        improvement_ci_high = -float(paired["ci_low"])
    else:
        improvement = None
        improvement_ci_low = None
        improvement_ci_high = None
    return {
        "direction": direction,
        **paired,
        "improvement_over_c6": improvement,
        "improvement_ci_low": improvement_ci_low,
        "improvement_ci_high": improvement_ci_high,
    }


def _scalar_condition_summary(
    baseline_value: float, comparison_value: float, *, metric: str
) -> dict[str, Any]:
    raw_delta = comparison_value - baseline_value
    direction = _direction(metric)
    improvement = raw_delta if direction == "higher_is_better" else -raw_delta
    return {
        "direction": direction,
        "baseline_mean": baseline_value,
        "comparison_mean": comparison_value,
        "comparison_minus_baseline": raw_delta,
        "n_chunks": None,
        "n_episodes": None,
        "ci_low": None,
        "ci_high": None,
        "improvement_over_c6": improvement,
        "improvement_ci_low": None,
        "improvement_ci_high": None,
    }


def _metric_rows(
    *,
    c6_output: Mapping[str, Any],
    condition_output: Mapping[str, Any],
    episode_ids: np.ndarray,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    summary: dict[str, Any] = {}
    raw: dict[str, np.ndarray] = {}
    for metric in KEY_METRICS:
        baseline_values = c6_output["per_chunk"][metric]
        comparison_values = condition_output["per_chunk"][metric]
        raw[metric] = np.asarray(comparison_values, dtype=np.float32)
        summary[metric] = _paired_condition_summary(
            baseline_values,
            comparison_values,
            episode_ids,
            metric=metric,
            seed=seed,
        )
    metric = "representation.linear_cka"
    c6_value = float(c6_output[metric])
    condition_value = float(condition_output[metric])
    summary[metric] = _scalar_condition_summary(c6_value, condition_value, metric=metric)
    return summary, raw


def _discard_comparison_tensors(output: dict[str, Any]) -> None:
    """Drop large paired-feature arrays once actor/CKA metrics have been materialized."""
    output.pop("features", None)
    output.pop("actor_log_probs", None)


def _write_raw_metrics(path: Path, records: Sequence[Mapping[str, Any]], values: Sequence[np.ndarray]) -> str:
    offsets = [0]
    flattened = []
    for value in values:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        flattened.append(vector)
        offsets.append(offsets[-1] + len(vector))
    return _write_npz_atomic(
        path,
        {
            "records": np.asarray([json.dumps(record, sort_keys=True) for record in records]),
            "offsets": np.asarray(offsets, dtype=np.int64),
            "values": np.concatenate(flattened) if flattened else np.asarray([], dtype=np.float32),
        },
    )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4g}"


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Component Swap Restoration Audit (Pilot P1)",
        "",
        "Each row restores named C_i parameter groups into C6, then evaluates the same fixed task-i natural chunks. Positive `improvement over C6` means movement in the metric's declared better direction.",
        "",
    ]
    for task in payload["tasks"]:
        lines.extend(
            [
                f"## {task['task_name']} ({task['baseline_checkpoint']} to {task['comparison_checkpoint']})",
                "",
                "| Condition | TF reconstruction MSE | H=1 visual MSE | H=16 visual MSE | Rep. CKA vs C_i | Actor top-1 agreement | Actor symmetric KL |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for condition in task["conditions"]:
            metric = condition["metrics"]
            lines.append(
                "| {name} | {recon} | {h1} | {h16} | {cka} | {agreement} | {kl} |".format(
                    name=condition["name"],
                    recon=_fmt(metric["teacher_forced.reconstruction_mse"]["comparison_mean"]),
                    h1=_fmt(metric["open_loop.h1.visual_mse"]["comparison_mean"]),
                    h16=_fmt(metric["open_loop.h16.visual_mse"]["comparison_mean"]),
                    cka=_fmt(metric["representation.linear_cka"]["comparison_mean"]),
                    agreement=_fmt(metric["actor.top1_agreement"]["comparison_mean"]),
                    kl=_fmt(metric["actor.symmetric_kl"]["comparison_mean"]),
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation Boundaries",
            "",
            "- A single-group swap can fail because adjacent modules co-adapted to a different latent coordinate system. It narrows functional dependency; it is not a proof that the swapped group alone caused training-time forgetting.",
            "- `C6_old_rssm_decoder` is the co-adapted visual-world-model restoration condition. Compare it with decoder-only and RSSM-only before proposing a component-specific replay intervention.",
            "- The actor swaps are interface probes: old actor on C6 latents and C6 actor on old latents distinguish parameter drift from latent-readout incompatibility only jointly with the world-model conditions.",
            "- This remains a single-seed pilot based on a dirty-launch source training run. It is hypothesis-generating, not a paper claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_swap_audit(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    audit_dir = args.audit_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing swap-audit output {output_dir}")
    collection_path = audit_dir / "collection_manifest.json"
    _validate_sidecar(collection_path)
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    if not collection.get("complete") or collection.get("role") != "pilot":
        raise ValueError("Swap audit requires a completed pilot collection")
    if args.burn_in != int(collection["protocol"]["burn_in"]):
        raise ValueError("Swap audit burn-in must match the frozen diagnostic dataset")
    if tuple(args.horizons) != tuple(collection["protocol"]["horizons"]):
        raise ValueError("Swap audit horizons must match the frozen diagnostic dataset")
    specs = load_snapshot_specs(run_dir / "analysis_snapshots")
    boundaries = [spec for spec in specs if spec.reason == "task_boundary"]
    c6 = boundaries[-1]
    if len(collection["datasets"]) != len(boundaries):
        raise ValueError("Collection task count does not match snapshot boundaries")
    tasks = []
    raw_records: list[dict[str, Any]] = []
    raw_values: list[np.ndarray] = []
    for task_index, ci in enumerate(boundaries[:-1]):
        entry = collection["datasets"][task_index]
        if int(entry["task"]["index"]) != task_index:
            raise ValueError("Collection task order changed")
        natural_info = entry["natural"]
        dataset = _load_dataset(audit_dir / natural_info["path"], natural_info["sha256"])
        reward_scale = float(entry["task"]["reward_scale"])
        ci_condition = CONDITIONS[0]
        ci_model = _condition_model(ci_condition, ci, c6, device=args.device)
        ci_output = _evaluate_one_checkpoint(
            ci_model,
            dataset,
            burn_in=args.burn_in,
            horizons=args.horizons,
            batch_size=args.batch_size,
            event_anchor=False,
            reward_scale=reward_scale,
        )
        _append_actor_comparisons(ci_output, ci_output)
        del ci_model

        c6_condition = CONDITIONS[1]
        c6_model = _condition_model(c6_condition, ci, c6, device=args.device)
        c6_output = _evaluate_one_checkpoint(
            c6_model,
            dataset,
            burn_in=args.burn_in,
            horizons=args.horizons,
            batch_size=args.batch_size,
            event_anchor=False,
            reward_scale=reward_scale,
        )
        _append_actor_comparisons(c6_output, ci_output)
        del c6_model

        condition_rows = []

        def record_condition(condition_index: int, condition: SwapCondition, output: dict[str, Any]) -> None:
            summary, raw = _metric_rows(
                c6_output=c6_output,
                condition_output=output,
                episode_ids=dataset["episode_ids"],
                seed=args.bootstrap_seed + task_index * 10_000 + condition_index * 100,
            )
            condition_rows.append(
                {
                    "name": condition.name,
                    "description": condition.description,
                    "world_model_mode": condition.world_model_mode,
                    "world_model_prefixes": list(condition.world_model_prefixes),
                    "actor_mode": condition.actor_mode,
                    "metrics": summary,
                }
            )
            for metric, values in raw.items():
                raw_records.append(
                    {
                        "task_index": task_index,
                        "condition": condition.name,
                        "metric": metric,
                    }
                )
                raw_values.append(values)

        record_condition(0, ci_condition, ci_output)
        record_condition(1, c6_condition, c6_output)
        _discard_comparison_tensors(c6_output)
        for condition_index, condition in enumerate(CONDITIONS[2:], start=2):
            model = _condition_model(condition, ci, c6, device=args.device)
            output = _evaluate_one_checkpoint(
                model,
                dataset,
                burn_in=args.burn_in,
                horizons=args.horizons,
                batch_size=args.batch_size,
                event_anchor=False,
                reward_scale=reward_scale,
            )
            _append_actor_comparisons(output, ci_output)
            record_condition(condition_index, condition, output)
            _discard_comparison_tensors(output)
            del model
            torch = __import__("torch")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        _discard_comparison_tensors(ci_output)
        tasks.append(
            {
                "task_index": task_index,
                "task_name": entry["task"]["name"],
                "baseline_checkpoint": ci.label,
                "comparison_checkpoint": c6.label,
                "natural_dataset": {
                    "path": natural_info["path"],
                    "sha256": natural_info["sha256"],
                    "chunks": natural_info["chunks"],
                },
                "conditions": condition_rows,
            }
        )
        print(f"[swap-audit] task={task_index} complete")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "per_condition_metrics.npz"
    raw_digest = _write_raw_metrics(raw_path, raw_records, raw_values)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "component_swap_restoration_audit",
        "complete": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "pilot",
        "project_git": git_state(ROOT),
        "source": {
            "run_dir": str(run_dir),
            "collection_manifest": {
                "path": str(collection_path),
                "sha256": _sha256(collection_path),
            },
            "comparison_checkpoint": {
                "label": c6.label,
                "name": c6.path.name,
                "sha256": c6.sha256,
            },
        },
        "evaluation": {
            "environment_interaction": False,
            "gradient_updates": False,
            "device": args.device,
            "burn_in": args.burn_in,
            "horizons": list(args.horizons),
            "batch_size": args.batch_size,
            "bootstrap": "1,000 paired episode-cluster resamples",
        },
        "conditions": [
            {
                "name": condition.name,
                "description": condition.description,
                "world_model_mode": condition.world_model_mode,
                "world_model_prefixes": list(condition.world_model_prefixes),
                "actor_mode": condition.actor_mode,
            }
            for condition in CONDITIONS
        ],
        "tasks": tasks,
        "raw_metrics": {"path": raw_path.name, "sha256": raw_digest},
    }
    result_path = output_dir / "swap_results.json"
    _write_json_atomic(result_path, payload)
    _write_sha256_sidecar(result_path)
    report_path = output_dir / "SWAP_REPORT.md"
    _write_text_atomic(report_path, _markdown(payload))
    _write_sha256_sidecar(report_path)
    print(f"[swap-audit] complete output={output_dir}")


def _parse_horizons(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item)
    if not values or values != tuple(sorted(set(values))) or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("horizons must be ascending, unique positive integers")
    return values


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least one")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--burn-in", type=_positive_int, default=16)
    parser.add_argument("--horizons", type=_parse_horizons, default=(1, 2, 4, 8, 16))
    parser.add_argument("--bootstrap-seed", type=int, default=1_010_000)
    args = parser.parse_args()
    run_swap_audit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
