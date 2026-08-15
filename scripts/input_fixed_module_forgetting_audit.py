#!/usr/bin/env python3
"""Audit module-output forgetting on fixed old-task activation coordinates.

Unlike an end-to-end checkpoint comparison, every downstream module here is
fed the exact activation produced by the old task-boundary snapshot.  This
keeps encoder/RSSM input drift from being misattributed to a reward, actor, or
critic head.  The audit is offline: it consumes immutable analysis snapshots
and held-out diagnostic chunks only, with no environment interaction or
parameter updates.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from component_audit_metrics import (
    linear_cka,
    mean_and_episode_bootstrap_ci,
    orthogonal_procrustes_residual,
    paired_episode_bootstrap_difference,
)
from component_forgetting_audit import (
    DEFAULT_BOOTSTRAP_REPETITIONS,
    ROOT,
    SnapshotSpec,
    _fixed_returns_torch,
    _load_dataset,
    _model_bundle,
    _sha256,
    _to_time_batch,
    _write_json_atomic,
    _write_metrics_npz,
    _write_sha256_sidecar,
    _write_text_atomic,
    load_snapshot_specs,
)
from git_provenance import git_state


SCHEMA_VERSION = 2
DEFAULT_BURN_IN = 16
DEFAULT_BATCH_SIZE = 16
DEFAULT_BOOTSTRAP_SEED = 20260816


@dataclass(frozen=True)
class MetricSpec:
    """A module metric with its native direction and interpretation."""

    name: str
    group: str
    direction: str
    description: str


METRICS = (
    MetricSpec(
        "encoder.linear_cka",
        "encoder",
        "higher_is_better",
        "Linear CKA between paired image-embedder outputs on the same old observations.",
    ),
    MetricSpec(
        "encoder.procrustes_residual",
        "encoder",
        "lower_is_better",
        "Centered feature residual after best global orthogonal alignment.",
    ),
    MetricSpec(
        "posterior.symmetric_kl",
        "posterior",
        "lower_is_better",
        "Symmetric KL between posterior categoricals on frozen old encoder and hidden inputs.",
    ),
    MetricSpec(
        "rssm.recurrent_normalized_rmse",
        "rssm",
        "lower_is_better",
        "Deterministic recurrent-state RMSE normalized by old hidden-state scale.",
    ),
    MetricSpec(
        "rssm.prior_symmetric_kl",
        "rssm",
        "lower_is_better",
        "Symmetric KL between one-step priors on frozen old transition inputs.",
    ),
    MetricSpec(
        "reward_head.output_symlog_mae",
        "reward_head",
        "lower_is_better",
        "Absolute drift of reward-head symlog output on frozen old model states.",
    ),
    MetricSpec(
        "reward_head.target_symlog_mse",
        "reward_head",
        "lower_is_better",
        "Reward-head symlog MSE against the fixed old-task reward target.",
    ),
    MetricSpec(
        "continue_head.output_probability_mae",
        "continue_head",
        "lower_is_better",
        "Absolute drift of continuation probability on frozen old model states.",
    ),
    MetricSpec(
        "continue_head.target_bce",
        "continue_head",
        "lower_is_better",
        "Continuation BCE against the fixed old-task target; event balance is reported separately.",
    ),
    MetricSpec(
        "actor_head.symmetric_kl",
        "actor_head",
        "lower_is_better",
        "Symmetric action-distribution KL on frozen old actor states.",
    ),
    MetricSpec(
        "actor_head.top1_agreement",
        "actor_head",
        "higher_is_better",
        "Top-1 action agreement on frozen old actor states.",
    ),
    MetricSpec(
        "critic_head.distribution_symmetric_kl",
        "critic_head",
        "lower_is_better",
        "Symmetric KL between critic return distributions on frozen old actor states.",
    ),
    MetricSpec(
        "critic_head.value_reference_mae",
        "critic_head",
        "lower_is_better",
        "Absolute scalar-value drift relative to the old critic on frozen old actor states.",
    ),
    MetricSpec(
        "critic_head.anchored_return_mae",
        "critic_head",
        "lower_is_better",
        "MAE against fixed finite-horizon historical returns from the old diagnostic chunk.",
    ),
)
METRIC_BY_NAME = {metric.name: metric for metric in METRICS}


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - server dependency
        raise RuntimeError(
            "The input-fixed module audit needs the experiment PyTorch environment."
        ) from error
    return torch


def _symmetric_kl_torch(reference_log_probs: Any, comparison_log_probs: Any) -> Any:
    """Return per-time, per-chunk symmetric KL and sum latent factors if present."""

    reference_probs = reference_log_probs.exp()
    comparison_probs = comparison_log_probs.exp()
    divergence = 0.5 * (
        (reference_probs * (reference_log_probs - comparison_log_probs)).sum(dim=-1)
        + (comparison_probs * (comparison_log_probs - reference_log_probs)).sum(dim=-1)
    )
    while divergence.ndim > 2:
        divergence = divergence.sum(dim=-1)
    return divergence


def _chunk_temporal_mean(values: Any, *, start: int) -> np.ndarray:
    """Convert a [time, chunk] tensor to one deterministic scalar per chunk."""

    selected = values[start:]
    if selected.ndim != 2 or selected.shape[0] == 0:
        raise ValueError(f"Expected non-empty [time, chunk] metric tensor, got {tuple(selected.shape)}")
    return selected.mean(dim=0).detach().cpu().numpy().astype(np.float64, copy=False)


def _feature_chunk_metrics(reference: Any, comparison: Any, *, start: int) -> dict[str, np.ndarray]:
    """Compute encoder geometry measurements independently for each audit chunk."""

    reference_chunks = reference[start:].permute(1, 0, 2).detach().cpu().numpy()
    comparison_chunks = comparison[start:].permute(1, 0, 2).detach().cpu().numpy()
    if reference_chunks.shape != comparison_chunks.shape:
        raise ValueError("Encoder feature traces must have identical shapes")
    cka_values = []
    procrustes_values = []
    for reference_chunk, comparison_chunk in zip(reference_chunks, comparison_chunks):
        cka_values.append(linear_cka(reference_chunk, comparison_chunk))
        procrustes_values.append(orthogonal_procrustes_residual(reference_chunk, comparison_chunk))
    return {
        "encoder.linear_cka": np.asarray(cka_values, dtype=np.float64),
        "encoder.procrustes_residual": np.asarray(procrustes_values, dtype=np.float64),
    }


def _reference_trace(model: Any, batch: Mapping[str, Any]) -> dict[str, Any]:
    """Run the old boundary snapshot once and retain its old-coordinate trace."""

    actions = batch["actions"]
    observations = batch["observations"]
    resets = batch["resets"]
    chunk_count = actions.shape[1]
    z0, h0 = model.world_model.rssm.initial_state(chunk_count)
    posterior_log_probs, posterior_z, hiddens = model.world_model.rssm(
        z0, actions, h0, observations, resets, stochastic=False
    )
    time_steps, _, channels, height, width = observations.shape
    encoder_features = model.world_model.rssm.image_embedder(
        observations.reshape(-1, channels, height, width)
    ).view(time_steps, chunk_count, -1)
    zhs = model.world_model.zh_transform(posterior_z, hiddens)
    actor_states = model.vendor.zh_to_ac_state(posterior_z, hiddens)
    return {
        "actions": actions,
        "observations": observations,
        "continues": batch["continues"],
        "encoder_features": encoder_features,
        "posterior_log_probs": posterior_log_probs,
        "posterior_z": posterior_z,
        "hiddens": hiddens,
        "zhs": zhs,
        "actor_states": actor_states,
        "reward_targets": batch["rewards"],
        "fixed_returns": batch["fixed_returns"],
        "resets": resets,
    }


def _old_coordinate_metrics(
    reference_model: Any,
    comparison_model: Any,
    reference: Mapping[str, Any],
    *,
    burn_in: int,
) -> dict[str, np.ndarray]:
    """Evaluate one later checkpoint on the exact old task-boundary inputs."""

    torch = _torch()
    actions = reference["actions"]
    posterior_z = reference["posterior_z"]
    hiddens = reference["hiddens"]
    resets = reference["resets"]
    zhs = reference["zhs"]
    actor_states = reference["actor_states"]
    rewards = reference["reward_targets"]
    continues = reference["continues"]
    fixed_returns = reference["fixed_returns"]
    time_steps, chunk_count = actions.shape[:2]
    if time_steps <= burn_in:
        raise ValueError("Burn-in must leave at least one evaluated timestep")

    channels, height, width = 3, 64, 64
    observations = reference["observations"]
    comparison_encoder_features = comparison_model.world_model.rssm.image_embedder(
        observations.reshape(-1, channels, height, width)
    ).view(time_steps, chunk_count, -1)
    metrics = _feature_chunk_metrics(
        reference["encoder_features"], comparison_encoder_features, start=burn_in
    )

    comparison_posterior = comparison_model.world_model.rssm.representation(
        reference["encoder_features"], hiddens
    )
    metrics["posterior.symmetric_kl"] = _chunk_temporal_mean(
        _symmetric_kl_torch(reference["posterior_log_probs"], comparison_posterior), start=burn_in
    )

    # Reconstruct the deterministic transition at time t from the immutable
    # old-coordinate state at t-1.  This isolates recurrent/transition weights
    # from encoder and posterior changes.
    if time_steps <= burn_in + 1:
        raise ValueError("Transition audit needs burn-in plus at least two timesteps")
    transition_resets = resets[1:]
    transition_z = posterior_z[:-1] * (1 - transition_resets).unsqueeze(-1)
    transition_h = hiddens[:-1] * (1 - transition_resets)
    comparison_hiddens = comparison_model.world_model.rssm.recurrent(
        transition_z, actions[1:], transition_h
    )
    reference_hiddens = hiddens[1:]
    comparison_priors = comparison_model.world_model.rssm.transition(comparison_hiddens)
    reference_priors = reference_model.world_model.rssm.transition(reference_hiddens)
    hidden_difference = (comparison_hiddens - reference_hiddens).square().mean(dim=-1).sqrt()
    hidden_scale = (
        (reference_hiddens - reference_hiddens.mean(dim=(0, 2), keepdim=True))
        .square()
        .mean(dim=(0, 2))
        .sqrt()
        .clamp_min(1e-8)
    )
    metrics["rssm.recurrent_normalized_rmse"] = _chunk_temporal_mean(
        hidden_difference / hidden_scale.unsqueeze(0), start=burn_in - 1
    )
    metrics["rssm.prior_symmetric_kl"] = _chunk_temporal_mean(
        _symmetric_kl_torch(reference_priors, comparison_priors), start=burn_in - 1
    )

    reference_reward = reference_model.world_model.reward_fc(zhs)
    comparison_reward = comparison_model.world_model.reward_fc(zhs)
    metrics["reward_head.output_symlog_mae"] = _chunk_temporal_mean(
        (comparison_reward - reference_reward).abs().squeeze(-1), start=burn_in
    )
    metrics["reward_head.target_symlog_mse"] = _chunk_temporal_mean(
        (comparison_reward - comparison_model.vendor.symlog(rewards)).square().squeeze(-1),
        start=burn_in,
    )

    reference_continue = reference_model.world_model.continue_fc(zhs)
    comparison_continue = comparison_model.world_model.continue_fc(zhs)
    metrics["continue_head.output_probability_mae"] = _chunk_temporal_mean(
        (comparison_continue - reference_continue).abs().squeeze(-1), start=burn_in
    )
    metrics["continue_head.target_bce"] = _chunk_temporal_mean(
        torch.nn.functional.binary_cross_entropy(comparison_continue, continues, reduction="none").squeeze(-1),
        start=burn_in,
    )

    reference_actor = reference_model.actor_critic.actor(actor_states)
    comparison_actor = comparison_model.actor_critic.actor(actor_states)
    metrics["actor_head.symmetric_kl"] = _chunk_temporal_mean(
        _symmetric_kl_torch(reference_actor, comparison_actor), start=burn_in
    )
    metrics["actor_head.top1_agreement"] = _chunk_temporal_mean(
        (reference_actor.argmax(dim=-1) == comparison_actor.argmax(dim=-1)).float(),
        start=burn_in,
    )

    reference_critic = reference_model.actor_critic.critic(actor_states)
    comparison_critic = comparison_model.actor_critic.critic(actor_states)
    reference_value = reference_model.vendor.symexp(
        reference_critic.exp() @ reference_model.actor_critic.symlog_bins
    )
    comparison_value = comparison_model.vendor.symexp(
        comparison_critic.exp() @ comparison_model.actor_critic.symlog_bins
    )
    metrics["critic_head.distribution_symmetric_kl"] = _chunk_temporal_mean(
        _symmetric_kl_torch(reference_critic, comparison_critic), start=burn_in
    )
    metrics["critic_head.value_reference_mae"] = _chunk_temporal_mean(
        (comparison_value - reference_value).abs().squeeze(-1), start=burn_in
    )
    metrics["critic_head.anchored_return_mae"] = _chunk_temporal_mean(
        (comparison_value - fixed_returns).abs().squeeze(-1), start=burn_in
    )
    return metrics


def _batch_from_dataset(
    dataset: Mapping[str, np.ndarray], start: int, stop: int, device: Any
) -> dict[str, Any]:
    """Create a typed time-major batch without mutating the frozen diagnostic set."""

    torch = _torch()
    actions = _to_time_batch(torch, dataset["actions"][start:stop], device, dtype=torch.float32)
    observations = (
        _to_time_batch(torch, dataset["observations"][start:stop], device, dtype=torch.float32)
        / 255.0
    )
    rewards = _to_time_batch(
        torch, dataset["scaled_rewards"][start:stop], device, dtype=torch.float32
    )
    continues = _to_time_batch(
        torch, dataset["continues"][start:stop], device, dtype=torch.float32
    )
    resets = _to_time_batch(torch, dataset["resets"][start:stop], device, dtype=torch.float32)
    return {
        "actions": actions,
        "observations": observations,
        "rewards": rewards,
        "continues": continues,
        "resets": resets,
        "fixed_returns": _fixed_returns_torch(rewards, continues, discount=0.997),
    }


def _validate_metric_bundle(values: Mapping[str, np.ndarray], expected_chunks: int) -> None:
    if set(values) != set(METRIC_BY_NAME):
        missing = sorted(set(METRIC_BY_NAME) - set(values))
        extra = sorted(set(values) - set(METRIC_BY_NAME))
        raise ValueError(f"Unexpected metric bundle; missing={missing}, extra={extra}")
    for metric, array in values.items():
        if array.shape != (expected_chunks,) or not np.isfinite(array).all():
            raise ValueError(f"Malformed {metric} values: shape={array.shape}")


def _validate_model_interface(reference_model: Any, comparison_model: Any) -> None:
    """Reject a future architecture where old-coordinate head inputs are ambiguous."""

    reference_world_model = reference_model.world_model
    comparison_world_model = comparison_model.world_model
    if (
        reference_world_model.ls != comparison_world_model.ls
        or reference_world_model.h_dim != comparison_world_model.h_dim
        or reference_world_model.a_dim != comparison_world_model.a_dim
    ):
        raise ValueError("Snapshots have incompatible RSSM/actor interfaces")
    if (
        reference_world_model.zh_transform.linear is not None
        or comparison_world_model.zh_transform.linear is not None
    ):
        raise ValueError(
            "This audit requires the pinned identity zh_transform; add an explicit "
            "old-coordinate projection audit before using a learned zh transform."
        )


def _evaluate_pair(
    reference_model: Any,
    comparison_model: Any,
    dataset: Mapping[str, np.ndarray],
    *,
    burn_in: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Evaluate one C_i/C_j pair and return exactly one value per frozen chunk."""

    torch = _torch()
    _validate_model_interface(reference_model, comparison_model)
    chunk_count = int(dataset["actions"].shape[0])
    all_values: dict[str, list[np.ndarray]] = {metric.name: [] for metric in METRICS}
    with torch.no_grad():
        for start in range(0, chunk_count, batch_size):
            stop = min(start + batch_size, chunk_count)
            batch = _batch_from_dataset(dataset, start, stop, reference_model.device)
            reference = _reference_trace(reference_model, batch)
            metrics = _old_coordinate_metrics(
                reference_model, comparison_model, reference, burn_in=burn_in
            )
            for metric, values in metrics.items():
                all_values[metric].append(values)
    output = {metric: np.concatenate(values) for metric, values in all_values.items()}
    _validate_metric_bundle(output, chunk_count)
    return output


def _assert_baseline_invariance(values: Mapping[str, np.ndarray]) -> None:
    """Fail closed if a C_i module does not match itself under identical inputs."""

    if not np.allclose(values["encoder.linear_cka"], 1.0, atol=1e-10, rtol=1e-10):
        raise RuntimeError("Baseline encoder CKA is not exactly one")
    if not np.allclose(values["actor_head.top1_agreement"], 1.0, atol=0, rtol=0):
        raise RuntimeError("Baseline actor agreement is not exactly one")
    for metric, metric_values in values.items():
        if metric in {"encoder.linear_cka", "actor_head.top1_agreement", "reward_head.target_symlog_mse", "continue_head.target_bce", "critic_head.anchored_return_mae"}:
            continue
        if not np.allclose(metric_values, 0.0, atol=1e-8, rtol=1e-8):
            raise RuntimeError(f"Baseline input-fixed drift is not zero for {metric}")


def _summary_row(
    *,
    task_index: int,
    task_name: str,
    reference_checkpoint: SnapshotSpec,
    comparison_checkpoint: SnapshotSpec,
    metric: MetricSpec,
    reference_values: np.ndarray,
    comparison_values: np.ndarray,
    episode_ids: np.ndarray,
    bootstrap_seed: int,
    repetitions: int,
) -> dict[str, Any]:
    paired = paired_episode_bootstrap_difference(
        reference_values,
        comparison_values,
        episode_ids,
        seed=bootstrap_seed,
        repetitions=repetitions,
    )
    current_summary = mean_and_episode_bootstrap_ci(
        comparison_values,
        episode_ids,
        seed=bootstrap_seed + 1,
        repetitions=repetitions,
    )
    raw_delta = float(paired["comparison_minus_baseline"])
    forgetting = raw_delta if metric.direction == "lower_is_better" else -raw_delta
    return {
        "task_index": task_index,
        "task_name": task_name,
        "reference_checkpoint": reference_checkpoint.label,
        "reference_epoch": reference_checkpoint.epoch,
        "reference_sha256": reference_checkpoint.sha256,
        "comparison_checkpoint": comparison_checkpoint.label,
        "comparison_epoch": comparison_checkpoint.epoch,
        "comparison_sha256": comparison_checkpoint.sha256,
        "metric": metric.name,
        "group": metric.group,
        "direction": metric.direction,
        "description": metric.description,
        "reference_mean": float(paired["baseline_mean"]),
        "comparison_mean": float(paired["comparison_mean"]),
        "comparison_minus_reference": raw_delta,
        "comparison_minus_reference_ci_low": float(paired["ci_low"]),
        "comparison_minus_reference_ci_high": float(paired["ci_high"]),
        "boundary_relative_forgetting": float(forgetting),
        "comparison_ci_low": float(current_summary["ci_low"]),
        "comparison_ci_high": float(current_summary["ci_high"]),
        "n_chunks": int(paired["n_chunks"]),
        "n_episodes": int(paired["n_episodes"]),
    }


def _render_report(payload: Mapping[str, Any]) -> str:
    """Render a compact, human-readable profile while leaving raw arrays in NPZ."""

    rows = payload["summary_rows"]
    lines = [
        "# Input-Fixed Module Forgetting Audit (V2)",
        "",
        "This is an offline, single-seed pilot analysis. Every downstream module is evaluated on the exact old boundary activation trace, so a head is not blamed for a changed encoder or RSSM input.",
        "",
        "`Cfinal_e540` is excluded: it follows one additional Task 1 update. The original training run had a dirty launch worktree, so this remains hypothesis-generating rather than an official result.",
        "",
        "## Reading This Report",
        "",
        "- Positive `forgetting` means lower retention in that metric's native direction.",
        "- Encoder CKA and Procrustes use the direct image-embedder output, not the combined RSSM `zh` state.",
        "- Reward/continue/actor/critic rows are head-specific because all receive the frozen `C_i` input state.",
        "- Decoder is intentionally absent from headline metrics. It is not part of the task-specific control path being quantified here.",
        "",
    ]
    for task in payload["tasks"]:
        task_rows = [row for row in rows if row["task_index"] == task["task_index"]]
        lines.extend([f"## {task['task_name']}", ""])
        lines.extend(
            [
                "| Later checkpoint | Encoder CKA loss | Posterior KL | Recurrent nRMSE | Prior KL | Actor KL | Actor action loss | Critic KL |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        by_checkpoint_metric = {
            (row["comparison_checkpoint"], row["metric"]): row for row in task_rows
        }
        checkpoint_labels = task["comparison_checkpoints"]
        for label in checkpoint_labels:
            def value(metric_name: str) -> float:
                return float(by_checkpoint_metric[(label, metric_name)]["boundary_relative_forgetting"])

            lines.append(
                "| {label} | {encoder:.4g} | {posterior:.4g} | {recurrent:.4g} | {prior:.4g} | {actor_kl:.4g} | {actor_loss:.4g} | {critic:.4g} |".format(
                    label=label,
                    encoder=value("encoder.linear_cka"),
                    posterior=value("posterior.symmetric_kl"),
                    recurrent=value("rssm.recurrent_normalized_rmse"),
                    prior=value("rssm.prior_symmetric_kl"),
                    actor_kl=value("actor_head.symmetric_kl"),
                    actor_loss=value("actor_head.top1_agreement"),
                    critic=value("critic_head.distribution_symmetric_kl"),
                )
            )
        lines.extend(
            [
                "",
                "| Later checkpoint | Reward output drift | Reward target MSE delta | Continue output drift | Continue BCE delta | Critic anchored-return MAE delta |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for label in checkpoint_labels:
            def value(metric_name: str) -> float:
                return float(by_checkpoint_metric[(label, metric_name)]["boundary_relative_forgetting"])

            lines.append(
                "| {label} | {reward_drift:.4g} | {reward_error:.4g} | {continue_drift:.4g} | {continue_error:.4g} | {critic_error:.4g} |".format(
                    label=label,
                    reward_drift=value("reward_head.output_symlog_mae"),
                    reward_error=value("reward_head.target_symlog_mse"),
                    continue_drift=value("continue_head.output_probability_mae"),
                    continue_error=value("continue_head.target_bce"),
                    critic_error=value("critic_head.anchored_return_mae"),
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Limits",
            "",
            "- These are descriptive module-output retention measurements, not causal return-restoration claims.",
            "- Natural chunks are not terminal-balanced. Continue BCE measures output drift on the natural distribution, not terminal-event discrimination.",
            "- Raw metrics stay in their native units. Do not rank pixel MSE, categorical KL, and action KL by their numerical magnitude alone.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_full_checksum_manifest(output_dir: Path) -> str:
    entries = []
    for path in sorted(path for path in output_dir.rglob("*") if path.is_file() and path.name != "SHA256SUMS"):
        entries.append(f"{_sha256(path)}  {path.relative_to(output_dir)}")
    text = "\n".join(entries) + "\n"
    _write_text_atomic(output_dir / "SHA256SUMS", text)
    return _sha256(output_dir / "SHA256SUMS")


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
    parser.add_argument(
        "--include-final",
        action="store_true",
        help="Include Cfinal_e540, which is excluded from continual-learning headline claims by default.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.burn_in < 1:
        raise ValueError("burn-in must be positive")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.bootstrap_repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    run_dir = args.run_dir.resolve()
    audit_dir = args.audit_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit output: {output_dir}")
    collection_path = audit_dir / "collection_manifest.json"
    if not collection_path.is_file():
        raise FileNotFoundError(f"Missing collection manifest: {collection_path}")
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    if not collection.get("complete") or collection.get("artifact_kind") != "component_forgetting_audit_collection":
        raise ValueError("Audit collection is incomplete or has the wrong artifact kind")

    all_specs = load_snapshot_specs(run_dir / "analysis_snapshots")
    boundary_specs = [spec for spec in all_specs if spec.reason == "task_boundary"]
    if args.include_final:
        evaluation_specs = all_specs
    else:
        evaluation_specs = boundary_specs
    if len(collection.get("datasets", [])) != len(boundary_specs):
        raise ValueError("Collection datasets do not match the number of task-boundary snapshots")

    source_script = Path(__file__).resolve()
    protocol_path = ROOT / "docs" / "protocols" / "module_forgetting_audit_v2.md"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "input_fixed_module_forgetting_audit",
        "complete": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "pilot",
        "project_git": git_state(ROOT),
        "source_script": {"path": str(source_script), "sha256": _sha256(source_script)},
        "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
        "source_run": str(run_dir),
        "collection_manifest": {"path": str(collection_path), "sha256": _sha256(collection_path)},
        "input_contract": "Every downstream module receives the C_i old-coordinate activation trace; no decoder metric is headline.",
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
        expected_snapshot = dataset_entry["snapshot"]
        if expected_snapshot["sha256"] != reference_spec.sha256:
            raise ValueError(f"Dataset task {task_index} does not match its boundary snapshot")
        natural = dataset_entry["natural"]
        dataset = _load_dataset(audit_dir / natural["path"], natural["sha256"])
        if int(dataset["actions"].shape[1]) <= args.burn_in + 1:
            raise ValueError("Diagnostic chunk is too short for the requested input-fixed audit")
        reference_model = _model_bundle(reference_spec, args.device)
        comparison_specs = [spec for spec in evaluation_specs if spec.epoch >= reference_spec.epoch]
        if not comparison_specs or comparison_specs[0].label != reference_spec.label:
            raise ValueError("Each task must begin its audit at its own boundary snapshot")
        baseline_values: dict[str, np.ndarray] | None = None
        task_rows.append(
            {
                "task_index": task_index,
                "task_name": task["name"],
                "reference_checkpoint": reference_spec.label,
                "comparison_checkpoints": [spec.label for spec in comparison_specs],
                "dataset": {"path": natural["path"], "sha256": natural["sha256"]},
            }
        )
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
                raise RuntimeError("Reference metrics must be evaluated before comparison metrics")
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

    raw_metrics_path = output_dir / "per_chunk_module_metrics.npz"
    _write_metrics_npz(raw_metrics_path, records, arrays)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "input_fixed_module_forgetting_results",
        "complete": True,
        "tasks": task_rows,
        "summary_rows": summary_rows,
        "metric_contract": [metric.__dict__ for metric in METRICS],
        "interpretation": {
            "primary_question": "How much does each module output drift on fixed old-task inputs at every later checkpoint?",
            "decoder": "Not a headline metric. It is optional only as a frozen downstream feature probe.",
            "actor": "Actor-head rows hold the old actor state fixed, so they quantify policy-head drift rather than end-to-end latent-plus-policy drift.",
            "reward": "Reward-head rows hold the old model state fixed, so they quantify reward-readout drift rather than an end-to-end world-model path.",
        },
    }
    results_path = output_dir / "results.json"
    _write_json_atomic(results_path, payload)
    report_path = output_dir / "MODULE_FORGETTING_REPORT.md"
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
    print(f"[input-fixed-module-audit] complete output={output_dir}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
