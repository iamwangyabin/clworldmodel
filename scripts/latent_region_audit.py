#!/usr/bin/env python3
"""Test whether Atari tasks occupy distinct frozen-DINO and RSSM regions."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from component_forgetting_audit import (
    _load_dataset,
    _model_bundle,
    _sha256,
    load_snapshot_specs,
)


ROOT = Path(__file__).resolve().parents[1]


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("Run the latent audit in the pinned PyTorch environment") from error
    return torch


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        json.dump(value, temporary, indent=2, ensure_ascii=False)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".npz", dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


class _SupportAccumulator:
    def __init__(self, residuals: dict[str, Any]) -> None:
        self.enabled = False
        self.sums: dict[str, Any] = {}
        self.counts: dict[str, int] = {}
        self.handles = [
            residual.register_forward_pre_hook(self._hook(name))
            for name, residual in residuals.items()
        ]

    def _hook(self, name: str):
        def observe(module: Any, inputs: tuple[Any, ...]) -> None:
            if not self.enabled:
                return
            if len(inputs) != 1:
                raise ValueError("Residual support hook expects one input tensor")
            basis = module.basis_activations(inputs[0]).detach().float()
            flat = basis.reshape(-1, basis.shape[-2], basis.shape[-1])
            contribution = flat.sum(dim=0).cpu()
            self.sums[name] = self.sums.get(name, 0) + contribution
            self.counts[name] = self.counts.get(name, 0) + flat.shape[0]

        return observe

    def means(self) -> dict[str, np.ndarray]:
        return {
            name: (value / self.counts[name]).numpy()
            for name, value in self.sums.items()
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _select_snapshot(specs: list[Any], checkpoint: str) -> Any:
    if checkpoint == "final":
        return next(spec for spec in specs if spec.reason == "final")
    if checkpoint.startswith("boundary:"):
        boundary_number = int(checkpoint.split(":", 1)[1])
        boundaries = [spec for spec in specs if spec.reason == "task_boundary"]
        if not 1 <= boundary_number <= len(boundaries):
            raise ValueError(f"Boundary checkpoint must lie in 1..{len(boundaries)}")
        return boundaries[boundary_number - 1]
    raise ValueError("checkpoint must be 'final' or 'boundary:N'")


def _post_burnin_posterior(
    model: Any,
    actions: Any,
    observations: Any,
    resets: Any,
    *,
    burn_in: int,
    support: _SupportAccumulator,
) -> tuple[Any, Any, Any, Any]:
    torch = _torch()
    embeddings = model.world_model.rssm.embed_observations(observations)
    z, h = model.world_model.rssm.initial_state(actions.shape[1])
    log_probabilities = []
    samples = []
    hiddens = []
    for time_index, (embedding, action, reset) in enumerate(
        zip(embeddings, actions, resets)
    ):
        support.enabled = time_index >= burn_in
        h = model.world_model.rssm.recurrent(
            z * (1.0 - reset).unsqueeze(-1),
            action,
            h * (1.0 - reset),
        )
        log_probability = model.world_model.rssm.representation(embedding, h)
        z = torch.nn.functional.one_hot(
            log_probability.argmax(dim=-1),
            num_classes=log_probability.shape[-1],
        ).to(log_probability.dtype)
        if time_index >= burn_in:
            log_probabilities.append(log_probability)
            samples.append(z)
            hiddens.append(h)
    support.enabled = True
    return (
        embeddings[burn_in:],
        torch.stack(log_probabilities),
        torch.stack(samples),
        torch.stack(hiddens),
    )


def _exercise_post_burnin_heads(
    model: Any,
    posterior_z: Any,
    hiddens: Any,
) -> None:
    wm = model.world_model
    prior_log_probs = wm.rssm.transition(hiddens)
    zhs = wm.zh_transform(posterior_z, hiddens)
    for residual in (wm.reward_residual, wm.continue_residual):
        if residual is not None:
            residual(zhs)
    if wm.feature_predictor_residual is not None:
        feature_state = (
            wm.zh_transform(prior_log_probs.exp(), hiddens)
            if wm.observation_objective == "dinov3_next_feature"
            else zhs
        )
        wm.feature_predictor_residual(feature_state)
    state = model.vendor.zh_to_ac_state(posterior_z, hiddens)
    model.actor_critic.actor(state)
    model.actor_critic.critic(state)


def _extract_task_representations(
    model: Any,
    dataset: dict[str, np.ndarray],
    *,
    burn_in: int,
    batch_size: int,
    max_samples: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    torch = _torch()
    from clworldmodel.continual import named_kan_residuals

    chunk_count, sequence_length = dataset["actions"].shape[:2]
    usable_length = sequence_length - burn_in
    if usable_length < 1:
        raise ValueError("Burn-in must be shorter than each diagnostic chunk")
    total_samples = chunk_count * usable_length
    rng = np.random.default_rng(seed)
    selected = np.sort(
        rng.choice(total_samples, size=min(total_samples, max_samples), replace=False)
    )
    residuals = named_kan_residuals(
        {"world_model": model.world_model, "actor_critic": model.actor_critic}
    )
    support = _SupportAccumulator(residuals)
    collected: dict[str, list[np.ndarray]] = {
        "dinov3_features": [],
        "rssm_posterior_probs": [],
        "rssm_hidden": [],
        "rssm_state": [],
    }
    try:
        with torch.no_grad():
            for start in range(0, chunk_count, batch_size):
                stop = min(start + batch_size, chunk_count)
                actions = torch.from_numpy(dataset["actions"][start:stop]).to(
                    model.device, dtype=torch.float32
                ).swapaxes(0, 1)
                observations = torch.from_numpy(
                    dataset["observations"][start:stop]
                ).to(model.device, dtype=torch.float32).swapaxes(0, 1) / 255.0
                resets = torch.from_numpy(dataset["resets"][start:stop]).to(
                    model.device, dtype=torch.float32
                ).swapaxes(0, 1)
                embeddings, posterior_log, posterior_z, hiddens = (
                    _post_burnin_posterior(
                        model,
                        actions,
                        observations,
                        resets,
                        burn_in=burn_in,
                        support=support,
                    )
                )
                _exercise_post_burnin_heads(model, posterior_z, hiddens)

                posterior_probs = posterior_log.exp().flatten(-2)
                states = torch.cat((posterior_probs, hiddens), dim=-1)
                tensors = {
                    "dinov3_features": embeddings,
                    "rssm_posterior_probs": posterior_probs,
                    "rssm_hidden": hiddens,
                    "rssm_state": states,
                }
                global_start = start * usable_length
                global_stop = stop * usable_length
                local_indices = selected[
                    (selected >= global_start) & (selected < global_stop)
                ] - global_start
                torch_indices = torch.from_numpy(local_indices).to(model.device)
                for name, values in tensors.items():
                    chunk_major = values.swapaxes(0, 1).reshape(-1, values.shape[-1])
                    collected[name].append(
                        chunk_major.index_select(0, torch_indices).float().cpu().numpy()
                    )
    finally:
        support.close()
    arrays = {name: np.concatenate(parts) for name, parts in collected.items()}
    if any(len(values) != len(selected) for values in arrays.values()):
        raise RuntimeError("Latent-region sample selection lost alignment")
    return arrays, support.means()


def _pca_two_dimensional(
    task_arrays: dict[str, np.ndarray],
    *,
    seed: int,
    max_samples_per_task: int,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    names = list(task_arrays)
    values = []
    labels = []
    for task_index, name in enumerate(names):
        task_values = np.asarray(task_arrays[name], dtype=np.float64)
        count = min(len(task_values), max_samples_per_task)
        indices = rng.choice(len(task_values), size=count, replace=False)
        values.append(task_values[indices])
        labels.append(np.full(count, task_index, dtype=np.int64))
    matrix = np.concatenate(values)
    label_values = np.concatenate(labels)
    mean = matrix.mean(0)
    scale = matrix.std(0)
    informative = scale > 1e-8
    standardized = (matrix[:, informative] - mean[informative]) / scale[informative]
    if min(standardized.shape) < 2:
        raise ValueError("PCA projection requires at least two informative dimensions")
    sketch_width = min(16, *standardized.shape)
    random_directions = rng.normal(
        size=(standardized.shape[1], sketch_width)
    )
    sketch = standardized @ random_directions
    sketch = standardized @ (standardized.T @ sketch)
    orthogonal, _ = np.linalg.qr(sketch, mode="reduced")
    compressed = orthogonal.T @ standardized
    _, singular_values, components = np.linalg.svd(compressed, full_matrices=False)
    coordinates = standardized @ components[:2].T
    denominator = np.square(standardized).sum()
    explained = np.square(singular_values[:2]) / max(denominator, 1e-12)
    return coordinates.astype(np.float32), label_values, names, explained.astype(np.float32)


def _plot_projections(
    path: Path,
    projections: dict[str, tuple[np.ndarray, np.ndarray, list[str], np.ndarray]],
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    titles = {
        "dinov3_features": "Frozen DINOv3 spatial features",
        "rssm_posterior_probs": "RSSM posterior probabilities",
        "rssm_hidden": "RSSM deterministic hidden state",
        "rssm_state": "Combined RSSM state",
    }
    for axis, (name, (coordinates, labels, task_names, explained)) in zip(
        axes.flat, projections.items()
    ):
        for task_index, task_name in enumerate(task_names):
            mask = labels == task_index
            axis.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                s=8,
                alpha=0.45,
                label=task_name.replace("ALE/", "").replace("-v5", ""),
            )
        axis.set_title(titles[name])
        axis.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
        axis.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
        axis.grid(alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=min(3, len(labels)))
    figure.suptitle("Task regions at one fixed checkpoint")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _markdown_report(report: dict[str, object]) -> str:
    lines = [
        "# Frozen DINOv3 / RSSM Task-Region Audit",
        "",
        f"Checkpoint: `{report['checkpoint']['name']}`",
        "",
        "| Representation | Nearest centroid | 5-NN | Permutation p | Detected |",
        "|---|---:|---:|---:|:---:|",
    ]
    for name, result in report["representations"].items():
        nearest = result["nearest_centroid"]
        knn = result["knn"]
        lines.append(
            f"| {name} | {nearest['accuracy']:.3f} | {knn['accuracy']:.3f} | "
            f"{nearest['permutation_p_value']:.4f} | "
            f"{'yes' if result['region_separation_detected'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Joint DINO/RSSM evidence: "
            f"**{'yes' if report['joint_region_separation_detected'] else 'no'}**.",
            "",
            "Task labels are used only by this offline audit. The fixed checkpoint "
            "receives no task ID, no evaluation transition enters replay, and no "
            "parameter is updated.",
            "",
            "Above-chance task decodability supports the local-KAN routing premise, "
            "but it does not by itself prove disjoint RBF support or reduced "
            "forgetting. Use the reported support-overlap matrices and continual-return "
            "results together.",
            "",
        ]
    )
    return "\n".join(lines)


def _region_support_alignment(
    region_result: dict[str, object],
    support_results: dict[str, dict[str, object]],
) -> dict[str, object]:
    distances = np.asarray(
        region_result["normalized_centroid_distance"]["matrix"],
        dtype=np.float64,
    )
    pair_indices = np.triu_indices(len(distances), 1)
    distance_values = distances[pair_indices]
    correlations: dict[str, float | None] = {}
    for module_name, support in support_results.items():
        if support["task_names"] != region_result["task_names"]:
            raise ValueError("Region and support task order must match")
        overlap = np.asarray(support["weighted_jaccard_matrix"], dtype=np.float64)
        disjointness = 1.0 - overlap[pair_indices]
        if distance_values.std() <= 1e-12 or disjointness.std() <= 1e-12:
            correlations[module_name] = None
        else:
            correlations[module_name] = float(
                np.corrcoef(distance_values, disjointness)[0, 1]
            )
    finite = [value for value in correlations.values() if value is not None]
    return {
        "region_representation": "rssm_state",
        "support_quantity": "one minus weighted-Jaccard overlap",
        "pairwise_pearson_by_module": correlations,
        "mean_pairwise_pearson": float(np.mean(finite)) if finite else None,
        "interpretation": (
            "A positive value means games farther apart in RSSM state tend to use "
            "more disjoint local RBF support. It is descriptive, not causal."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="final")
    parser.add_argument("--dataset-role", choices=("natural", "event"), default="natural")
    parser.add_argument("--dinov3-model-path", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--burn-in", type=int)
    parser.add_argument("--max-samples-per-task", type=int, default=2_048)
    parser.add_argument("--classifier-samples-per-task", type=int, default=512)
    parser.add_argument("--plot-samples-per-task", type=int, default=256)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260820)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.batch_size < 1 or args.max_samples_per_task < 4:
        raise ValueError("Batch size must be positive and each task needs four samples")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dir = args.run_dir.expanduser().resolve()
    audit_dir = args.audit_dir.expanduser().resolve()
    specs = load_snapshot_specs(run_dir / "analysis_snapshots")
    snapshot = _select_snapshot(specs, args.checkpoint)
    model = _model_bundle(snapshot, args.device, args.dinov3_model_path)
    manifest_path = audit_dir / "collection_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise ValueError("Diagnostic collection manifest is incomplete")
    burn_in = (
        int(manifest["protocol"]["burn_in"])
        if args.burn_in is None
        else args.burn_in
    )

    representations: dict[str, dict[str, np.ndarray]] = {
        "dinov3_features": {},
        "rssm_posterior_probs": {},
        "rssm_hidden": {},
        "rssm_state": {},
    }
    supports_by_module: dict[str, dict[str, np.ndarray]] = {}
    dataset_records = []
    for task_index, entry in enumerate(manifest["datasets"]):
        task_name = str(entry["task"]["name"])
        dataset_info = entry.get(args.dataset_role)
        if dataset_info is None:
            raise ValueError(f"Task {task_name} has no {args.dataset_role} dataset")
        dataset_path = audit_dir / dataset_info["path"]
        dataset = _load_dataset(dataset_path, dataset_info["sha256"])
        task_values, task_supports = _extract_task_representations(
            model,
            dataset,
            burn_in=burn_in,
            batch_size=args.batch_size,
            max_samples=args.max_samples_per_task,
            seed=args.seed + task_index,
        )
        for name, values in task_values.items():
            representations[name][task_name] = values
        for module_name, support in task_supports.items():
            supports_by_module.setdefault(module_name, {})[task_name] = support
        dataset_records.append(
            {
                "task_index": task_index,
                "task_name": task_name,
                "path": dataset_info["path"],
                "sha256": dataset_info["sha256"],
                "samples_used": len(next(iter(task_values.values()))),
            }
        )
        print(f"[latent-region] task={task_index} name={task_name}")

    from clworldmodel.evaluation import analyze_task_regions, task_support_overlap

    representation_results = {
        name: analyze_task_regions(
            task_values,
            seed=args.seed + 10_000 + index,
            max_samples_per_task=args.classifier_samples_per_task,
            permutation_repetitions=args.permutations,
        )
        for index, (name, task_values) in enumerate(representations.items())
    }
    support_results = {
        module_name: task_support_overlap(task_supports)
        for module_name, task_supports in supports_by_module.items()
    }
    region_support_alignment = _region_support_alignment(
        representation_results["rssm_state"],
        support_results,
    )
    projections = {
        name: _pca_two_dimensional(
            task_values,
            seed=args.seed + 20_000 + index,
            max_samples_per_task=args.plot_samples_per_task,
        )
        for index, (name, task_values) in enumerate(representations.items())
    }
    projection_arrays: dict[str, np.ndarray] = {}
    for name, (coordinates, labels, task_names, explained) in projections.items():
        projection_arrays[f"{name}_coordinates"] = coordinates
        projection_arrays[f"{name}_task_indices"] = labels
        projection_arrays[f"{name}_task_names"] = np.asarray(task_names)
        projection_arrays[f"{name}_explained_variance"] = explained
    projection_path = output_dir / "latent_region_projections.npz"
    _write_npz_atomic(projection_path, projection_arrays)
    figure_path = output_dir / "latent_region_pca.png"
    _plot_projections(figure_path, projections)

    joint_detected = bool(
        representation_results["dinov3_features"]["region_separation_detected"]
        and representation_results["rssm_state"]["region_separation_detected"]
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "frozen_dinov3_rssm_task_region_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(run_dir),
        "source_audit": str(audit_dir),
        "checkpoint": {
            "name": snapshot.path.name,
            "path": str(snapshot.path),
            "sha256": snapshot.sha256,
            "epoch": snapshot.epoch,
            "reason": snapshot.reason,
        },
        "collection_manifest_sha256": _sha256(manifest_path),
        "dataset_role": args.dataset_role,
        "collection_policy": manifest["protocol"].get(
            "collection_policy", "checkpoint_actor"
        ),
        "datasets": dataset_records,
        "evaluation": {
            "device": args.device,
            "seed": args.seed,
            "burn_in": burn_in,
            "deterministic_posterior": True,
            "task_labels_available_to_model": False,
            "environment_interactions": 0,
            "gradient_updates": 0,
            "evaluation_transitions_enter_replay": False,
        },
        "representations": representation_results,
        "kan_support_overlap": support_results,
        "region_support_alignment": region_support_alignment,
        "joint_region_separation_detected": joint_detected,
        "claim_scope": (
            "Diagnostic evidence for task-conditioned representation regions and "
            "input-local KAN activation; not causal evidence of reduced forgetting."
        ),
        "artifacts": {
            "pca_data": projection_path.name,
            "pca_figure": figure_path.name,
        },
    }
    report_path = output_dir / "latent_region_report.json"
    _write_json_atomic(report_path, report)
    markdown_path = output_dir / "REPORT.md"
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    print(
        f"[latent-region] complete joint_separation={joint_detected} "
        f"output={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
