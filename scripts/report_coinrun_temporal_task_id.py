#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read-only raw-prediction audit and paired seed-cluster uncertainty report."""
import argparse
import json
from pathlib import Path

import numpy as np

from artifact_io import sha256_file, write_json_atomic


def capped_correct(predicted, labels, valid):
    """[S,N,T] predictions -> correctness at min(budget, last real frame)."""
    if predicted.ndim != 3 or predicted.shape[1:] != valid.shape:
        raise ValueError("Expected predicted [S,N,T] and validity [N,T]")
    if not valid[:, 0].all() or (valid[:, 1:] & ~valid[:, :-1]).any():
        raise ValueError("Expected nonempty contiguous observation prefixes")
    stop = valid.sum(1) - 1
    take = np.minimum(np.arange(valid.shape[1])[None, :], stop[:, None])
    return (np.take_along_axis(predicted, take[None], axis=2) == labels[None, :, None]).astype(float)


def cluster_delta(candidate, reference, group_ids, *, seed=20260909, draws=4000):
    """Paired episode accuracy differences; resample paired environment seeds.

    This conditions on the fitted routers and WM checkpoint. Router-seed means
    are not treated as additional independent test observations.
    """
    candidate, reference = np.asarray(candidate), np.asarray(reference)
    if candidate.shape != reference.shape or candidate.shape != np.shape(group_ids):
        raise ValueError("Candidate, reference and group IDs must be aligned vectors")
    _, groups = np.unique(group_ids, return_inverse=True)
    counts = np.bincount(groups)
    if len(counts) < 2 or not np.all(counts == counts[0]):
        raise ValueError("Require at least two equally sized paired task/seed groups")
    delta = np.bincount(groups, weights=candidate - reference) / counts
    rng = np.random.default_rng(seed)
    bootstrap = delta[rng.integers(len(delta), size=(draws, len(delta)))].mean(1)
    return {"paired_delta": float(delta.mean()), "ci95": np.quantile(bootstrap, [.025, .975]).tolist(),
            "seed_group_count": len(counts), "bootstrap_draws": draws, "bootstrap_seed": seed}


def audit(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text())
    result = json.loads((directory / "results.json").read_text())
    if not result["complete"] or manifest["frozen_state_sha256_before"] != result["frozen_state_sha256_after"]:
        raise ValueError("Require a completed run and unchanged WM/Actor states")
    path = directory / "test_predictions.npz"
    if sha256_file(path) != path.with_suffix(".npz.sha256").read_text().split()[0]:
        raise ValueError("Raw prediction checksum mismatch")
    data = np.load(path)
    valid, labels = data["valid"], data["labels"]
    pred = {name: data[name][None] for name in ("first_frame_mse", "mean_prefix_mse")}
    for name in ("pointwise_mlp", "temporal_gru"):
        keys = sorted(k for k in data.files if k.startswith(name + "_") and k.endswith("_logits"))
        pred[name] = np.stack([data[k].argmax(-1) for k in keys])
        if len(keys) != manifest["resolved_config"]["router_seed_count"]:
            raise ValueError("Missing predeclared router training seed")
    pred["learned_first_frame_mlp"] = np.broadcast_to(pred["pointwise_mlp"][:, :, :1], pred["pointwise_mlp"].shape)
    if not (pred["temporal_gru"][:, :, 0] == data["first_frame_mse"][:, 0]).all():
        raise ValueError("GRU did not preserve initial routing")
    correct = {k: capped_correct(v, labels, valid) for k, v in pred.items()}
    for name, values in correct.items():
        for t in range(valid.shape[1]):
            recorded = [r["accuracy"] for r in result["tables"] if r["router"] == name and
                        r["observed_frame_budget"] == t + 1 and r["population"] == "all_episodes_up_to_frames"]
            np.testing.assert_allclose(sorted(recorded), sorted(values[:, :, t].mean(1)), rtol=0, atol=1e-12)
    rows = json.loads((directory / "test/episodes.json").read_text())["episodes"]
    groups = np.array([r["environment_seed"] for r in rows])
    np.testing.assert_array_equal(labels, [r["task_index_for_audit_only"] for r in rows])
    comparisons = []
    for frames in (1, 2, 5, 9, 17):
        if frames > valid.shape[1]: continue
        for reference in ("first_frame_mse", "mean_prefix_mse", "pointwise_mlp"):
            comparisons.append({"frame_budget": frames, "candidate": "temporal_gru", "reference": reference,
                                **cluster_delta(correct["temporal_gru"][:, :, frames - 1].mean(0),
                                                correct[reference][:, :, frames - 1].mean(0), groups)})
    summary = {"raw_prediction_audit_passed": True, "all_cohort_tables_recomputed": True,
               "exact_first_frame_preserved": True, "test_episode_count": len(labels),
               "world_model_seed_count": 1, "router_seed_count": len(pred["temporal_gru"]),
               "prediction_sha256": sha256_file(path), "paired_comparisons": comparisons,
               "uncertainty_scope": "environment-seed cluster bootstrap, conditional on this WM and fitted routers; no multiple-comparison adjustment",
               "accuracy_seed_ranges": {k: {"mean": v.mean((0, 1)).tolist(), "min": v.mean(1).min(0).tolist(),
                                           "max": v.mean(1).max(0).tolist()} for k, v in correct.items()}}
    return summary


def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"first_frame_mse": "Original first frame", "mean_prefix_mse": "Prefix-mean MSE",
              "learned_first_frame_mlp": "Trained head, first frame only",
              "pointwise_mlp": "Current-error MLP", "temporal_gru": "Temporal GRU"}
    fig, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
    for name, label in labels.items():
        row = summary["accuracy_seed_ranges"][name]
        x = np.arange(1, len(row["mean"]) + 1)
        line, = ax.plot(x, 100 * np.array(row["mean"]), label=label,
                        linewidth=2.3 if name == "temporal_gru" else 1.5,
                        linestyle="--" if "first_frame" in name else "-")
        if name in ("pointwise_mlp", "temporal_gru"):
            ax.fill_between(x, 100 * np.array(row["min"]), 100 * np.array(row["max"]), color=line.get_color(), alpha=.12)
    ax.set(xlabel="Observation budget (frames; capped at episode end)", ylabel="Task-ID accuracy (%)",
           title=f"CoinRun: {summary['test_episode_count']:,} held-out episode prefixes", xticks=[1, 2, 5, 9, 13, 17])
    ax.grid(alpha=.2); ax.legend(fontsize=9, loc="best")
    fig.suptitle("One frozen WM checkpoint | 3 router seeds | shading: seed range, not CI", fontsize=10)
    fig.savefig(path, dpi=180); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    directory = args.run_directory.expanduser().resolve()
    summary = audit(directory)
    write_json_atomic(directory / "prediction_audit.json", summary)
    if args.plot: plot(summary, directory / "task_id_accuracy.png")
    print(json.dumps(summary, indent=2))
