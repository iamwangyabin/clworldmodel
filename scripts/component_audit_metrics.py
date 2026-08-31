"""Numerical helpers for the checkpoint-differencing audit.

This module deliberately has no dependency on the vendored trainer.  Keeping
the summary math separate makes the most important metric definitions easy to
unit test on CPU-only machines.
"""

from __future__ import annotations

import math

import numpy as np


def linear_cka(features_x: np.ndarray, features_y: np.ndarray) -> float:
    """Return linear CKA for paired feature matrices without forming an N x N Gram matrix."""
    x = np.asarray(features_x, dtype=np.float64)
    y = np.asarray(features_y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("linear CKA expects two rank-2 feature matrices")
    if x.shape[0] != y.shape[0]:
        raise ValueError("linear CKA requires paired rows")
    if x.shape[0] < 2:
        raise ValueError("linear CKA requires at least two samples")

    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    # Use the smaller of the sample-space and feature-space Gram matrices.
    # Audit chunks contain far fewer timesteps than encoder channels, whereas
    # the global statistic often has the opposite shape.
    if x.shape[0] <= x.shape[1]:
        x_gram = x @ x.T
        y_gram = y @ y.T
        cross_sq = float((x_gram * y_gram).sum())
        x_sq = float(np.square(x_gram).sum())
        y_sq = float(np.square(y_gram).sum())
    else:
        cross_sq = float(np.square(x.T @ y).sum())
        x_sq = float(np.square(x.T @ x).sum())
        y_sq = float(np.square(y.T @ y).sum())
    denominator = math.sqrt(x_sq * y_sq)
    if denominator == 0:
        return 0.0
    return cross_sq / denominator


def orthogonal_procrustes_residual(
    reference_features: np.ndarray, comparison_features: np.ndarray
) -> float:
    """Return the centered relative residual after the best orthogonal alignment.

    This complements CKA for encoder features.  CKA asks whether pairwise
    geometry is retained; this statistic asks how far the comparison features
    remain after removing a global rotation/reflection and scale.  It is zero
    for identical features and is deliberately not interpreted as a universal
    cross-module score.
    """

    reference = np.asarray(reference_features, dtype=np.float64)
    comparison = np.asarray(comparison_features, dtype=np.float64)
    if reference.ndim != 2 or comparison.ndim != 2:
        raise ValueError("Procrustes residual expects two rank-2 feature matrices")
    if reference.shape != comparison.shape:
        raise ValueError("Procrustes residual requires paired feature matrices of equal shape")
    if reference.shape[0] < 2:
        raise ValueError("Procrustes residual requires at least two paired samples")

    reference = reference - reference.mean(axis=0, keepdims=True)
    comparison = comparison - comparison.mean(axis=0, keepdims=True)
    reference_norm = float(np.linalg.norm(reference))
    comparison_norm = float(np.linalg.norm(comparison))
    if reference_norm == 0 or comparison_norm == 0:
        return 0.0 if np.array_equal(reference, comparison) else float("inf")

    reference /= reference_norm
    comparison /= comparison_norm

    # Features have many more channels than paired examples in the audit.  The
    # non-zero singular values of comparison.T @ reference can be obtained
    # from compact left singular factors, avoiding a costly channel-by-channel
    # SVD for every diagnostic chunk.
    def compact_left_svd(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        eigenvalues, left_vectors = np.linalg.eigh(values @ values.T)
        keep = eigenvalues > np.finfo(np.float64).eps
        return left_vectors[:, keep], np.sqrt(eigenvalues[keep])

    reference_left, reference_singular = compact_left_svd(reference)
    comparison_left, comparison_singular = compact_left_svd(comparison)
    middle = (
        comparison_singular[:, None]
        * (comparison_left.T @ reference_left)
        * reference_singular[None, :]
    )
    nuclear_norm = float(np.linalg.svd(middle, compute_uv=False).sum())
    squared_residual = max(0.0, 2.0 - 2.0 * nuclear_norm)
    return float(np.sqrt(squared_residual))


def normalized_rmse(reference: np.ndarray, comparison: np.ndarray, *, epsilon: float = 1e-12) -> float:
    """Return RMSE normalized by the centered RMS scale of the reference tensor."""

    base = np.asarray(reference, dtype=np.float64)
    current = np.asarray(comparison, dtype=np.float64)
    if base.shape != current.shape:
        raise ValueError("normalized RMSE requires tensors with identical shapes")
    if base.size == 0:
        raise ValueError("normalized RMSE requires at least one value")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    rmse = float(np.sqrt(np.mean(np.square(current - base))))
    scale = float(np.sqrt(np.mean(np.square(base - base.mean()))))
    return rmse / max(scale, epsilon)


def relative_rms_perturbation(
    reference: np.ndarray, comparison: np.ndarray, *, epsilon: float = 1e-12
) -> float:
    """Return direct output perturbation relative to the old output RMS.

    Unlike ``normalized_rmse``, this deliberately does not center, rotate, or
    rescale either feature tensor. It is intended for an interface-compatibility
    question: how large is the new output's raw perturbation in the old output
    coordinate system?
    """

    base = np.asarray(reference, dtype=np.float64)
    current = np.asarray(comparison, dtype=np.float64)
    if base.shape != current.shape:
        raise ValueError("relative RMS perturbation requires tensors with identical shapes")
    if base.size == 0:
        raise ValueError("relative RMS perturbation requires at least one value")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    perturbation_rms = float(np.sqrt(np.mean(np.square(current - base))))
    reference_rms = float(np.sqrt(np.mean(np.square(base))))
    return perturbation_rms / max(reference_rms, epsilon)


def symmetric_kl_from_log_probs(
    log_probs_p: np.ndarray, log_probs_q: np.ndarray
) -> np.ndarray:
    """Compute 0.5 * (KL(p || q) + KL(q || p)) along the final action axis."""
    p_log = np.asarray(log_probs_p, dtype=np.float64)
    q_log = np.asarray(log_probs_q, dtype=np.float64)
    if p_log.shape != q_log.shape:
        raise ValueError("paired policies must have identical shapes")
    p = np.exp(p_log)
    q = np.exp(q_log)
    return 0.5 * ((p * (p_log - q_log)).sum(axis=-1) + (q * (q_log - p_log)).sum(axis=-1))


def discounted_returns(
    rewards: np.ndarray, continues: np.ndarray, discount: float
) -> np.ndarray:
    """Compute fixed, finite-chunk return targets along the penultimate time axis."""
    if not 0 <= discount <= 1:
        raise ValueError("discount must be in [0, 1]")
    rews = np.asarray(rewards, dtype=np.float64)
    conts = np.asarray(continues, dtype=np.float64)
    if rews.shape != conts.shape or rews.ndim < 2:
        raise ValueError("rewards and continues must share a time axis")

    output = np.zeros_like(rews)
    running = np.zeros_like(rews[..., 0, :])
    for time_index in range(rews.shape[-2] - 1, -1, -1):
        running = rews[..., time_index, :] + discount * conts[..., time_index, :] * running
        output[..., time_index, :] = running
    return output


def mean_and_episode_bootstrap_ci(
    values: np.ndarray,
    episode_ids: np.ndarray,
    *,
    seed: int,
    repetitions: int = 1_000,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Summarize chunk values with an episode-cluster bootstrap interval."""
    sample_values = np.asarray(values, dtype=np.float64).reshape(-1)
    clusters = np.asarray(episode_ids).reshape(-1)
    if sample_values.shape != clusters.shape:
        raise ValueError("values and episode ids must have the same number of entries")
    if sample_values.size == 0:
        raise ValueError("cannot summarize an empty metric")
    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")

    unique_ids, inverse = np.unique(clusters, return_inverse=True)
    cluster_sums = np.bincount(inverse, weights=sample_values, minlength=len(unique_ids))
    cluster_counts = np.bincount(inverse, minlength=len(unique_ids))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique_ids), size=(repetitions, len(unique_ids)))
    draw_sums = cluster_sums[draws].sum(axis=1)
    draw_counts = cluster_counts[draws].sum(axis=1)
    bootstrap_means = draw_sums / draw_counts
    alpha = (1 - confidence) / 2
    return {
        "mean": float(sample_values.mean()),
        "n_chunks": int(sample_values.size),
        "n_episodes": int(len(unique_ids)),
        "ci_low": float(np.quantile(bootstrap_means, alpha)),
        "ci_high": float(np.quantile(bootstrap_means, 1 - alpha)),
    }


def paired_episode_bootstrap_difference(
    baseline_values: np.ndarray,
    comparison_values: np.ndarray,
    episode_ids: np.ndarray,
    *,
    seed: int,
    repetitions: int = 1_000,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Compare paired chunk metrics with an episode-cluster bootstrap CI.

    The returned difference is always ``comparison - baseline``.  Callers that
    use a higher-is-better metric can reverse its sign explicitly when they
    render a directional forgetting score, while this primitive keeps the raw
    effect and its interval unambiguous.
    """
    baseline = np.asarray(baseline_values, dtype=np.float64).reshape(-1)
    comparison = np.asarray(comparison_values, dtype=np.float64).reshape(-1)
    clusters = np.asarray(episode_ids).reshape(-1)
    if baseline.shape != comparison.shape or baseline.shape != clusters.shape:
        raise ValueError("paired values and episode ids must have identical shapes")
    if baseline.size == 0:
        raise ValueError("cannot compare empty metrics")
    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")

    unique_ids, inverse = np.unique(clusters, return_inverse=True)
    cluster_counts = np.bincount(inverse, minlength=len(unique_ids))
    baseline_sums = np.bincount(inverse, weights=baseline, minlength=len(unique_ids))
    comparison_sums = np.bincount(inverse, weights=comparison, minlength=len(unique_ids))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique_ids), size=(repetitions, len(unique_ids)))
    draw_counts = cluster_counts[draws].sum(axis=1)
    bootstrap_differences = (
        comparison_sums[draws].sum(axis=1) - baseline_sums[draws].sum(axis=1)
    ) / draw_counts
    alpha = (1 - confidence) / 2
    return {
        "baseline_mean": float(baseline.mean()),
        "comparison_mean": float(comparison.mean()),
        "comparison_minus_baseline": float(comparison.mean() - baseline.mean()),
        "n_chunks": int(baseline.size),
        "n_episodes": int(len(unique_ids)),
        "ci_low": float(np.quantile(bootstrap_differences, alpha)),
        "ci_high": float(np.quantile(bootstrap_differences, 1 - alpha)),
    }
