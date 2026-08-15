"""Numerical helpers for the checkpoint-differencing audit.

This module deliberately has no dependency on the vendored trainer.  Keeping
the summary math separate makes the most important metric definitions easy to
unit test on CPU-only machines.
"""

from __future__ import annotations

import math
from typing import Iterable

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
    cross_sq = float(np.square(x.T @ y).sum())
    x_sq = float(np.square(x.T @ x).sum())
    y_sq = float(np.square(y.T @ y).sum())
    denominator = math.sqrt(x_sq * y_sq)
    if denominator == 0:
        return 0.0
    return cross_sq / denominator


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


def scalar_rows(values: Iterable[float]) -> np.ndarray:
    """Convert a scalar iterable to a finite float64 vector with clear validation."""
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("metric values must be a finite one-dimensional sequence")
    return array
