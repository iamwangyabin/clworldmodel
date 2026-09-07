"""Self-supervised latent representation objective used by R2-Dreamer."""

from __future__ import annotations

import torch


def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Cross-correlation matrix must be square")
    size = matrix.shape[0]
    return matrix.flatten()[:-1].view(size - 1, size + 1)[:, 1:].flatten()


def barlow_twins_loss(
    projected: torch.Tensor,
    target: torch.Tensor,
    redundancy_scale: float = 5e-4,
    normalization_eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, invariance, and redundancy losses for two `[S, E]` views."""
    if projected.ndim != 2 or projected.shape != target.shape:
        raise ValueError(
            "Projected and target representations must have equal [samples, features] "
            f"shapes, got {tuple(projected.shape)} and {tuple(target.shape)}"
        )
    if projected.shape[0] < 2:
        raise ValueError("Barlow Twins normalization requires at least two samples")
    if redundancy_scale < 0:
        raise ValueError("Redundancy scale must be non-negative")
    if normalization_eps <= 0:
        raise ValueError("Normalization epsilon must be positive")

    target = target.detach()
    projected_norm = (projected - projected.mean(0)) / (
        projected.std(0) + normalization_eps
    )
    target_norm = (target - target.mean(0)) / (target.std(0) + normalization_eps)
    correlation = projected_norm.T @ target_norm / projected.shape[0]
    invariance = (torch.diagonal(correlation) - 1).square().sum()
    redundancy = _off_diagonal(correlation).square().sum()
    return invariance + redundancy_scale * redundancy, invariance, redundancy
