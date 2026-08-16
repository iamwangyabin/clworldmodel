"""Self-supervised latent representation objective used by R2-Dreamer."""

from __future__ import annotations

import torch
import torch.nn as nn


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


class R2Projector(nn.Linear):
    """Bias-free linear map from the RSSM feature to the encoder feature space."""

    def __init__(self, rssm_features: int, encoder_features: int) -> None:
        if rssm_features < 1 or encoder_features < 1:
            raise ValueError("R2 projector dimensions must be positive")
        super().__init__(rssm_features, encoder_features, bias=False)


class R2BarlowObjective(nn.Module):
    """Configured R2-Dreamer objective with a stop-gradient encoder target."""

    def __init__(
        self,
        redundancy_scale: float = 5e-4,
        normalization_eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if redundancy_scale < 0:
            raise ValueError("Redundancy scale must be non-negative")
        if normalization_eps <= 0:
            raise ValueError("Normalization epsilon must be positive")
        self.redundancy_scale = redundancy_scale
        self.normalization_eps = normalization_eps

    def forward(
        self, projected: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return barlow_twins_loss(
            projected,
            target,
            redundancy_scale=self.redundancy_scale,
            normalization_eps=self.normalization_eps,
        )
