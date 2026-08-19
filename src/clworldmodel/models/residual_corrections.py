"""Fixed-capacity residual corrections for KARROW."""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fast_kan import FixedGaussianRBF, RMSNorm, _dreamer_truncated_normal_


ResidualKind = Literal["none", "mlp", "kan"]


class LocalRBFKANCore(nn.Module):
    """A fixed-grid, basis-only KAN layer used inside a residual adapter."""

    def __init__(
        self,
        features: int,
        *,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
    ) -> None:
        super().__init__()
        if features < 1:
            raise ValueError("features must be positive")
        self.features = features
        self.num_grids = num_grids
        self.rbf = FixedGaussianRBF(
            grid_min=grid_min,
            grid_max=grid_max,
            num_grids=num_grids,
        )
        self.rbf_weight = nn.Parameter(
            torch.empty(features, features, num_grids)
        )
        _dreamer_truncated_normal_(self.rbf_weight)

    def basis_activations(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.features:
            raise ValueError(
                f"Expected {self.features} input features, got {inputs.shape[-1]}"
            )
        return self.rbf(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        basis = self.basis_activations(inputs)
        return torch.einsum("...ig,oig->...o", basis, self.rbf_weight)


class ParameterMatchedMLPCore(nn.Module):
    """Bias-free MLP with the same parameter count as LocalRBFKANCore."""

    def __init__(self, features: int, num_grids: int) -> None:
        super().__init__()
        if features < 1 or num_grids < 2:
            raise ValueError("features must be positive and num_grids at least two")
        expanded = features * num_grids
        if expanded % 2:
            raise ValueError(
                "features * num_grids must be even for exact MLP parameter matching"
            )
        hidden_features = expanded // 2
        self.network = nn.Sequential(
            nn.Linear(features, hidden_features, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_features, features, bias=False),
        )
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                _dreamer_truncated_normal_(layer.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class ResidualCorrection(nn.Module):
    """Bottleneck correction with either local KAN or matched MLP capacity."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        kind: Literal["mlp", "kan"],
        bottleneck_features: int = 64,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        rms_norm_epsilon: float = 1e-4,
        alpha: float = 0.1,
    ) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1 or bottleneck_features < 1:
            raise ValueError("residual dimensions must be positive")
        if kind not in {"mlp", "kan"}:
            raise ValueError(f"Unknown residual kind: {kind!r}")
        if alpha <= 0:
            raise ValueError("residual alpha must be positive")

        self.kind = kind
        self.in_features = in_features
        self.out_features = out_features
        self.bottleneck_features = bottleneck_features
        self.alpha = float(alpha)
        self.down = nn.Linear(in_features, bottleneck_features)
        self.norm = RMSNorm(bottleneck_features, rms_norm_epsilon)
        if kind == "kan":
            self.core: nn.Module = LocalRBFKANCore(
                bottleneck_features,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
            )
        else:
            self.core = ParameterMatchedMLPCore(
                bottleneck_features,
                num_grids,
            )
        self.up = nn.Linear(bottleneck_features, out_features)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def bottleneck_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected {self.in_features} input features, got {inputs.shape[-1]}"
            )
        return self.norm(self.down(inputs))

    def basis_activations(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.kind != "kan":
            raise RuntimeError("Basis activations are defined only for KAN corrections")
        core = self.core
        if not isinstance(core, LocalRBFKANCore):
            raise TypeError("KAN correction has an unexpected core module")
        return core.basis_activations(self.bottleneck_inputs(inputs))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = self.core(self.bottleneck_inputs(inputs))
        return self.alpha * self.up(F.silu(values))


def build_residual_correction(
    kind: ResidualKind,
    in_features: int,
    out_features: int,
    *,
    bottleneck_features: int = 64,
    grid_min: float = -2.0,
    grid_max: float = 2.0,
    num_grids: int = 8,
    rms_norm_epsilon: float = 1e-4,
    alpha: float = 0.1,
) -> Optional[ResidualCorrection]:
    if kind == "none":
        return None
    if kind not in {"mlp", "kan"}:
        raise ValueError(f"Unknown residual kind: {kind!r}")
    return ResidualCorrection(
        in_features,
        out_features,
        kind=kind,
        bottleneck_features=bottleneck_features,
        grid_min=grid_min,
        grid_max=grid_max,
        num_grids=num_grids,
        rms_norm_epsilon=rms_norm_epsilon,
        alpha=alpha,
    )


def soft_basis_support_overlap(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Weighted Jaccard overlap between two sets of KAN basis activations."""
    if first.ndim < 2 or second.ndim < 2:
        raise ValueError("Basis activations require feature and grid dimensions")
    if first.shape[-2:] != second.shape[-2:]:
        raise ValueError("Basis activation feature/grid dimensions must match")
    if first.numel() == 0 or second.numel() == 0:
        raise ValueError("Basis activation sets must not be empty")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    first_support = first.reshape(-1, *first.shape[-2:]).mean(0)
    second_support = second.reshape(-1, *second.shape[-2:]).mean(0)
    intersection = torch.minimum(first_support, second_support).sum()
    union = torch.maximum(first_support, second_support).sum()
    return intersection / union.clamp_min(epsilon)
