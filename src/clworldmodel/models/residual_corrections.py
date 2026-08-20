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
        consolidation_enabled: bool = False,
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

        coefficient_shape = (
            (bottleneck_features, bottleneck_features, num_grids)
            if kind == "kan" and consolidation_enabled
            else (0,)
        )
        self.register_buffer(
            "consolidation_importance",
            torch.zeros(coefficient_shape, dtype=torch.float32),
        )
        self.register_buffer(
            "consolidation_anchor",
            torch.zeros(coefficient_shape, dtype=torch.float32),
        )
        self.register_buffer(
            "consolidation_gradient_scale",
            torch.ones(coefficient_shape, dtype=torch.float32),
        )
        self.register_buffer(
            "consolidation_anchor_loss_scale",
            torch.zeros((), dtype=torch.float32),
        )
        self.register_buffer(
            "consolidation_boundaries",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "consolidation_active",
            torch.zeros((), dtype=torch.bool),
        )
        self.register_buffer(
            "_importance_sum",
            torch.empty(0, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_importance_samples",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )
        self._collecting_importance = False

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

    def begin_importance_estimation(self) -> None:
        """Start a no-gradient replay estimate for every RBF coefficient."""
        core = self._require_kan()
        if self._collecting_importance:
            raise RuntimeError("KAN importance estimation is already active")
        if not self.consolidation_importance.numel():
            coefficient_shape = core.rbf_weight.shape
            self.consolidation_importance = torch.zeros(
                coefficient_shape,
                device=core.rbf_weight.device,
                dtype=torch.float32,
            )
            self.consolidation_anchor = torch.zeros_like(
                self.consolidation_importance
            )
            self.consolidation_gradient_scale = torch.ones_like(
                self.consolidation_importance
            )
        self._importance_sum = torch.zeros_like(self.consolidation_importance)
        self._importance_samples.zero_()
        self._collecting_importance = True

    def cancel_importance_estimation(self) -> None:
        self._collecting_importance = False
        self._importance_sum = self._importance_sum.new_empty(0)
        self._importance_samples.zero_()

    @torch.no_grad()
    def finish_importance_estimation(
        self,
        *,
        gradient_power: float,
        min_plasticity: float,
        anchor_loss_scale: float,
    ) -> dict[str, float | int]:
        """Consolidate the replay estimate and anchor newly important weights."""
        core = self._require_kan()
        if not self._collecting_importance:
            raise RuntimeError("KAN importance estimation has not been started")
        if self._importance_samples.item() < 1:
            raise RuntimeError("KAN importance estimation observed no samples")
        if gradient_power <= 0:
            raise ValueError("gradient_power must be positive")
        if not 0 <= min_plasticity <= 1:
            raise ValueError("min_plasticity must lie in [0, 1]")
        if anchor_loss_scale < 0:
            raise ValueError("anchor_loss_scale must be non-negative")

        estimate = self._importance_sum / self._importance_samples.float()
        positive = estimate[estimate > 0]
        if positive.numel():
            robust_scale = torch.quantile(positive, 0.99).clamp_min(1e-12)
            normalized = (estimate / robust_scale).clamp_(0.0, 1.0)
        else:
            robust_scale = torch.ones((), device=estimate.device)
            normalized = torch.zeros_like(estimate)

        current_weight = core.rbf_weight.detach().float()
        if not self.consolidation_active.item():
            self.consolidation_anchor.copy_(current_weight)
        else:
            newly_dominant = normalized > self.consolidation_importance
            self.consolidation_anchor.copy_(
                torch.where(newly_dominant, current_weight, self.consolidation_anchor)
            )
        self.consolidation_importance.copy_(
            torch.maximum(self.consolidation_importance, normalized)
        )
        gradient_scale = (
            (1.0 - self.consolidation_importance).pow(gradient_power)
            * (1.0 - min_plasticity)
            + min_plasticity
        )
        self.consolidation_gradient_scale.copy_(gradient_scale)
        self.consolidation_anchor_loss_scale.fill_(anchor_loss_scale)
        self.consolidation_boundaries.add_(1)
        self.consolidation_active.fill_(True)
        self._collecting_importance = False

        diagnostics = self.consolidation_diagnostics()
        diagnostics.update(
            {
                "importance_samples": int(self._importance_samples.item()),
                "raw_importance_p99": float(robust_scale.item()),
            }
        )
        self._importance_sum = self._importance_sum.new_empty(0)
        self._importance_samples.zero_()
        return diagnostics

    def freeze_coordinate_map(self) -> None:
        """Freeze the shared coordinate system and leave only RBF values plastic."""
        core = self._require_kan()
        self.down.requires_grad_(False)
        self.norm.requires_grad_(False)
        self.up.requires_grad_(False)
        core.rbf_weight.requires_grad_(True)

    def consolidation_penalty(self) -> torch.Tensor:
        """Quadratic anchor loss for replay-important RBF coefficients."""
        core = self._require_kan()
        if not self.consolidation_importance.numel():
            return core.rbf_weight.sum() * 0.0
        squared_drift = (core.rbf_weight.float() - self.consolidation_anchor).square()
        return self.consolidation_anchor_loss_scale * (
            self.consolidation_importance * squared_drift
        ).mean()

    @torch.no_grad()
    def consolidation_diagnostics(self) -> dict[str, float | int]:
        self._require_kan()
        importance = self.consolidation_importance
        if not importance.numel():
            raise RuntimeError("KAN consolidation has not been initialized")
        scale = self.consolidation_gradient_scale
        return {
            "boundaries": int(self.consolidation_boundaries.item()),
            "coefficients": importance.numel(),
            "importance_mean": float(importance.mean().item()),
            "importance_max": float(importance.max().item()),
            "protected_fraction_ge_0_5": float((importance >= 0.5).float().mean().item()),
            "protected_fraction_ge_0_9": float((importance >= 0.9).float().mean().item()),
            "gradient_scale_mean": float(scale.mean().item()),
            "gradient_scale_min": float(scale.min().item()),
        }

    def _require_kan(self) -> LocalRBFKANCore:
        if self.kind != "kan" or not isinstance(self.core, LocalRBFKANCore):
            raise RuntimeError("Replay consolidation is defined only for KAN corrections")
        return self.core

    def _effective_rbf_weight(self, core: LocalRBFKANCore) -> torch.Tensor:
        weight = core.rbf_weight
        if not self.consolidation_gradient_scale.numel():
            return weight
        scale = self.consolidation_gradient_scale.to(dtype=weight.dtype)
        # The value is exactly ``weight``; only its backward gradient is scaled.
        return weight.detach() + scale * (weight - weight.detach())

    @torch.no_grad()
    def _accumulate_importance(
        self,
        basis: torch.Tensor,
        core_values: torch.Tensor,
    ) -> None:
        """Accumulate the squared output Jacobian for each RBF coefficient."""
        flat_basis = basis.reshape(-1, basis.shape[-2], basis.shape[-1]).float()
        flat_values = core_values.reshape(-1, core_values.shape[-1]).float()
        if flat_basis.shape[0] != flat_values.shape[0]:
            raise ValueError("KAN basis and value samples must align")

        sigmoid = torch.sigmoid(flat_values)
        silu_derivative = sigmoid * (1.0 + flat_values * (1.0 - sigmoid))
        output_column_norm = self.up.weight.detach().float().square().sum(dim=0)
        output_sensitivity = (
            self.alpha**2
            * silu_derivative.square()
            * output_column_norm.unsqueeze(0)
        )
        self._importance_sum.add_(
            torch.einsum(
                "so,sig->oig",
                output_sensitivity,
                flat_basis.square(),
            )
        )
        self._importance_samples.add_(flat_basis.shape[0])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        bottleneck = self.bottleneck_inputs(inputs)
        if self.kind == "kan":
            core = self._require_kan()
            basis = core.basis_activations(bottleneck)
            values = torch.einsum(
                "...ig,oig->...o",
                basis,
                self._effective_rbf_weight(core),
            )
            if self._collecting_importance:
                self._accumulate_importance(basis, values)
        else:
            values = self.core(bottleneck)
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
    consolidation_enabled: bool = False,
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
        consolidation_enabled=consolidation_enabled,
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
