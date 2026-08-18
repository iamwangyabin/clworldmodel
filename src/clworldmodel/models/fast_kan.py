"""Fixed-grid FastKAN behavior networks for the KAN-Dreamer ablation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Dreamer-style RMS normalization over the final feature axis."""

    def __init__(self, features: int, epsilon: float = 1e-4) -> None:
        super().__init__()
        if features < 1:
            raise ValueError("features must be positive")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = epsilon
        self.scale = nn.Parameter(torch.ones(features))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        source_dtype = inputs.dtype
        normalized = inputs.float()
        inverse_rms = torch.rsqrt(normalized.square().mean(-1, keepdim=True) + self.epsilon)
        normalized = normalized * inverse_rms * self.scale.float()
        return normalized.to(source_dtype)


class FixedGaussianRBF(nn.Module):
    """Gaussian basis with fixed, uniformly spaced FastKAN centers."""

    def __init__(
        self,
        *,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
    ) -> None:
        super().__init__()
        if num_grids < 2:
            raise ValueError("num_grids must be at least 2")
        if grid_max <= grid_min:
            raise ValueError("grid_max must exceed grid_min")
        centers = torch.linspace(grid_min, grid_max, num_grids)
        self.register_buffer("centers", centers)
        self.bandwidth = (grid_max - grid_min) / (num_grids - 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        centers = self.centers.to(dtype=inputs.dtype)
        return torch.exp(-((inputs.unsqueeze(-1) - centers) / self.bandwidth).square())


def _dreamer_truncated_normal_(weight: torch.Tensor, scale: float = 1.0) -> None:
    """Initialize like DreamerV3's fan-in truncated-normal linear kernel."""

    if scale == 0:
        nn.init.zeros_(weight)
        return
    fan_in = weight.shape[1:].numel()
    std = 1.1368 * scale / math.sqrt(fan_in)
    nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2 * std, b=2 * std)


class FastKANLayer(nn.Module):
    """Vectorized FastKAN layer with RBF and SiLU base branches."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        rms_norm_epsilon: float = 1e-4,
        output_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError("in_features and out_features must be positive")
        if output_scale < 0:
            raise ValueError("output_scale must be non-negative")
        self.in_features = in_features
        self.out_features = out_features
        self.num_grids = num_grids
        self.norm = RMSNorm(in_features, rms_norm_epsilon)
        self.rbf = FixedGaussianRBF(
            grid_min=grid_min,
            grid_max=grid_max,
            num_grids=num_grids,
        )
        self.rbf_weight = nn.Parameter(
            torch.empty(out_features, in_features, num_grids)
        )
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.base_bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters(output_scale)

    def reset_parameters(self, output_scale: float = 1.0) -> None:
        _dreamer_truncated_normal_(self.rbf_weight, output_scale)
        _dreamer_truncated_normal_(self.base_weight, output_scale)
        nn.init.zeros_(self.base_bias)

    def basis_activations(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.rbf(self.norm(inputs))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected {self.in_features} input features, got {inputs.shape[-1]}"
            )
        basis = self.basis_activations(inputs)
        rbf_output = torch.einsum("...ig,oig->...o", basis, self.rbf_weight)
        base_output = F.linear(F.silu(inputs), self.base_weight, self.base_bias)
        return rbf_output + base_output


class FastKAN(nn.Module):
    """Three-hidden-layer FastKAN backbone used by KAN-Dreamer behavior heads."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        hidden_features: int = 34,
        hidden_layers: int = 3,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        rms_norm_epsilon: float = 1e-4,
        output_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_features < 1 or hidden_layers < 1:
            raise ValueError("hidden_features and hidden_layers must be positive")
        widths = [in_features, *([hidden_features] * hidden_layers), out_features]
        self.layers = nn.ModuleList(
            FastKANLayer(
                input_width,
                output_width,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                rms_norm_epsilon=rms_norm_epsilon,
                output_scale=output_scale if index == len(widths) - 2 else 1.0,
            )
            for index, (input_width, output_width) in enumerate(
                zip(widths[:-1], widths[1:])
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = inputs
        for layer in self.layers:
            values = layer(values)
        return values


class FastKANActor(nn.Module):
    """Discrete FastKAN policy with DreamerV3 categorical unimix."""

    def __init__(
        self,
        in_features: int,
        action_features: int,
        *,
        hidden_features: int = 34,
        hidden_layers: int = 3,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        rms_norm_epsilon: float = 1e-4,
        output_scale: float = 0.01,
        unimix: float = 0.01,
    ) -> None:
        super().__init__()
        if not 0.0 <= unimix < 1.0:
            raise ValueError("unimix must lie in [0, 1)")
        self.action_features = action_features
        self.unimix = unimix
        self.network = FastKAN(
            in_features,
            action_features,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            grid_min=grid_min,
            grid_max=grid_max,
            num_grids=num_grids,
            rms_norm_epsilon=rms_norm_epsilon,
            output_scale=output_scale,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        probabilities = self.network(state).softmax(-1)
        if self.unimix:
            probabilities = (
                (1.0 - self.unimix) * probabilities
                + self.unimix / self.action_features
            )
        return probabilities.log()


class FastKANCritic(nn.Module):
    """FastKAN categorical value head with DreamerV3 zero output init."""

    def __init__(
        self,
        in_features: int,
        value_features: int = 255,
        *,
        hidden_features: int = 34,
        hidden_layers: int = 3,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        rms_norm_epsilon: float = 1e-4,
    ) -> None:
        super().__init__()
        self.network = FastKAN(
            in_features,
            value_features,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            grid_min=grid_min,
            grid_max=grid_max,
            num_grids=num_grids,
            rms_norm_epsilon=rms_norm_epsilon,
            output_scale=0.0,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.network(state), dim=-1)
