"""Task-private, zero-effect adapters for shared prediction heads."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroEffectFeatureAdapter(nn.Module):
    """Add a low-rank task-private feature delta before a frozen head.

    The final projection is initialized to exactly zero, so installing an
    adapter cannot change the shared head's output before the adapter receives
    an optimizer update.  The module owns no reference to the shared head; one
    adapter can therefore be routed independently for observation, reward, or
    continuation prediction without duplicating the head itself.
    """

    def __init__(
        self,
        in_features: int,
        rank: int,
        *,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if in_features < 1:
            raise ValueError("Prediction-adapter input width must be positive")
        if rank < 1:
            raise ValueError("Prediction-adapter rank must be positive")
        if residual_scale <= 0:
            raise ValueError("Prediction-adapter residual scale must be positive")

        self.in_features = int(in_features)
        self.rank = int(rank)
        self.residual_scale = float(residual_scale)
        self.norm = nn.LayerNorm(self.in_features, eps=1e-3)
        self.down = nn.Linear(self.in_features, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.in_features)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.norm.reset_parameters()
        self.down.reset_parameters()
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def delta(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected {self.in_features} prediction features, "
                f"got {inputs.shape[-1]}"
            )
        hidden = F.silu(self.down(self.norm(inputs)))
        return self.residual_scale * self.up(hidden)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.delta(inputs)

    def parameter_report(self) -> dict[str, int | float | str]:
        return {
            "kind": "zero_effect_low_rank_feature_adapter",
            "in_features": self.in_features,
            "rank": self.rank,
            "residual_scale": self.residual_scale,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
        }
