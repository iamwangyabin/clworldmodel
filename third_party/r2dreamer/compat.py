"""Small compatibility layer for the pinned R2-Dreamer vendor."""

from __future__ import annotations

from collections.abc import Sequence
import torch
from torch import nn


if hasattr(nn, "RMSNorm"):
    RMSNorm = nn.RMSNorm
else:

    class RMSNorm(nn.Module):
        """PyTorch < 2.4 compatible RMSNorm with the upstream parameter layout."""

        def __init__(
            self,
            normalized_shape: int | Sequence[int],
            eps: float | None = None,
            elementwise_affine: bool = True,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
        ) -> None:
            super().__init__()
            shape = (
                (normalized_shape,)
                if isinstance(normalized_shape, int)
                else tuple(normalized_shape)
            )
            self.normalized_shape = shape
            self.eps = torch.finfo(torch.float32).eps if eps is None else eps
            self.elementwise_affine = elementwise_affine
            if elementwise_affine:
                self.weight = nn.Parameter(torch.ones(shape, device=device, dtype=dtype))
            else:
                self.register_parameter("weight", None)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            dimensions = tuple(range(-len(self.normalized_shape), 0))
            variance = value.square().mean(dim=dimensions, keepdim=True)
            result = value * torch.rsqrt(variance + self.eps)
            if self.weight is not None:
                result = result * self.weight
            return result
