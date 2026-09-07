"""Task-private zero-effect projection of shared CNN features."""

import torch
import torch.nn as nn


class SpatialFeatureProjector(nn.Module):
    """A zero-effect residual projector for flattened 4x4 CNN features."""

    def __init__(
        self,
        *,
        channels: int = 256,
        spatial_size: int = 4,
        bottleneck_channels: int = 64,
    ) -> None:
        super().__init__()
        if channels < 1 or spatial_size < 1 or bottleneck_channels < 1:
            raise ValueError("Projector dimensions must be positive")
        if channels % 32:
            raise ValueError("Projector channels must be divisible by 32")
        self.channels = channels
        self.spatial_size = spatial_size
        self.output_size = channels * spatial_size * spatial_size
        self.norm = nn.GroupNorm(32, channels)
        self.project = nn.Sequential(
            nn.Conv2d(channels, bottleneck_channels, 1),
            nn.SiLU(),
            nn.Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                3,
                padding=1,
                groups=bottleneck_channels,
            ),
            nn.SiLU(),
            nn.Conv2d(bottleneck_channels, channels, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        final = self.project[-1]
        if not isinstance(final, nn.Conv2d):
            raise TypeError("Projector output must be a convolution")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.output_size:
            raise ValueError(
                f"Expected {self.output_size} CNN features, got {features.shape[-1]}"
            )
        leading = features.shape[:-1]
        spatial = features.reshape(
            -1, self.channels, self.spatial_size, self.spatial_size
        )
        projected = spatial + self.project(self.norm(spatial))
        return projected.flatten(1).reshape(*leading, self.output_size)
