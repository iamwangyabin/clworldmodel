"""Trainable adapters between frozen DINO patch tokens and Dreamer RSSMs."""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelLayerNorm(nn.Module):
    """Apply LayerNorm over channels while preserving an NCHW interface."""

    def __init__(self, channels: int, *, eps: float = 1e-3) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4:
            raise ValueError(
                "ChannelLayerNorm expects [batch, channels, height, width], "
                f"got {tuple(values.shape)}"
            )
        return self.norm(values.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class DinoPatchConvAdapter(nn.Module):
    """Compress a flattened 16x16 DINO patch grid to a 4096-D embedding."""

    def __init__(
        self,
        *,
        patch_grid_size: int = 16,
        in_channels: int = 384,
        out_channels: int = 64,
    ) -> None:
        super().__init__()
        if patch_grid_size < 1:
            raise ValueError("patch_grid_size must be positive")
        if in_channels < 1 or out_channels < 1:
            raise ValueError("adapter channels must be positive")
        self.patch_grid_size = patch_grid_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = patch_grid_size * patch_grid_size * in_channels
        kernel_size = 3
        stride = 2
        padding = 1
        self.output_grid_size = (
            patch_grid_size + 2 * padding - kernel_size
        ) // stride + 1
        self.output_size = out_channels * self.output_grid_size**2
        self.adapter = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            ChannelLayerNorm(out_channels),
            nn.SiLU(),
            nn.Flatten(),
        )

    def forward(self, flattened_patches: torch.Tensor) -> torch.Tensor:
        if flattened_patches.ndim != 2:
            raise ValueError(
                "DINO patch adapter expects [batch, flattened_patches], "
                f"got {tuple(flattened_patches.shape)}"
            )
        if flattened_patches.shape[-1] != self.input_size:
            raise ValueError(
                "DINO patch adapter received an unexpected feature width: "
                f"expected {self.input_size}, got {flattened_patches.shape[-1]}"
            )
        patch_grid = flattened_patches.reshape(
            -1,
            self.patch_grid_size,
            self.patch_grid_size,
            self.in_channels,
        )
        channels_first = patch_grid.permute(0, 3, 1, 2)
        return self.adapter(channels_first)
