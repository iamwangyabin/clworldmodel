"""Task-specific projector and affine LoRA utilities for a frozen RSSM core."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from torch.nn.utils import parametrize


class LowRankWeightDelta(nn.Module):
    """Add a zero-effect trainable low-rank delta to a frozen matrix."""

    def __init__(self, weight: torch.Tensor, rank: int) -> None:
        super().__init__()
        if weight.ndim != 2:
            raise ValueError("LoRA weights must be matrices")
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        out_features, in_features = weight.shape
        self.rank = min(rank, out_features, in_features)
        self.a = nn.Parameter(weight.new_empty(self.rank, in_features))
        self.b = nn.Parameter(weight.new_zeros(out_features, self.rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def reset_delta(self) -> None:
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        nn.init.zeros_(self.b)

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        return original + self.b @ self.a


class ExactVectorDelta(nn.Module):
    """Keep task-specific affine vectors exact while matrices use LoRA."""

    def __init__(self, original: torch.Tensor) -> None:
        super().__init__()
        if original.ndim != 1:
            raise ValueError("Exact vector deltas require one-dimensional tensors")
        self.delta = nn.Parameter(torch.zeros_like(original))

    def reset_delta(self) -> None:
        nn.init.zeros_(self.delta)

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        return original + self.delta


class BottleneckOutputDelta(nn.Module):
    """Zero-effect nonlinear correction for a frozen module output."""

    def __init__(self, features: int, bottleneck_features: int) -> None:
        super().__init__()
        if features < 1 or bottleneck_features < 1:
            raise ValueError("Output adapter dimensions must be positive")
        self.features = features
        self.bottleneck_features = bottleneck_features
        self.norm = nn.LayerNorm(features, eps=1e-3)
        self.down = nn.Linear(features, bottleneck_features)
        self.activation = nn.SiLU()
        self.up = nn.Linear(bottleneck_features, features)
        self.reset_delta()

    def reset_delta(self) -> None:
        self.norm.reset_parameters()
        self.down.reset_parameters()
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, output: torch.Tensor) -> torch.Tensor:
        if output.shape[-1] != self.features:
            raise ValueError(
                f"Expected {self.features} output features, got {output.shape[-1]}"
            )
        return self.up(self.activation(self.down(self.norm(output))))


class TaskRecurrentOutputRoute(nn.Module):
    """Reuse one frozen recurrent core and own only an output correction."""

    def __init__(
        self,
        base: nn.Module,
        *,
        output_features: int,
        bottleneck_features: int,
    ) -> None:
        super().__init__()
        # The base is already registered by the parent RSSM. Keeping this
        # reference non-registered avoids duplicate checkpoint tensors.
        object.__setattr__(self, "_base", base)
        self.adapter = BottleneckOutputDelta(
            output_features, bottleneck_features
        )

    @property
    def base(self) -> nn.Module:
        return object.__getattribute__(self, "_base")

    def reset_delta(self) -> None:
        self.adapter.reset_delta()

    def freeze_shared_core(self) -> None:
        self.base.requires_grad_(False)
        self.adapter.requires_grad_(True)

    def forward(
        self,
        prev_z: torch.Tensor,
        prev_a: torch.Tensor,
        prev_h: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.base(prev_z, prev_a, prev_h)
        return hidden + self.adapter(hidden)

    def parameter_report(self) -> dict[str, int | str]:
        parameters = sum(parameter.numel() for parameter in self.parameters())
        return {
            "kind": "gru_output_bottleneck_residual",
            "output_features": self.adapter.features,
            "bottleneck_features": self.adapter.bottleneck_features,
            "trainable_parameters": parameters,
        }


def set_recurrent_output_adapter_trainable(
    route: TaskRecurrentOutputRoute, trainable: bool
) -> None:
    """Keep the shared recurrent base frozen and select one route adapter."""
    route.base.requires_grad_(False)
    route.adapter.requires_grad_(trainable)


def _affine_parameter_names(module: nn.Module) -> Iterable[str]:
    if isinstance(module, nn.GRUCell):
        return ("weight_ih", "weight_hh", "bias_ih", "bias_hh")
    if isinstance(module, (nn.Linear, nn.LayerNorm)):
        return ("weight", "bias")
    return ()


def install_affine_lora(module: nn.Module, rank: int) -> dict[str, Any]:
    """Parameterize every Linear, GRUCell, and LayerNorm affine in ``module``."""
    if rank < 1:
        raise ValueError("LoRA rank must be positive")
    layers: dict[str, Any] = {}
    matrix_parameters = 0
    vector_parameters = 0
    for module_name, child in list(module.named_modules()):
        for parameter_name in _affine_parameter_names(child):
            parameter = getattr(child, parameter_name, None)
            if parameter is None:
                continue
            label = ".".join(
                part for part in (module_name, parameter_name) if part
            )
            if parameter.ndim == 2:
                delta: nn.Module = LowRankWeightDelta(parameter, rank)
                count = delta.a.numel() + delta.b.numel()  # type: ignore[attr-defined]
                matrix_parameters += count
                metadata = {
                    "shape": list(parameter.shape),
                    "rank": delta.rank,  # type: ignore[attr-defined]
                    "parameters": count,
                }
            elif parameter.ndim == 1:
                delta = ExactVectorDelta(parameter)
                count = delta.delta.numel()  # type: ignore[attr-defined]
                vector_parameters += count
                metadata = {
                    "shape": list(parameter.shape),
                    "stored_as_exact_delta": True,
                    "parameters": count,
                }
            else:
                raise TypeError(
                    f"Unsupported affine parameter {label} with shape {parameter.shape}"
                )
            parametrize.register_parametrization(child, parameter_name, delta)
            getattr(child.parametrizations, parameter_name).original.requires_grad_(
                False
            )
            layers[label] = metadata
    if not layers:
        raise ValueError("No supported RSSM affine parameters were found")
    return {
        "rank": rank,
        "matrix_lora_parameters": matrix_parameters,
        "exact_vector_parameters": vector_parameters,
        "trainable_parameters": matrix_parameters + vector_parameters,
        "layers": layers,
    }


def reset_affine_lora_from(module: nn.Module, source: nn.Module) -> None:
    """Share one frozen base route and reset all task-specific deltas."""
    source_modules = dict(source.named_modules())
    target_modules = dict(module.named_modules())
    missing = set(source_modules) - set(target_modules)
    if missing:
        raise ValueError(
            f"Source and target RSSM routes have different module graphs: {missing}"
        )
    for module_name, source_child in source_modules.items():
        target_child = target_modules[module_name]
        with torch.no_grad():
            for parameter_name in _affine_parameter_names(target_child):
                parameter = getattr(target_child, parameter_name, None)
                if parameter is None:
                    continue
                if not parametrize.is_parametrized(target_child, parameter_name):
                    raise ValueError(
                        f"Target affine {module_name}.{parameter_name} is not LoRA"
                    )
                source_parameter = getattr(source_child, parameter_name, None)
                if source_parameter is None or source_parameter.shape != parameter.shape:
                    raise ValueError(
                        f"Source affine mismatch for {module_name}.{parameter_name}"
                    )
                parametrizations = getattr(
                    target_child.parametrizations, parameter_name
                )
                if parametrize.is_parametrized(source_child, parameter_name):
                    raise ValueError(
                        "RSSM LoRA routes must share the unparameterized Task-1 base"
                    )
                # Reusing the exact Parameter keeps one Task-1 RSSM core in memory;
                # each later route owns only its low-rank/vector deltas.
                parametrizations.original = source_parameter
                for delta in parametrizations:
                    if not isinstance(delta, (LowRankWeightDelta, ExactVectorDelta)):
                        raise TypeError("Unexpected RSSM parametrization")
                    delta.reset_delta()


def set_affine_lora_trainable(module: nn.Module, trainable: bool) -> None:
    """Freeze route originals and expose only its LoRA/vector deltas."""
    module.requires_grad_(False)
    for child in module.modules():
        parametrizations = getattr(child, "parametrizations", None)
        if parametrizations is None:
            continue
        for parameter_name in _affine_parameter_names(child):
            if parameter_name not in parametrizations:
                continue
            parameterizations = parametrizations[parameter_name]
            parameterizations.original.requires_grad_(False)
            for delta in parameterizations:
                delta.requires_grad_(trainable)


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
