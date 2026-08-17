"""ReLU-KAN actor layers for continual world-model experiments.

The basis follows Equation 6 of ReLU-KAN (Qiu et al., 2024), implemented
independently with PyTorch tensor operations. Inputs and outputs use the last
axis as the feature axis, so the modules accept both ``[N, D]`` and
``[T, N, D]`` tensors without adapter reshapes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedGridReLUKANLayer(nn.Module):
    """A ReLU-KAN layer with fixed compact-support basis locations."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        grid_size: int = 5,
        spline_order: int = 3,
        input_min: float = 0.0,
        input_max: float = 1.0,
    ) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError("ReLU-KAN feature dimensions must be positive")
        if grid_size < 1:
            raise ValueError("ReLU-KAN grid_size must be positive")
        if spline_order < 0:
            raise ValueError("ReLU-KAN spline_order must be non-negative")
        if input_max <= input_min:
            raise ValueError("ReLU-KAN input_max must be greater than input_min")

        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.basis_count = grid_size + spline_order

        step = (input_max - input_min) / grid_size
        phase_low = input_min + step * torch.arange(
            -spline_order, grid_size, dtype=torch.float32
        )
        support_width = step * (spline_order + 1)
        self.register_buffer("phase_low", phase_low)
        self.register_buffer("phase_high", phase_low + support_width)
        self.register_buffer(
            "basis_scale", torch.tensor(4.0 / support_width**2, dtype=torch.float32)
        )

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, self.basis_count)
        )
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.in_features * self.basis_count
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def basis_activations(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return compact basis values with shape ``[..., in_features, B]``."""
        if not inputs.is_floating_point():
            raise TypeError("ReLU-KAN inputs must use a floating-point dtype")
        if inputs.ndim < 1 or inputs.shape[-1] != self.in_features:
            raise ValueError(
                "ReLU-KAN expected the final input dimension to be "
                f"{self.in_features}, got {tuple(inputs.shape)}"
            )

        expanded = inputs.unsqueeze(-1)
        rising = torch.relu(expanded - self.phase_low)
        falling = torch.relu(self.phase_high - expanded)
        return (rising * falling * self.basis_scale).square()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        basis = self.basis_activations(inputs)
        flattened_basis = basis.flatten(-2)
        flattened_weight = self.weight.flatten(1)
        return F.linear(flattened_basis, flattened_weight, self.bias)


class FixedGridReLUKAN(nn.Module):
    """Compose fixed-grid ReLU-KAN layers over a sequence of feature widths."""

    def __init__(
        self,
        widths: Sequence[int],
        *,
        grid_size: int = 5,
        spline_order: int = 3,
        input_min: float = 0.0,
        input_max: float = 1.0,
    ) -> None:
        super().__init__()
        if len(widths) < 2:
            raise ValueError("ReLU-KAN requires at least input and output widths")
        if any(width < 1 for width in widths):
            raise ValueError("All ReLU-KAN widths must be positive")

        self.widths = tuple(widths)
        self.layers = nn.ModuleList(
            FixedGridReLUKANLayer(
                in_features,
                out_features,
                grid_size=grid_size,
                spline_order=spline_order,
                input_min=input_min,
                input_max=input_max,
            )
            for in_features, out_features in zip(widths[:-1], widths[1:])
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


class TrainableAnchorReLUKANLayer(nn.Module):
    """ReLU-KAN layer with a learned support interval per input feature.

    ReLU-KAN defines each compact basis by its start and end points.  This
    implementation stores an unconstrained start and a softplus-parameterized
    width, so the support remains ordered while both its position and width can
    receive gradients.  The initialized basis exactly matches
    :class:`FixedGridReLUKANLayer` for the same grid settings.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        grid_size: int = 5,
        spline_order: int = 3,
        input_min: float = 0.0,
        input_max: float = 1.0,
    ) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError("ReLU-KAN feature dimensions must be positive")
        if grid_size < 1:
            raise ValueError("ReLU-KAN grid_size must be positive")
        if spline_order < 0:
            raise ValueError("ReLU-KAN spline_order must be non-negative")
        if input_max <= input_min:
            raise ValueError("ReLU-KAN input_max must be greater than input_min")

        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.basis_count = grid_size + spline_order

        step = (input_max - input_min) / grid_size
        phase_low = input_min + step * torch.arange(
            -spline_order, grid_size, dtype=torch.float32
        )
        support_width = step * (spline_order + 1)
        initial_start = phase_low.expand(in_features, -1).clone()
        initial_width = torch.full_like(initial_start, support_width)
        self.register_buffer("initial_anchor_start", initial_start)
        self.register_buffer("initial_anchor_width", initial_width)
        self.anchor_start = nn.Parameter(initial_start.clone())
        self.anchor_raw_width = nn.Parameter(self._inverse_softplus(initial_width))

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, self.basis_count)
        )
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    @staticmethod
    def _inverse_softplus(values: torch.Tensor) -> torch.Tensor:
        if torch.any(values <= 0):
            raise ValueError("ReLU-KAN support widths must be positive")
        return values + torch.log(-torch.expm1(-values))

    def anchor_widths(self) -> torch.Tensor:
        """Return the positive learned support widths for every input and basis."""
        widths = F.softplus(self.anchor_raw_width)
        return widths.clamp_min(torch.finfo(widths.dtype).eps)

    def anchor_ends(self) -> torch.Tensor:
        """Return the learned right endpoints of every compact support."""
        return self.anchor_start + self.anchor_widths()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.in_features * self.basis_count
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def basis_activations(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return learned-support basis values with shape ``[..., in_features, B]``."""
        if not inputs.is_floating_point():
            raise TypeError("ReLU-KAN inputs must use a floating-point dtype")
        if inputs.ndim < 1 or inputs.shape[-1] != self.in_features:
            raise ValueError(
                "ReLU-KAN expected the final input dimension to be "
                f"{self.in_features}, got {tuple(inputs.shape)}"
            )

        expanded = inputs.unsqueeze(-1)
        starts = self.anchor_start
        widths = self.anchor_widths()
        rising = torch.relu(expanded - starts)
        falling = torch.relu(starts + widths - expanded)
        basis_scale = 4.0 / widths.square()
        return (rising * falling * basis_scale).square()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        basis = self.basis_activations(inputs)
        flattened_basis = basis.flatten(-2)
        flattened_weight = self.weight.flatten(1)
        return F.linear(flattened_basis, flattened_weight, self.bias)


class TrainableAnchorReLUKAN(nn.Module):
    """Compose trainable-anchor ReLU-KAN layers over feature widths."""

    def __init__(
        self,
        widths: Sequence[int],
        *,
        grid_size: int = 5,
        spline_order: int = 3,
        input_min: float = 0.0,
        input_max: float = 1.0,
    ) -> None:
        super().__init__()
        if len(widths) < 2:
            raise ValueError("ReLU-KAN requires at least input and output widths")
        if any(width < 1 for width in widths):
            raise ValueError("All ReLU-KAN widths must be positive")

        self.widths = tuple(widths)
        self.layers = nn.ModuleList(
            TrainableAnchorReLUKANLayer(
                in_features,
                out_features,
                grid_size=grid_size,
                spline_order=spline_order,
                input_min=input_min,
                input_max=input_max,
            )
            for in_features, out_features in zip(widths[:-1], widths[1:])
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


class ReLUKANActor(nn.Module):
    """Original fixed-grid ReLU-KAN actor for discrete DreamerV3 states.

    This class preserves the first ``relu_kan`` pilot exactly. Its two KAN
    layers are directly composed, so it is retained for checkpoint
    compatibility and for documenting the failed-grid pilot. New experiments
    should use :class:`BoundedReLUKANActor` instead.
    """

    network_class = FixedGridReLUKAN

    def __init__(
        self,
        in_features: int,
        action_features: int,
        *,
        recurrent_features: int,
        hidden_features: int = 64,
        grid_size: int = 5,
        spline_order: int = 3,
        input_min: float = 0.0,
        input_max: float = 1.0,
        normalize_recurrent_state: bool = True,
    ) -> None:
        super().__init__()
        if recurrent_features < 1 or recurrent_features >= in_features:
            raise ValueError(
                "recurrent_features must be positive and smaller than in_features"
            )
        if not normalize_recurrent_state:
            raise ValueError(
                "The fixed actor grid requires recurrent-state normalization"
            )

        self.in_features = in_features
        self.action_features = action_features
        self.recurrent_features = recurrent_features
        self.input_min = input_min
        self.input_max = input_max
        self.normalize_recurrent_state = normalize_recurrent_state
        self.network = self.network_class(
            (in_features, hidden_features, action_features),
            grid_size=grid_size,
            spline_order=spline_order,
            input_min=input_min,
            input_max=input_max,
        )

    def grid_inputs(self, state: torch.Tensor) -> torch.Tensor:
        """Convert a Dreamer state into the fixed ``[0, 1]`` KAN domain."""
        if state.ndim < 1 or state.shape[-1] != self.in_features:
            raise ValueError(
                f"KAN actor expected state shape [..., {self.in_features}], "
                f"got {tuple(state.shape)}"
            )
        discrete = state[..., : -self.recurrent_features]
        recurrent = state[..., -self.recurrent_features :]
        recurrent = self.input_min + 0.5 * (recurrent + 1.0) * (
            self.input_max - self.input_min
        )
        return torch.cat((discrete, recurrent), dim=-1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.network(self.grid_inputs(state))
        return F.log_softmax(logits, dim=-1)


class BoundedReLUKANActor(ReLUKANActor):
    """Fixed-grid ReLU-KAN actor with a bounded hidden KAN interface.

    A fixed local grid is meaningful only when every KAN layer receives values
    inside its support. The original pilot normalized the external Dreamer
    state but sent the unbounded first KAN output directly into the second
    fixed grid. This variant keeps the same local basis and coefficient budget
    while inserting a LayerNorm--sigmoid adapter between the two KAN layers.
    The adapter guarantees that the second KAN layer receives values in
    ``(0, 1)`` for every batch and time shape.
    """

    def __init__(
        self,
        in_features: int,
        action_features: int,
        *,
        recurrent_features: int,
        hidden_features: int = 64,
        grid_size: int = 5,
        spline_order: int = 3,
        input_min: float = 0.0,
        input_max: float = 1.0,
        normalize_recurrent_state: bool = True,
        layer_norm_epsilon: float = 1e-3,
    ) -> None:
        if layer_norm_epsilon <= 0:
            raise ValueError("layer_norm_epsilon must be positive")
        super().__init__(
            in_features,
            action_features,
            recurrent_features=recurrent_features,
            hidden_features=hidden_features,
            grid_size=grid_size,
            spline_order=spline_order,
            input_min=input_min,
            input_max=input_max,
            normalize_recurrent_state=normalize_recurrent_state,
        )
        self.hidden_adapter = nn.Sequential(
            nn.LayerNorm(hidden_features, eps=layer_norm_epsilon),
            nn.Sigmoid(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        hidden = self.network.layers[0](self.grid_inputs(state))
        hidden = self.hidden_adapter(hidden)
        logits = self.network.layers[1](hidden)
        return F.log_softmax(logits, dim=-1)


class AdaptiveReLUKANActor(BoundedReLUKANActor):
    """Bounded ReLU-KAN actor with trainable per-input support anchors.

    The bounded interface is retained so the second KAN layer starts with active
    bases.  Unlike :class:`BoundedReLUKANActor`, each ReLU-KAN basis learns its
    start and positive width independently for every input feature.
    """

    network_class = TrainableAnchorReLUKAN
