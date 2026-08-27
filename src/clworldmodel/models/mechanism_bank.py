"""Reusable, task-routed residual mechanisms for a frozen RSSM base."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMechanism(nn.Module):
    """A nonlinear residual branch with an exact zero-effect initialization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if min(in_features, out_features, hidden_features) < 1:
            raise ValueError("Mechanism dimensions must be positive")
        if residual_scale <= 0:
            raise ValueError("residual_scale must be positive")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.hidden_features = int(hidden_features)
        self.residual_scale = float(residual_scale)
        self.norm = nn.LayerNorm(self.in_features, eps=1e-3)
        self.down = nn.Linear(self.in_features, self.hidden_features)
        self.up = nn.Linear(self.hidden_features, self.out_features)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.norm.reset_parameters()
        self.down.reset_parameters()
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected {self.in_features} mechanism features, "
                f"got {inputs.shape[-1]}"
            )
        hidden = F.silu(self.down(self.norm(inputs)))
        return self.residual_scale * self.up(hidden)

    def parameter_report(self) -> dict[str, int | float | str]:
        return {
            "kind": "zero_effect_nonlinear_residual",
            "in_features": self.in_features,
            "out_features": self.out_features,
            "hidden_features": self.hidden_features,
            "residual_scale": self.residual_scale,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
        }


class ReuseRoute(nn.Module):
    """Independent tanh gates over mechanisms learned by earlier tasks."""

    def __init__(self, num_old_mechanisms: int) -> None:
        super().__init__()
        if num_old_mechanisms < 0:
            raise ValueError("num_old_mechanisms cannot be negative")
        if num_old_mechanisms == 0:
            self.register_parameter("logits", None)
        else:
            self.logits = nn.Parameter(torch.zeros(num_old_mechanisms))

    def forward(self, reference: torch.Tensor) -> torch.Tensor:
        if self.logits is None:
            return reference.new_empty(0)
        # Preserve the activation dtype while keeping gradients to FP32 logits.
        return torch.tanh(self.logits).to(
            dtype=reference.dtype, device=reference.device
        )

    def reset_parameters(self) -> None:
        if self.logits is not None:
            nn.init.zeros_(self.logits)


class MechanismBank(nn.Module):
    """One full residual mechanism per later task plus optional old-task reuse."""

    def __init__(
        self,
        *,
        num_tasks: int,
        in_features: int,
        out_features: int,
        hidden_features: int,
        residual_scale: float = 0.1,
        reuse_enabled: bool = True,
    ) -> None:
        super().__init__()
        if num_tasks < 2:
            raise ValueError("MechanismBank requires at least two tasks")

        self.num_tasks = int(num_tasks)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.hidden_features = int(hidden_features)
        self.residual_scale = float(residual_scale)
        self.reuse_enabled = bool(reuse_enabled)
        self.mechanisms = nn.ModuleList(
            ResidualMechanism(
                in_features=self.in_features,
                out_features=self.out_features,
                hidden_features=self.hidden_features,
                residual_scale=self.residual_scale,
            )
            for _ in range(self.num_tasks - 1)
        )
        self.routes = nn.ModuleList(
            ReuseRoute(num_old_mechanisms=task_id - 1)
            for task_id in range(1, self.num_tasks)
        )

    def _task_index(self, task_id: int) -> int:
        if not isinstance(task_id, int):
            raise TypeError("Mechanism-bank task_id must be an integer")
        if not 0 <= task_id < self.num_tasks:
            raise ValueError(f"Invalid mechanism-bank task_id: {task_id}")
        return task_id

    def forward(self, inputs: torch.Tensor, task_id: int) -> torch.Tensor:
        task_index = self._task_index(task_id)
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected {self.in_features} mechanism-bank features, "
                f"got {inputs.shape[-1]}"
            )
        output_shape = (*inputs.shape[:-1], self.out_features)
        if task_index == 0:
            return inputs.new_zeros(output_shape)

        current_index = task_index - 1
        correction = self.mechanisms[current_index](inputs)
        if self.reuse_enabled:
            gates = self.routes[current_index](inputs)
            for mechanism_index, gate in enumerate(gates.unbind()):
                correction = correction + gate * self.mechanisms[mechanism_index](
                    inputs
                )
        return correction

    def activate_task(self, task_id: int) -> None:
        """Expose only the selected task's new mechanism and reuse gates."""
        task_index = self._task_index(task_id)
        self.requires_grad_(False)
        if task_index == 0:
            return
        current_index = task_index - 1
        self.mechanisms[current_index].requires_grad_(True)
        if self.reuse_enabled:
            self.routes[current_index].requires_grad_(True)

    def reset_task(self, task_id: int) -> None:
        task_index = self._task_index(task_id)
        if task_index == 0:
            return
        current_index = task_index - 1
        self.mechanisms[current_index].reset_parameters()
        self.routes[current_index].reset_parameters()

    @torch.no_grad()
    def route_values(self, task_id: int) -> list[float]:
        task_index = self._task_index(task_id)
        if task_index == 0:
            return []
        logits = self.routes[task_index - 1].logits
        if logits is None:
            return []
        return torch.tanh(logits).detach().cpu().tolist()

    def parameter_report(self) -> dict[str, Any]:
        mechanism_parameters = [
            sum(parameter.numel() for parameter in mechanism.parameters())
            for mechanism in self.mechanisms
        ]
        route_parameters = [
            sum(parameter.numel() for parameter in route.parameters())
            for route in self.routes
        ]
        return {
            "kind": "task_residual_mechanism_bank",
            "num_tasks": self.num_tasks,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "hidden_features": self.hidden_features,
            "residual_scale": self.residual_scale,
            "reuse_enabled": self.reuse_enabled,
            "mechanism_parameters_per_later_task": mechanism_parameters,
            "route_parameters_per_later_task": route_parameters,
            "parameters": sum(mechanism_parameters) + sum(route_parameters),
        }
