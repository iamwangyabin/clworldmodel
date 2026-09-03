"""Shared categorical heads with task-routed adaptive residual capacity.

The shared logits network is stored once.  Every task owns a zero-effect
nonlinear residual mechanism whose hidden channels can be physically removed
after acquisition.  Task identity is supplied by the named task-aware protocol;
it is not inferred from observations inside this module.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from .mechanism_bank import MechanismBank, ResidualMechanism


class TaskRoutedResidualCategoricalHead(nn.Module):
    """One shared logits function plus one adaptive residual per task."""

    def __init__(
        self,
        base_logits: nn.Module,
        *,
        in_features: int,
        out_features: int,
        num_tasks: int,
        hidden_features: int,
        residual_scale: float,
        num_atoms: int,
        reuse_enabled: bool,
    ) -> None:
        super().__init__()
        if not isinstance(base_logits, nn.Module):
            raise TypeError("Shared categorical logits must be a torch module")
        if min(in_features, out_features, hidden_features) < 1:
            raise ValueError("Shared behavior dimensions must be positive")
        if num_tasks < 2:
            raise ValueError("Task-routed behavior requires at least two tasks")
        if num_atoms < 1 or hidden_features % num_atoms:
            raise ValueError(
                "Shared behavior hidden width must be divisible by its atom count"
            )
        self.base_logits = base_logits
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_tasks = int(num_tasks)
        self.hidden_features = int(hidden_features)
        self.residual_scale = float(residual_scale)
        self.num_atoms = int(num_atoms)
        self.reuse_enabled = bool(reuse_enabled)
        self.residual_bank = MechanismBank(
            num_tasks=self.num_tasks,
            in_features=self.in_features,
            out_features=self.out_features,
            hidden_features=self.hidden_features,
            residual_scale=self.residual_scale,
            reuse_enabled=self.reuse_enabled,
            num_atoms=self.num_atoms,
            include_task0=True,
            parameterization="adaptive_dense_width",
        )
        # The route is orchestration state, not a learned tensor. Keeping it as
        # a Python integer avoids a CUDA ``Tensor.item()`` synchronization in
        # every imagined Actor/Critic forward. Checkpoints restore the current
        # scheduler task explicitly before collection or optimization.
        self._active_task_id = 0

    def set_task_route(self, task_id: int) -> None:
        """Select the explicit task route without changing trainability."""

        if not isinstance(task_id, int):
            raise TypeError("Behavior task route must be an integer")
        if not 0 <= task_id < self.num_tasks:
            raise ValueError(f"Invalid behavior task route: {task_id}")
        self._active_task_id = task_id

    def activate_training_task(self, task_id: int) -> None:
        """Train the shared base and only the acquiring task's private path."""

        self.set_task_route(task_id)
        self.base_logits.requires_grad_(True)
        self.residual_bank.activate_task(task_id, phase="full")

    def shared_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.base_logits.parameters())

    def private_parameters(self, task_id: int) -> tuple[nn.Parameter, ...]:
        mechanism = self.residual_bank.mechanism_for(task_id)
        if not isinstance(mechanism, nn.Module):
            raise RuntimeError(f"Task {task_id} has no private behavior mechanism")
        return tuple(mechanism.parameters())

    def route_parameters(self, task_id: int) -> tuple[nn.Parameter, ...]:
        route = self.residual_bank.route_for(task_id)
        return () if route is None else tuple(route.parameters())

    def compression_layout(self) -> list[int]:
        return self.residual_bank.compression_layout()

    def mechanism_for(self, task_id: int) -> ResidualMechanism:
        mechanism = self.residual_bank.mechanism_for(task_id)
        if not isinstance(mechanism, ResidualMechanism):
            raise TypeError("Adaptive behavior requires a dense residual mechanism")
        return mechanism

    def install_task_mechanism(
        self, task_id: int, mechanism: ResidualMechanism
    ) -> dict[str, object]:
        return self.residual_bank.install_task_mechanism(task_id, mechanism)

    def forward_logits(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected {self.in_features} behavior features, "
                f"got {inputs.shape[-1]}"
            )
        task_id = self._active_task_id
        logits = self.base_logits(inputs)
        if logits.shape[-1] != self.out_features:
            raise RuntimeError(
                "Shared behavior base changed its categorical output width"
            )
        return logits + self.residual_bank(inputs, task_id)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.forward_logits(inputs), dim=-1)

    def parameter_report(self) -> dict[str, object]:
        shared = sum(parameter.numel() for parameter in self.base_logits.parameters())
        bank = self.residual_bank.parameter_report()
        return {
            "kind": "shared_logits_plus_task_adaptive_residuals",
            "active_task_id": self._active_task_id,
            "shared_parameters": shared,
            "residual_bank": bank,
            "parameters": shared + int(bank["parameters"]),
        }


def unique_parameters(groups: Iterable[Iterable[nn.Parameter]]) -> list[nn.Parameter]:
    """Flatten parameter groups while preserving ownership exactly once."""

    result: list[nn.Parameter] = []
    seen: set[int] = set()
    for group in groups:
        for parameter in group:
            if id(parameter) not in seen:
                result.append(parameter)
                seen.add(id(parameter))
    return result
