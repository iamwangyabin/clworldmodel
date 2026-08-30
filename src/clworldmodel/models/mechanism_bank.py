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
        num_atoms: int = 1,
        shared_down: nn.Linear | None = None,
        hidden_film: bool = False,
    ) -> None:
        super().__init__()
        if min(in_features, out_features, hidden_features) < 1:
            raise ValueError("Mechanism dimensions must be positive")
        if residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        if num_atoms < 1:
            raise ValueError("num_atoms must be positive")
        if hidden_features % num_atoms:
            raise ValueError("hidden_features must be divisible by num_atoms")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.hidden_features = int(hidden_features)
        self.residual_scale = float(residual_scale)
        self.num_atoms = int(num_atoms)
        self.atom_width = self.hidden_features // self.num_atoms
        self.uses_shared_down = shared_down is not None
        self.hidden_film = bool(hidden_film)
        if shared_down is not None and (
            shared_down.in_features != self.in_features
            or shared_down.out_features != self.hidden_features
        ):
            raise ValueError(
                "The shared down projection must match mechanism input/hidden widths"
            )
        if self.uses_shared_down != self.hidden_film:
            raise ValueError(
                "The shared-down parameterization requires private hidden FiLM"
            )
        self.norm = nn.LayerNorm(self.in_features, eps=1e-3)
        if shared_down is None:
            self.down = nn.Linear(self.in_features, self.hidden_features)
            object.__setattr__(self, "_shared_down", None)
        else:
            # MechanismBank owns and serializes the common projection exactly
            # once. Bypass Module.__setattr__ here so this non-owning reference
            # does not duplicate state_dict entries for every task.
            self.register_module("down", None)
            object.__setattr__(self, "_shared_down", shared_down)
        if self.hidden_film:
            self.hidden_scale = nn.Parameter(torch.ones(self.hidden_features))
            self.hidden_shift = nn.Parameter(torch.zeros(self.hidden_features))
        else:
            self.register_parameter("hidden_scale", None)
            self.register_parameter("hidden_shift", None)
        self.up = nn.Linear(self.hidden_features, self.out_features)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.norm.reset_parameters()
        if self.down is not None:
            self.down.reset_parameters()
        if self.hidden_scale is not None:
            nn.init.ones_(self.hidden_scale)
        if self.hidden_shift is not None:
            nn.init.zeros_(self.hidden_shift)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def down_projection(self) -> nn.Linear:
        if self.down is not None:
            return self.down
        shared_down = self._shared_down
        if shared_down is None:
            raise RuntimeError("A shared-down mechanism lost its bank-owned projection")
        return shared_down

    def hidden_features_for(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected {self.in_features} mechanism features, "
                f"got {inputs.shape[-1]}"
            )
        hidden = self.down_projection()(self.norm(inputs))
        if self.hidden_scale is not None:
            hidden = hidden * self.hidden_scale + self.hidden_shift
        return F.silu(hidden)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.hidden_features_for(inputs)
        return self.residual_scale * self.up(hidden)

    def atom_outputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return ``[..., num_atoms, out_features]`` lossless atom outputs."""
        hidden = self.hidden_features_for(inputs).unflatten(
            -1, (self.num_atoms, self.atom_width)
        )
        weight = self.up.weight.reshape(
            self.out_features, self.num_atoms, self.atom_width
        )
        outputs = torch.einsum("...ad,oad->...ao", hidden, weight)
        outputs = outputs + self.up.bias / self.num_atoms
        return self.residual_scale * outputs

    def parameter_report(self) -> dict[str, int | float | str]:
        return {
            "kind": "zero_effect_nonlinear_residual",
            "in_features": self.in_features,
            "out_features": self.out_features,
            "hidden_features": self.hidden_features,
            "num_atoms": self.num_atoms,
            "atom_width": self.atom_width,
            "residual_scale": self.residual_scale,
            "parameterization": (
                "shared_frozen_down_film"
                if self.uses_shared_down
                else "dense_private"
            ),
            "shared_down": self.uses_shared_down,
            "hidden_film": self.hidden_film,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
        }


class ReuseRoute(nn.Module):
    """Independent tanh gates over atoms learned by earlier tasks."""

    def __init__(self, num_old_mechanisms: int, num_atoms: int = 1) -> None:
        super().__init__()
        if num_old_mechanisms < 0:
            raise ValueError("num_old_mechanisms cannot be negative")
        if num_atoms < 1:
            raise ValueError("num_atoms must be positive")
        self.num_old_mechanisms = int(num_old_mechanisms)
        self.num_atoms = int(num_atoms)
        if num_old_mechanisms == 0:
            self.register_parameter("logits", None)
        else:
            self.logits = nn.Parameter(
                torch.zeros(self.num_old_mechanisms, self.num_atoms)
            )
        self.register_buffer(
            "hard_mask",
            torch.ones(self.num_old_mechanisms, self.num_atoms),
        )
        self.register_buffer(
            "validated_shared_mask",
            torch.zeros(self.num_old_mechanisms, self.num_atoms),
        )

    def forward(self, reference: torch.Tensor) -> torch.Tensor:
        if self.logits is None:
            return reference.new_empty((0, self.num_atoms))
        # Preserve the activation dtype while keeping gradients to FP32 logits.
        return (torch.tanh(self.logits) * self.hard_mask).to(
            dtype=reference.dtype, device=reference.device
        )

    def reset_parameters(self) -> None:
        if self.logits is not None:
            nn.init.zeros_(self.logits)
        self.hard_mask.fill_(1)
        self.validated_shared_mask.zero_()

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Migrate legacy scalar gates to equivalent per-atom gates."""
        logits_key = prefix + "logits"
        if self.logits is not None and logits_key in state_dict:
            loaded_logits = state_dict[logits_key]
            if loaded_logits.shape == (self.num_old_mechanisms,):
                state_dict[logits_key] = loaded_logits.unsqueeze(-1).expand(
                    -1, self.num_atoms
                ).clone()
            elif loaded_logits.shape == (self.num_old_mechanisms, 1):
                state_dict[logits_key] = loaded_logits.expand(
                    -1, self.num_atoms
                ).clone()
        mask_key = prefix + "hard_mask"
        if mask_key not in state_dict:
            state_dict[mask_key] = torch.ones_like(self.hard_mask)
        elif state_dict[mask_key].shape == (self.num_old_mechanisms,):
            state_dict[mask_key] = state_dict[mask_key].unsqueeze(-1).expand(
                -1, self.num_atoms
            ).clone()
        elif state_dict[mask_key].shape == (self.num_old_mechanisms, 1):
            state_dict[mask_key] = state_dict[mask_key].expand(
                -1, self.num_atoms
            ).clone()
        shared_key = prefix + "validated_shared_mask"
        if shared_key not in state_dict:
            state_dict[shared_key] = torch.zeros_like(self.validated_shared_mask)
        elif state_dict[shared_key].shape == (self.num_old_mechanisms,):
            state_dict[shared_key] = state_dict[shared_key].unsqueeze(-1).expand(
                -1, self.num_atoms
            ).clone()
        elif state_dict[shared_key].shape == (self.num_old_mechanisms, 1):
            state_dict[shared_key] = state_dict[shared_key].expand(
                -1, self.num_atoms
            ).clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class MechanismBank(nn.Module):
    """Task-private residual mechanisms plus optional older-task atom reuse.

    ``include_task0=False`` preserves the frozen-Task-1 REC/MB-RSSM topology.
    ``include_task0=True`` gives every task, including Task 0, an isomorphic
    private mechanism and makes older-task reuse address all preceding tasks.
    """

    def __init__(
        self,
        *,
        num_tasks: int,
        in_features: int,
        out_features: int,
        hidden_features: int,
        residual_scale: float = 0.1,
        reuse_enabled: bool = True,
        num_atoms: int = 1,
        include_task0: bool = False,
        parameterization: str = "dense_private",
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
        self.num_atoms = int(num_atoms)
        self.include_task0 = bool(include_task0)
        if parameterization not in {
            "dense_private",
            "shared_frozen_down_film",
        }:
            raise ValueError(
                f"Unknown mechanism parameterization: {parameterization!r}"
            )
        self.parameterization = parameterization
        self.shared_down: nn.Linear | None
        if self.parameterization == "shared_frozen_down_film":
            self.shared_down = nn.Linear(self.in_features, self.hidden_features)
            self.shared_down.requires_grad_(False)
        else:
            self.shared_down = None
        self._recording_task_id: int | None = None
        self._recorded_atom_norm_sum: torch.Tensor | None = None
        self._recorded_correction_norm_sum: torch.Tensor | None = None
        self._recorded_value_count = 0
        self.mechanisms = nn.ModuleList(
            ResidualMechanism(
                in_features=self.in_features,
                out_features=self.out_features,
                hidden_features=self.hidden_features,
                residual_scale=self.residual_scale,
                num_atoms=self.num_atoms,
                shared_down=self.shared_down,
                hidden_film=self.shared_down is not None,
            )
            for _ in range(
                self.num_tasks if self.include_task0 else self.num_tasks - 1
            )
        )
        self.routes = nn.ModuleList(
            ReuseRoute(
                num_old_mechanisms=(
                    task_id if self.include_task0 else task_id - 1
                ),
                num_atoms=self.num_atoms,
            )
            for task_id in range(
                0 if self.include_task0 else 1, self.num_tasks
            )
        )

    def _task_index(self, task_id: int) -> int:
        if not isinstance(task_id, int):
            raise TypeError("Mechanism-bank task_id must be an integer")
        if not 0 <= task_id < self.num_tasks:
            raise ValueError(f"Invalid mechanism-bank task_id: {task_id}")
        return task_id

    def _mechanism_index(self, task_id: int) -> int | None:
        task_index = self._task_index(task_id)
        if task_index == 0 and not self.include_task0:
            return None
        return task_index if self.include_task0 else task_index - 1

    def _route_index(self, task_id: int) -> int | None:
        task_index = self._task_index(task_id)
        if task_index == 0 and not self.include_task0:
            return None
        return task_index if self.include_task0 else task_index - 1

    def mechanism_for(self, task_id: int) -> ResidualMechanism | None:
        """Return the private mechanism owned by ``task_id``, if one exists."""

        index = self._mechanism_index(task_id)
        return None if index is None else self.mechanisms[index]

    def route_for(self, task_id: int) -> ReuseRoute | None:
        """Return the reuse route owned by ``task_id``, if one exists."""

        index = self._route_index(task_id)
        return None if index is None else self.routes[index]

    def forward(self, inputs: torch.Tensor, task_id: int) -> torch.Tensor:
        correction, _current = self.forward_with_current(inputs, task_id)
        return correction

    def forward_with_current(
        self, inputs: torch.Tensor, task_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the full routed correction and the unscaled current-task path."""

        task_index = self._task_index(task_id)
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected {self.in_features} mechanism-bank features, "
                f"got {inputs.shape[-1]}"
            )
        output_shape = (*inputs.shape[:-1], self.out_features)
        current_index = self._mechanism_index(task_index)
        if current_index is None:
            zeros = inputs.new_zeros(output_shape)
            return zeros, zeros

        current_output = self.mechanisms[current_index](inputs)
        correction = current_output
        recorded_atom_norms = None
        if self.reuse_enabled:
            route_index = self._route_index(task_index)
            if route_index is None:
                raise RuntimeError("A task mechanism is missing its reuse route")
            gates = self.routes[route_index](inputs)
            if self._recording_task_id == task_index:
                recorded_atom_norms = inputs.new_zeros(
                    gates.shape[0], self.num_atoms, dtype=torch.float64
                )
            for mechanism_index, atom_gates in enumerate(gates.unbind(0)):
                old_atoms = self.mechanisms[mechanism_index].atom_outputs(inputs)
                weighted_atoms = old_atoms * atom_gates.reshape(
                    *((1,) * (old_atoms.ndim - 2)), self.num_atoms, 1
                )
                correction = correction + weighted_atoms.sum(dim=-2)
                if recorded_atom_norms is not None:
                    norms = torch.linalg.vector_norm(
                        weighted_atoms.detach().float(), dim=-1
                    )
                    recorded_atom_norms[mechanism_index] = norms.reshape(
                        -1, self.num_atoms
                    ).double().sum(dim=0)
        if recorded_atom_norms is not None:
            correction_norms = torch.linalg.vector_norm(
                correction.detach().float(), dim=-1
            )
            if self._recorded_atom_norm_sum is None:
                self._recorded_atom_norm_sum = recorded_atom_norms
                self._recorded_correction_norm_sum = correction_norms.double().sum()
            else:
                self._recorded_atom_norm_sum.add_(recorded_atom_norms)
                assert self._recorded_correction_norm_sum is not None
                self._recorded_correction_norm_sum.add_(
                    correction_norms.double().sum()
                )
            self._recorded_value_count += correction_norms.numel()
        return correction, current_output

    def activate_task(self, task_id: int, phase: str = "full") -> None:
        """Expose only the selected task's new mechanism and reuse gates."""
        task_index = self._task_index(task_id)
        if phase not in {"full", "reuse_probe"}:
            raise ValueError(f"Unknown mechanism phase: {phase!r}")
        self.requires_grad_(False)
        if self.shared_down is not None:
            self.shared_down.requires_grad_(False)
        current_index = self._mechanism_index(task_index)
        if current_index is None:
            return
        self.mechanisms[current_index].requires_grad_(phase == "full")
        if self.reuse_enabled:
            route_index = self._route_index(task_index)
            if route_index is None:
                raise RuntimeError("A task mechanism is missing its reuse route")
            self.routes[route_index].requires_grad_(True)

    def reset_task(self, task_id: int) -> None:
        task_index = self._task_index(task_id)
        current_index = self._mechanism_index(task_index)
        if current_index is None:
            return
        self.mechanisms[current_index].reset_parameters()
        route_index = self._route_index(task_index)
        if route_index is None:
            raise RuntimeError("A task mechanism is missing its reuse route")
        self.routes[route_index].reset_parameters()

    @torch.no_grad()
    def route_values(self, task_id: int) -> list[float] | list[list[float]]:
        task_index = self._task_index(task_id)
        route_index = self._route_index(task_index)
        if route_index is None:
            return []
        logits = self.routes[route_index].logits
        if logits is None:
            return []
        values = torch.tanh(logits) * self.routes[route_index].hard_mask
        values_list = values.detach().cpu().tolist()
        if self.num_atoms == 1:
            return [row[0] for row in values_list]
        return values_list

    @torch.no_grad()
    def apply_consolidated_mask(
        self, task_id: int, mask: torch.Tensor
    ) -> torch.Tensor:
        task_index = self._task_index(task_id)
        route_index = self._route_index(task_index)
        if route_index is None or self.routes[route_index].logits is None:
            raise ValueError("Task 0 has no reusable mechanism route")
        route = self.routes[route_index]
        expected_shape = (
            task_index if self.include_task0 else task_index - 1,
            self.num_atoms,
        )
        if tuple(mask.shape) != expected_shape:
            raise ValueError(
                f"Expected consolidated mask shape {expected_shape}, got {tuple(mask.shape)}"
            )
        previous = route.hard_mask.detach().clone()
        route.hard_mask.copy_(
            mask.to(device=route.hard_mask.device, dtype=route.hard_mask.dtype)
        )
        return previous

    def begin_contribution_recording(self, task_id: int) -> None:
        task_index = self._task_index(task_id)
        minimum_task = 1 if self.include_task0 else 2
        if task_index < minimum_task:
            raise ValueError("Atom reuse contributions require at least one old mechanism")
        if self._recording_task_id is not None:
            raise RuntimeError("Mechanism contribution recording is already active")
        self._recording_task_id = task_index
        self._recorded_atom_norm_sum = None
        self._recorded_correction_norm_sum = None
        self._recorded_value_count = 0

    @torch.no_grad()
    def apply_validated_shared_mask(
        self, task_id: int, mask: torch.Tensor
    ) -> torch.Tensor:
        task_index = self._task_index(task_id)
        route_index = self._route_index(task_index)
        if route_index is None or self.routes[route_index].logits is None:
            raise ValueError("Task 0 has no reusable mechanism route")
        route = self.routes[route_index]
        expected_shape = (
            task_index if self.include_task0 else task_index - 1,
            self.num_atoms,
        )
        if tuple(mask.shape) != expected_shape:
            raise ValueError(
                f"Expected validated-shared mask shape {expected_shape}, "
                f"got {tuple(mask.shape)}"
            )
        normalized = mask.to(
            device=route.validated_shared_mask.device,
            dtype=route.validated_shared_mask.dtype,
        )
        if bool((normalized < 0).any().item()) or bool(
            (normalized > route.hard_mask).any().item()
        ):
            raise ValueError("Validated-shared atoms must remain enabled")
        previous = route.validated_shared_mask.detach().clone()
        route.validated_shared_mask.copy_(normalized)
        return previous

    @torch.no_grad()
    def finish_contribution_recording(self) -> dict[str, Any]:
        if self._recording_task_id is None:
            raise RuntimeError("Mechanism contribution recording is not active")
        task_id = self._recording_task_id
        atom_sum = self._recorded_atom_norm_sum
        correction_sum = self._recorded_correction_norm_sum
        count = self._recorded_value_count
        self._recording_task_id = None
        self._recorded_atom_norm_sum = None
        self._recorded_correction_norm_sum = None
        self._recorded_value_count = 0
        if atom_sum is None or correction_sum is None or count == 0:
            raise RuntimeError("No mechanism contributions were recorded")
        denominator = correction_sum.clamp_min(torch.finfo(torch.float64).eps)
        return {
            "task_id": task_id,
            "value_count": count,
            "atom_norm_sum": atom_sum.detach().cpu().tolist(),
            "correction_norm_sum": float(correction_sum.detach().cpu()),
            "contribution_ratio": (atom_sum / denominator).detach().cpu().tolist(),
        }

    def cancel_contribution_recording(self) -> None:
        self._recording_task_id = None
        self._recorded_atom_norm_sum = None
        self._recorded_correction_norm_sum = None
        self._recorded_value_count = 0

    @torch.no_grad()
    def route_manifest(self, completed_through_task_id: int) -> dict[str, Any]:
        completed_task = self._task_index(completed_through_task_id)
        routes = []
        first_route_task = 0 if self.include_task0 else 1
        for task_id in range(first_route_task, completed_task + 1):
            route_index = self._route_index(task_id)
            if route_index is None:
                continue
            route = self.routes[route_index]
            routes.append(
                {
                    "task_id": task_id,
                    "gates": self.route_values(task_id),
                    "hard_mask": route.hard_mask.detach().cpu().tolist(),
                    "validated_shared_mask": (
                        route.validated_shared_mask.detach().cpu().tolist()
                    ),
                }
            )
        atoms = []
        completed_owner_count = completed_task + int(self.include_task0)
        for owner_index in range(min(completed_owner_count, len(self.mechanisms))):
            owner_task = owner_index if self.include_task0 else owner_index + 1
            for atom_index in range(self.num_atoms):
                users = [owner_task]
                for user_task in range(owner_task + 1, completed_task + 1):
                    route_index = self._route_index(user_task)
                    if route_index is None:
                        raise RuntimeError("A routed task is missing its route")
                    route = self.routes[route_index]
                    if bool(
                        route.validated_shared_mask[
                            owner_index, atom_index
                        ].item()
                    ):
                        users.append(user_task)
                atoms.append(
                    {
                        "owner_task": owner_task,
                        "atom_index": atom_index,
                        "users": users,
                        "status": "shared" if len(users) > 1 else "private",
                    }
                )
        return {"routes": routes, "atoms": atoms}

    def parameter_report(self) -> dict[str, Any]:
        mechanism_parameters = [
            sum(parameter.numel() for parameter in mechanism.parameters())
            for mechanism in self.mechanisms
        ]
        route_parameters = [
            sum(parameter.numel() for parameter in route.parameters())
            for route in self.routes
        ]
        shared_down_parameters = (
            0
            if self.shared_down is None
            else sum(parameter.numel() for parameter in self.shared_down.parameters())
        )
        return {
            "kind": "task_residual_mechanism_bank",
            "num_tasks": self.num_tasks,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "hidden_features": self.hidden_features,
            "num_atoms": self.num_atoms,
            "atom_width": self.hidden_features // self.num_atoms,
            "residual_scale": self.residual_scale,
            "reuse_enabled": self.reuse_enabled,
            "include_task0": self.include_task0,
            "parameterization": self.parameterization,
            "shared_down_parameters": shared_down_parameters,
            "shared_down_trainable_parameters": (
                0
                if self.shared_down is None
                else sum(
                    parameter.numel()
                    for parameter in self.shared_down.parameters()
                    if parameter.requires_grad
                )
            ),
            "mechanism_parameters_per_task": mechanism_parameters,
            "mechanism_parameters_per_later_task": mechanism_parameters,
            "route_parameters_per_later_task": route_parameters,
            "parameters": (
                shared_down_parameters
                + sum(mechanism_parameters)
                + sum(route_parameters)
            ),
        }
