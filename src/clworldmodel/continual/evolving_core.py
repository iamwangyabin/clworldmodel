"""Optimization primitives for the Evolving-Core Atomic RSSM protocol.

The vendored trainer owns orchestration, while the behavior-changing gradient
and interface objectives live in project-owned code.  Every public function is
independent of the Atari environment and operates on explicit tensors or
parameter groups.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ComponentProjectionDiagnostic:
    """Summary of one shared component's stability/plasticity allocation."""

    dot_product: float
    current_norm: float
    memory_norm: float
    projected_current_norm: float
    conflicted: bool


def _zeros_like_parameter(parameter: torch.nn.Parameter) -> torch.Tensor:
    return torch.zeros_like(parameter, memory_format=torch.preserve_format)


def _gradient_in_parameter_layout(
    parameter: torch.nn.Parameter,
    gradient: torch.Tensor,
) -> torch.Tensor:
    """Copy a gradient into the parameter's exact dense memory layout.

    ``autograd.grad`` may return convolution gradients with channels-last
    strides even when the parameter and Adam state use contiguous strides.
    PyTorch's fused Adam requires corresponding parameter, gradient, and state
    tensors to have the same layout, so assigning the raw tensor is unsafe.
    """

    if gradient.shape != parameter.shape:
        raise ValueError("A parameter gradient has an unexpected shape")
    materialized = torch.empty_like(
        parameter,
        memory_format=torch.preserve_format,
    )
    materialized.copy_(gradient)
    return materialized


def _materialize_gradients(
    parameters: Sequence[torch.nn.Parameter],
    gradients: Sequence[torch.Tensor | None],
) -> tuple[torch.Tensor, ...]:
    if len(parameters) != len(gradients):
        raise ValueError("Parameters and gradients must have equal length")
    return tuple(
        _zeros_like_parameter(parameter) if gradient is None else gradient
        for parameter, gradient in zip(parameters, gradients)
    )


def project_component_gradients(
    current_gradients: Sequence[torch.Tensor],
    memory_gradients: Sequence[torch.Tensor],
    *,
    memory_scale: float,
    epsilon: float = 1e-12,
    materialize_diagnostic: bool = True,
) -> tuple[
    tuple[torch.Tensor, ...],
    ComponentProjectionDiagnostic | None,
]:
    """Project a current-task gradient against one component's memory gradient.

    The dot product and projection coefficient are computed jointly over every
    parameter in the named component.  This preserves the method definition:
    encoder, recurrent, posterior, prior, and latent-interface gradients are
    protected independently rather than flattening the whole model into one
    vector.
    """

    if len(current_gradients) != len(memory_gradients):
        raise ValueError("Current and memory gradient groups must have equal length")
    if not current_gradients:
        raise ValueError("A shared component must own at least one parameter")
    if memory_scale < 0:
        raise ValueError("memory_scale must be non-negative")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    with torch.no_grad():
        dot = sum(
            (current.detach().float() * memory.detach().float()).sum()
            for current, memory in zip(current_gradients, memory_gradients)
        )
        memory_squared_norm = sum(
            memory.detach().float().square().sum()
            for memory in memory_gradients
        )
        coefficient = torch.where(
            dot < 0,
            dot / (memory_squared_norm + epsilon),
            dot.new_zeros(()),
        )

    projected_current = tuple(
        current - coefficient.to(device=current.device, dtype=current.dtype) * memory
        for current, memory in zip(current_gradients, memory_gradients)
    )
    combined = tuple(
        projected
        + memory_scale * memory.to(device=projected.device, dtype=projected.dtype)
        for projected, memory in zip(projected_current, memory_gradients)
    )
    diagnostic = None
    if materialize_diagnostic:
        with torch.no_grad():
            current_squared_norm = sum(
                current.detach().float().square().sum()
                for current in current_gradients
            )
            projected_squared_norm = sum(
                value.detach().float().square().sum()
                for value in projected_current
            )
            diagnostic_values = torch.stack(
                (
                    dot,
                    current_squared_norm.sqrt(),
                    memory_squared_norm.sqrt(),
                    projected_squared_norm.sqrt(),
                )
            ).detach().cpu().tolist()
        diagnostic = ComponentProjectionDiagnostic(
            dot_product=diagnostic_values[0],
            current_norm=diagnostic_values[1],
            memory_norm=diagnostic_values[2],
            projected_current_norm=diagnostic_values[3],
            conflicted=diagnostic_values[0] < 0.0,
        )
    return combined, diagnostic


def assign_component_projected_gradients(
    *,
    current_loss: torch.Tensor,
    memory_loss: torch.Tensor,
    shared_parameter_groups: Mapping[str, Sequence[torch.nn.Parameter]],
    private_parameters: Sequence[torch.nn.Parameter],
    memory_scale: float,
    project_conflicts: bool = True,
    epsilon: float = 1e-12,
    materialize_diagnostics: bool = True,
) -> dict[str, ComponentProjectionDiagnostic]:
    """Differentiate two losses and assign optimizer-ready ``.grad`` tensors.

    Current private modules receive their complete current-task gradient.  Only
    shared groups are combined with the memory gradient, and projection is
    performed independently for each named component.  Parameters must be
    uniquely owned; accidental optimizer/group overlap is rejected.
    """

    if current_loss.ndim != 0 or memory_loss.ndim != 0:
        raise ValueError("Current and memory losses must be scalar tensors")
    if memory_scale < 0:
        raise ValueError("memory_scale must be non-negative")

    ordered_groups: list[tuple[str, tuple[torch.nn.Parameter, ...]]] = []
    shared_parameters: list[torch.nn.Parameter] = []
    seen_ids: set[int] = set()
    for name, values in shared_parameter_groups.items():
        parameters = tuple(values)
        for parameter in parameters:
            if id(parameter) in seen_ids:
                raise ValueError("Shared parameter groups overlap")
            seen_ids.add(id(parameter))
        ordered_groups.append((name, parameters))
        shared_parameters.extend(parameters)

    private = tuple(private_parameters)
    for parameter in private:
        if id(parameter) in seen_ids:
            raise ValueError("Shared and private parameter ownership overlaps")
        seen_ids.add(id(parameter))

    current_targets = tuple(shared_parameters) + private
    current_raw = torch.autograd.grad(
        current_loss,
        current_targets,
        retain_graph=True,
        allow_unused=True,
    )
    current_gradients = _materialize_gradients(current_targets, current_raw)
    shared_current = current_gradients[: len(shared_parameters)]
    private_current = current_gradients[len(shared_parameters) :]
    memory_raw = torch.autograd.grad(
        memory_loss,
        tuple(shared_parameters),
        retain_graph=False,
        allow_unused=True,
    )
    shared_memory = _materialize_gradients(shared_parameters, memory_raw)

    diagnostics: dict[str, ComponentProjectionDiagnostic] = {}
    offset = 0
    for name, parameters in ordered_groups:
        stop = offset + len(parameters)
        current_group = shared_current[offset:stop]
        memory_group = shared_memory[offset:stop]
        if not parameters:
            if materialize_diagnostics:
                diagnostics[name] = ComponentProjectionDiagnostic(
                    dot_product=0.0,
                    current_norm=0.0,
                    memory_norm=0.0,
                    projected_current_norm=0.0,
                    conflicted=False,
                )
            offset = stop
            continue
        if project_conflicts:
            combined, diagnostic = project_component_gradients(
                current_group,
                memory_group,
                memory_scale=memory_scale,
                epsilon=epsilon,
                materialize_diagnostic=materialize_diagnostics,
            )
        else:
            combined = tuple(
                current + memory_scale * memory
                for current, memory in zip(current_group, memory_group)
            )
            diagnostic = None
            if materialize_diagnostics:
                with torch.no_grad():
                    dot = sum(
                        (current.detach().float() * memory.detach().float()).sum()
                        for current, memory in zip(current_group, memory_group)
                    )
                    current_norm = sum(
                        current.detach().float().square().sum()
                        for current in current_group
                    ).sqrt()
                    memory_norm = sum(
                        memory.detach().float().square().sum()
                        for memory in memory_group
                    ).sqrt()
                    diagnostic_values = torch.stack(
                        (dot, current_norm, memory_norm)
                    ).detach().cpu().tolist()
                diagnostic = ComponentProjectionDiagnostic(
                    dot_product=diagnostic_values[0],
                    current_norm=diagnostic_values[1],
                    memory_norm=diagnostic_values[2],
                    projected_current_norm=diagnostic_values[1],
                    conflicted=diagnostic_values[0] < 0.0,
                )
        for parameter, gradient in zip(parameters, combined):
            parameter.grad = _gradient_in_parameter_layout(
                parameter,
                gradient.detach(),
            )
        if diagnostic is not None:
            diagnostics[name] = diagnostic
        offset = stop

    for parameter, gradient in zip(private, private_current):
        parameter.grad = _gradient_in_parameter_layout(
            parameter,
            gradient.detach(),
        )
    return diagnostics


def prediction_head_distillation_losses(
    student_outputs: Mapping[str, torch.Tensor],
    teacher_outputs: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Match one shared prediction-head student to a frozen boundary teacher.

    The observation and symlog-reward outputs use mean squared error.  The
    continuation term is the Bernoulli KL from the stopped teacher probability
    to the student logit.  Mean reductions keep the distillation scale
    independent of Atari image size and minibatch shape; the ordinary old-task
    Dreamer loss still supplies the unchanged pixel-summed reconstruction
    objective against real LTDM observations.
    """

    required = {"observation", "reward_symlog", "continue_logits"}
    if set(student_outputs) != required or set(teacher_outputs) != required:
        raise ValueError(
            "Prediction-head outputs must contain exactly observation, "
            "reward_symlog, and continue_logits"
        )
    for name in required:
        if student_outputs[name].shape != teacher_outputs[name].shape:
            raise ValueError(
                f"Prediction-head student/teacher shape mismatch for {name}"
            )

    observation = F.mse_loss(
        student_outputs["observation"].float(),
        teacher_outputs["observation"].detach().float(),
        reduction="mean",
    )
    reward = F.mse_loss(
        student_outputs["reward_symlog"].float(),
        teacher_outputs["reward_symlog"].detach().float(),
        reduction="mean",
    )
    student_continue = student_outputs["continue_logits"].float()
    teacher_continue = teacher_outputs["continue_logits"].detach().float()
    teacher_probability = torch.sigmoid(teacher_continue)
    cross_entropy = F.binary_cross_entropy_with_logits(
        student_continue,
        teacher_probability,
        reduction="none",
    )
    teacher_entropy = F.binary_cross_entropy_with_logits(
        teacher_continue,
        teacher_probability,
        reduction="none",
    )
    continuation = (cross_entropy - teacher_entropy).mean().clamp_min(0.0)
    return {
        "observation": observation,
        "reward": reward,
        "continue": continuation,
        "total": observation + reward + continuation,
    }


def assign_unprojected_current_gradients(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
) -> None:
    """Assign full Task-1/current-only gradients without rebuilding optimizers."""

    owned = tuple(parameters)
    if loss.ndim != 0:
        raise ValueError("Loss must be scalar")
    if not owned:
        raise ValueError("At least one parameter is required")
    raw = torch.autograd.grad(loss, owned, allow_unused=True)
    for parameter, gradient in zip(owned, _materialize_gradients(owned, raw)):
        parameter.grad = _gradient_in_parameter_layout(
            parameter,
            gradient.detach(),
        )


def atom_output_penalty(trace: Mapping[str, Any]) -> torch.Tensor:
    """Return the sum of mean-squared current-atom outputs in an RSSM trace."""

    outputs = trace.get("current_atom_outputs")
    if not isinstance(outputs, Mapping) or not outputs:
        raise ValueError("Trace does not contain current atom outputs")
    penalties = []
    for component in ("recurrent", "posterior", "prior"):
        value = outputs.get(component)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Trace is missing {component!r} atom outputs")
        # The protocol specifies E[||A_i^c(x)||_2^2], not an elementwise MSE.
        penalties.append(value.float().square().sum(dim=-1).mean())
    return torch.stack(penalties).sum()


def _categorical_kl_from_log_probs(
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
) -> torch.Tensor:
    if teacher_log_probs.shape != student_log_probs.shape:
        raise ValueError("Teacher and student categorical tensors must match")
    teacher = teacher_log_probs.detach().float()
    student = student_log_probs.float()
    return (teacher.exp() * (teacher - student)).sum(dim=-1).mean()


def interface_distillation_losses(
    *,
    student_trace: Mapping[str, torch.Tensor],
    teacher_trace: Mapping[str, torch.Tensor],
    frozen_actor: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """Protect posterior, recurrent state, and old-policy latent interfaces."""

    required = ("posterior_log_probs", "hiddens", "actor_states")
    missing = [name for name in required if name not in student_trace]
    missing += [name for name in required if name not in teacher_trace]
    if missing:
        raise ValueError(f"Incomplete interface traces: {sorted(set(missing))}")

    posterior = _categorical_kl_from_log_probs(
        teacher_trace["posterior_log_probs"],
        student_trace["posterior_log_probs"],
    )
    teacher_h = F.layer_norm(
        teacher_trace["hiddens"].detach().float(),
        (teacher_trace["hiddens"].shape[-1],),
    )
    student_h = F.layer_norm(
        student_trace["hiddens"].float(),
        (student_trace["hiddens"].shape[-1],),
    )
    hidden = (teacher_h - student_h).square().mean()

    teacher_states = teacher_trace["actor_states"].detach()
    student_states = student_trace["actor_states"]
    with torch.no_grad():
        teacher_policy = frozen_actor(teacher_states).float()
    student_policy = frozen_actor(student_states).float()
    actor = _categorical_kl_from_log_probs(teacher_policy, student_policy)
    return {"posterior": posterior, "hidden": hidden, "actor": actor}


def recursive_python_scalars(value: Any) -> Any:
    """Convert NumPy/torch scalar leaves before atomic JSON serialization."""

    if isinstance(value, Mapping):
        return {str(key): recursive_python_scalars(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [recursive_python_scalars(item) for item in value]
    if isinstance(value, list):
        return [recursive_python_scalars(item) for item in value]
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except ModuleNotFoundError:  # pragma: no cover - NumPy is an Atari dependency.
        pass
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.detach().cpu().item()
    return value
