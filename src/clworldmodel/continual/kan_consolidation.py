"""Replay-guided consolidation utilities for fixed-grid KAN residuals."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from ..models.residual_corrections import ResidualCorrection


def named_kan_residuals(
    roots: Mapping[str, nn.Module] | nn.Module,
) -> dict[str, ResidualCorrection]:
    """Return every independent KAN residual exactly once with a stable name."""
    if isinstance(roots, nn.Module):
        roots = {"model": roots}
    result: dict[str, ResidualCorrection] = {}
    seen: set[int] = set()
    for root_name, root in roots.items():
        for module_name, module in root.named_modules():
            if not isinstance(module, ResidualCorrection) or module.kind != "kan":
                continue
            if id(module) in seen:
                continue
            seen.add(id(module))
            suffix = f".{module_name}" if module_name else ""
            result[f"{root_name}{suffix}"] = module
    return result


def begin_kan_importance_estimation(
    roots: Mapping[str, nn.Module] | nn.Module,
) -> dict[str, ResidualCorrection]:
    residuals = named_kan_residuals(roots)
    if not residuals:
        raise ValueError("Replay consolidation found no KAN residual corrections")
    for residual in residuals.values():
        residual.begin_importance_estimation()
    return residuals


def cancel_kan_importance_estimation(
    residuals: Mapping[str, ResidualCorrection],
) -> None:
    for residual in residuals.values():
        residual.cancel_importance_estimation()


def finish_kan_importance_estimation(
    residuals: Mapping[str, ResidualCorrection],
    *,
    gradient_power: float,
    min_plasticity: float,
    anchor_loss_scale: float,
) -> dict[str, dict[str, float | int]]:
    return {
        name: residual.finish_importance_estimation(
            gradient_power=gradient_power,
            min_plasticity=min_plasticity,
            anchor_loss_scale=anchor_loss_scale,
        )
        for name, residual in residuals.items()
    }


def freeze_kan_coordinate_maps(
    roots: Mapping[str, nn.Module] | nn.Module,
) -> dict[str, ResidualCorrection]:
    residuals = named_kan_residuals(roots)
    if not residuals:
        raise ValueError("Coordinate freezing found no KAN residual corrections")
    for residual in residuals.values():
        residual.freeze_coordinate_map()
    return residuals


@torch.no_grad()
def capture_kan_parameter_values(
    roots: Mapping[str, nn.Module] | nn.Module,
) -> dict[str, torch.Tensor]:
    """Snapshot RBF values immediately before an adaptive-optimizer step."""
    residuals = named_kan_residuals(roots)
    if not residuals:
        raise ValueError("Update protection found no KAN residual corrections")
    return {
        name: residual.core.rbf_weight.detach().clone()
        for name, residual in residuals.items()
    }


@torch.no_grad()
def protect_kan_parameter_updates(
    roots: Mapping[str, nn.Module] | nn.Module,
    before_step: Mapping[str, torch.Tensor],
) -> None:
    """Scale the realized optimizer delta, which remains effective under Adam."""
    residuals = named_kan_residuals(roots)
    if set(residuals) != set(before_step):
        raise ValueError("KAN update snapshot does not match the current residual set")
    for name, residual in residuals.items():
        if not residual.consolidation_gradient_scale.numel():
            raise RuntimeError("KAN update protection has not been initialized")
        weight = residual.core.rbf_weight
        previous = before_step[name].to(device=weight.device, dtype=weight.dtype)
        scale = residual.consolidation_gradient_scale.to(
            device=weight.device,
            dtype=weight.dtype,
        )
        weight.copy_(previous + scale * (weight - previous))


def kan_consolidation_penalty(module: nn.Module) -> torch.Tensor:
    """Sum each adapter's anchor penalty once per optimization objective."""
    residuals = named_kan_residuals(module)
    if residuals:
        return torch.stack(
            [residual.consolidation_penalty() for residual in residuals.values()]
        ).sum()
    parameter = next(module.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    return torch.zeros((), device=device)
