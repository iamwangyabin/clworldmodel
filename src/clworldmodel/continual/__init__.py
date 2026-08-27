"""Continual-learning mechanisms shared by world-model integrations."""

from .kan_consolidation import (
    begin_kan_importance_estimation,
    cancel_kan_importance_estimation,
    capture_kan_parameter_values,
    finish_kan_importance_estimation,
    freeze_kan_coordinate_maps,
    kan_consolidation_penalty,
    named_kan_residuals,
    protect_kan_parameter_updates,
)
from .moe_arrow import ActorCriticBank, allocate_task_updates, shuffled_task_schedule

__all__ = [
    "begin_kan_importance_estimation",
    "cancel_kan_importance_estimation",
    "capture_kan_parameter_values",
    "finish_kan_importance_estimation",
    "freeze_kan_coordinate_maps",
    "kan_consolidation_penalty",
    "named_kan_residuals",
    "protect_kan_parameter_updates",
    "ActorCriticBank",
    "allocate_task_updates",
    "shuffled_task_schedule",
]
