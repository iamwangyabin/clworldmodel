"""Continual-learning mechanisms shared by world-model integrations."""

from .kan_consolidation import (
    begin_kan_importance_estimation,
    cancel_kan_importance_estimation,
    capture_kan_parameter_values,
    finish_kan_importance_estimation,
    freeze_kan_coordinate_maps,
    named_kan_residuals,
    protect_kan_parameter_updates,
)
from .evolving_core import (
    ComponentProjectionDiagnostic,
    assign_component_projected_gradients,
    assign_unprojected_current_gradients,
    atom_output_penalty,
    interface_distillation_losses,
    mechanism_output_distillation_losses,
    prediction_head_distillation_losses,
    project_component_gradients,
    recursive_python_scalars,
)
from .moe_arrow import ActorCriticBank, allocate_task_updates, shuffled_task_schedule
from .dream_rehearsal import (
    DreamRehearsalConfig,
    crossed_rehearsal_intervals,
    realized_first_scores,
    rehearsal_update_allocation,
    selected_behavior_cloning_loss,
    top_fraction_indices,
)

__all__ = [
    "begin_kan_importance_estimation",
    "cancel_kan_importance_estimation",
    "capture_kan_parameter_values",
    "finish_kan_importance_estimation",
    "freeze_kan_coordinate_maps",
    "named_kan_residuals",
    "protect_kan_parameter_updates",
    "ComponentProjectionDiagnostic",
    "assign_component_projected_gradients",
    "assign_unprojected_current_gradients",
    "atom_output_penalty",
    "interface_distillation_losses",
    "mechanism_output_distillation_losses",
    "prediction_head_distillation_losses",
    "project_component_gradients",
    "recursive_python_scalars",
    "ActorCriticBank",
    "allocate_task_updates",
    "shuffled_task_schedule",
    "DreamRehearsalConfig",
    "crossed_rehearsal_intervals",
    "realized_first_scores",
    "rehearsal_update_allocation",
    "selected_behavior_cloning_loss",
    "top_fraction_indices",
]
