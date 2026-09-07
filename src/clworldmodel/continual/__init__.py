"""Continual-learning mechanisms shared by world-model integrations."""

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
from .task_replay import ActorCriticBank, allocate_task_updates, shuffled_task_schedule
from .dream_rehearsal import (
    DreamRehearsalConfig,
    crossed_rehearsal_intervals,
    realized_first_scores,
    rehearsal_update_allocation,
    selected_behavior_cloning_loss,
    top_fraction_indices,
)

__all__ = [
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
