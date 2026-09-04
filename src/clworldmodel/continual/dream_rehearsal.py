"""Bounded, framework-independent primitives for graded dream rehearsal.

The scoring rule follows Nijjer, *The World Model Remembers, the Actor
Forgets* (arXiv:2607.19749, 2026).  Replay retention and task scheduling stay
outside this module so the method can be composed with a fixed-capacity replay
policy instead of inheriting the paper artifact's never-clear buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


def _require_tensor_condition(condition: torch.Tensor, message: str) -> None:
    """Keep tensor validation asynchronous on the healthy CUDA path."""

    if condition.device.type == "cuda" and hasattr(torch, "_assert_async"):
        torch._assert_async(condition)
        return
    if not bool(condition):
        raise ValueError(message)


@dataclass(frozen=True)
class DreamRehearsalConfig:
    """Validated algorithm constants for one bounded-replay port.

    ``interval_agent_decisions`` and ``updates_per_prior_task`` reproduce the
    paper's update-rate definition.  A trainer whose collection unit is larger
    than the interval may execute all newly due updates at its next optimizer
    boundary, but must preserve and report the exact number of crossed
    intervals.
    """

    interval_agent_decisions: int = 2_000
    updates_per_prior_task: int = 50
    batch_sequences: int = 4
    context_steps: int = 16
    horizon: int = 15
    top_fraction: float = 0.25
    realized_threshold: float = 0.3
    realized_bonus: float = 10.0

    def __post_init__(self) -> None:
        integer_fields = {
            "interval_agent_decisions": self.interval_agent_decisions,
            "updates_per_prior_task": self.updates_per_prior_task,
            "batch_sequences": self.batch_sequences,
            "context_steps": self.context_steps,
            "horizon": self.horizon,
        }
        invalid = {
            name: value
            for name, value in integer_fields.items()
            if isinstance(value, bool) or not isinstance(value, int) or value < 1
        }
        if invalid:
            raise ValueError(
                "Dream-rehearsal integer settings must be positive: "
                f"{invalid}"
            )
        if not 0.0 < self.top_fraction <= 1.0:
            raise ValueError("Dream-rehearsal top fraction must lie in (0, 1]")
        if self.realized_threshold < 0.0:
            raise ValueError("Dream-rehearsal realized threshold must be non-negative")
        if self.realized_bonus <= 0.0:
            raise ValueError("Dream-rehearsal realized bonus must be positive")


def crossed_rehearsal_intervals(
    previous_agent_decisions: int,
    current_agent_decisions: int,
    interval_agent_decisions: int,
) -> int:
    """Count newly completed rehearsal intervals without losing remainders."""

    if (
        isinstance(previous_agent_decisions, bool)
        or not isinstance(previous_agent_decisions, int)
        or previous_agent_decisions < 0
    ):
        raise ValueError("Previous agent decisions must be a non-negative integer")
    if (
        isinstance(current_agent_decisions, bool)
        or not isinstance(current_agent_decisions, int)
        or current_agent_decisions < 0
    ):
        raise ValueError("Current agent decisions must be a non-negative integer")
    if current_agent_decisions < previous_agent_decisions:
        raise ValueError("Agent-decision counter must be monotonic")
    if (
        isinstance(interval_agent_decisions, bool)
        or not isinstance(interval_agent_decisions, int)
        or interval_agent_decisions < 1
    ):
        raise ValueError("Rehearsal interval must be a positive integer")
    return (
        current_agent_decisions // interval_agent_decisions
        - previous_agent_decisions // interval_agent_decisions
    )


def rehearsal_update_allocation(
    interval_count: int,
    prior_task_ids: Iterable[int],
    updates_per_prior_task: int,
) -> dict[int, int]:
    """Return the exact, task-balanced actor-only update allocation."""

    if (
        isinstance(interval_count, bool)
        or not isinstance(interval_count, int)
        or interval_count < 0
    ):
        raise ValueError("Rehearsal interval count must be non-negative")
    if (
        isinstance(updates_per_prior_task, bool)
        or not isinstance(updates_per_prior_task, int)
        or updates_per_prior_task < 1
    ):
        raise ValueError("Updates per prior task must be positive")
    task_ids = tuple(prior_task_ids)
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("Prior task IDs must be unique")
    if any(
        isinstance(task_id, bool)
        or not isinstance(task_id, int)
        or task_id < 0
        for task_id in task_ids
    ):
        raise ValueError("Prior task IDs must be non-negative integers")
    updates = interval_count * updates_per_prior_task
    return {task_id: updates for task_id in sorted(task_ids)}


def realized_first_scores(
    rewards: torch.Tensor,
    continues: torch.Tensor,
    bootstrap_values: torch.Tensor,
    *,
    discount: float,
    realized_threshold: float = 0.3,
    realized_bonus: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score imagined trajectories while suppressing post-terminal promises.

    Args:
        rewards: Predicted rewards in ``[H, N, ...]`` order.
        continues: Predicted continuation probabilities with the same shape.
        bootstrap_values: Value at the post-transition state ``s_H`` with
            shape ``[N, ...]``.

    Returns:
        ``(scores, realized_returns, survival_to_horizon)`` in ``[N, ...]``
        order.  The reach probability at step ``t`` is the product of
        continuations strictly before ``t``.
    """

    if rewards.shape != continues.shape:
        raise ValueError(
            "Dream rewards and continuations must have equal shapes, got "
            f"{rewards.shape} and {continues.shape}"
        )
    if rewards.ndim < 2 or rewards.shape[0] < 1:
        raise ValueError("Dream tensors must have non-empty horizon and batch axes")
    if bootstrap_values.shape != rewards.shape[1:]:
        raise ValueError(
            "Bootstrap values must match the non-horizon dream shape, got "
            f"{bootstrap_values.shape} and {rewards.shape[1:]}"
        )
    if not 0.0 < discount <= 1.0:
        raise ValueError("Dream-rehearsal discount must lie in (0, 1]")
    if realized_threshold < 0.0 or realized_bonus <= 0.0:
        raise ValueError("Dream-rehearsal grading thresholds are invalid")
    if not rewards.is_floating_point() or not continues.is_floating_point():
        raise TypeError("Dream rewards and continuations must be floating point")
    if not bootstrap_values.is_floating_point():
        raise TypeError("Dream bootstrap values must be floating point")
    _require_tensor_condition(
        ((continues >= 0) & (continues <= 1)).all(),
        "Dream continuation probabilities must lie in [0, 1]",
    )

    rewards = rewards.float()
    continues = continues.float()
    bootstrap_values = bootstrap_values.float()
    survival = torch.cumprod(continues, dim=0)
    reach = torch.cat((torch.ones_like(continues[:1]), survival[:-1]), dim=0)
    powers = torch.arange(
        rewards.shape[0], device=rewards.device, dtype=rewards.dtype
    )
    discounts = discount**powers
    discounts = discounts.reshape(
        (rewards.shape[0],) + (1,) * (rewards.ndim - 1)
    )
    realized = (discounts * reach * rewards).sum(dim=0)
    survived_horizon = survival[-1]
    score = (
        (realized > realized_threshold).to(realized.dtype) * realized_bonus
        + realized
        + discount**rewards.shape[0]
        * survived_horizon
        * bootstrap_values
    )
    return score, realized, survived_horizon


def top_fraction_indices(scores: torch.Tensor, fraction: float) -> torch.Tensor:
    """Select the paper-compatible floor of the best trajectory scores."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("Selection fraction must lie in (0, 1]")
    if scores.ndim == 2 and scores.shape[-1] == 1:
        scores = scores.squeeze(-1)
    if scores.ndim != 1 or scores.numel() < 1:
        raise ValueError("Trajectory scores must be a non-empty one-dimensional tensor")
    keep = max(1, int(fraction * scores.shape[0]))
    return torch.topk(scores, keep, sorted=False).indices


def selected_behavior_cloning_loss(
    actor_log_probs: torch.Tensor,
    sampled_actions: torch.Tensor,
    selected_indices: torch.Tensor,
) -> torch.Tensor:
    """Clone sampled dream actions for selected trajectories only.

    Tensors use ``[H, N, A]`` order and actions are one-hot.  Actor logits and
    world-model tensors are intentionally not accepted here: this boundary
    keeps the continual-learning loss independent from a particular Dreamer
    implementation.
    """

    if actor_log_probs.shape != sampled_actions.shape:
        raise ValueError("Actor log probabilities and dream actions must align")
    if actor_log_probs.ndim != 3 or actor_log_probs.shape[-1] < 2:
        raise ValueError("Behavior-cloning tensors must use [H, N, A] order")
    if selected_indices.ndim != 1 or selected_indices.numel() < 1:
        raise ValueError("Selected dream indices must be a non-empty vector")
    if selected_indices.dtype != torch.long:
        raise TypeError("Selected dream indices must use torch.long")
    _require_tensor_condition(
        (
            (selected_indices >= 0)
            & (selected_indices < actor_log_probs.shape[1])
        ).all(),
        "Selected dream index is outside the trajectory batch",
    )
    chosen_logs = actor_log_probs.index_select(1, selected_indices).float()
    chosen_actions = sampled_actions.index_select(1, selected_indices).float()
    return -(chosen_logs * chosen_actions).sum(dim=-1).mean()
