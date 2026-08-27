"""Task routing and fixed-budget update allocation for MoE-ARROW."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Generic, Optional, TypeVar

import numpy as np
import torch


T = TypeVar("T")


def allocate_task_updates(
    total_updates: int,
    *,
    current_task_id: int,
    available_task_ids: Iterable[int],
    current_task_fraction: float,
) -> dict[int, int]:
    """Split a fixed update budget between the current task and rehearsal tasks."""
    if total_updates < 0:
        raise ValueError("total_updates must be non-negative")
    if not 0 < current_task_fraction <= 1:
        raise ValueError("current_task_fraction must lie in (0, 1]")
    tasks = tuple(sorted(set(int(task_id) for task_id in available_task_ids)))
    if not tasks:
        raise ValueError("At least one available task is required")
    if current_task_id not in tasks:
        raise ValueError(f"Current task {current_task_id} is not available for replay")
    if len(tasks) == 1:
        return {current_task_id: total_updates}

    current_updates = int(total_updates * current_task_fraction + 0.5)
    current_updates = min(total_updates, max(0, current_updates))
    old_tasks = tuple(task_id for task_id in tasks if task_id != current_task_id)
    old_total = total_updates - current_updates
    quotient, remainder = divmod(old_total, len(old_tasks))
    allocation = {
        task_id: quotient + int(index < remainder)
        for index, task_id in enumerate(old_tasks)
    }
    allocation[current_task_id] = current_updates
    return dict(sorted(allocation.items()))


def shuffled_task_schedule(
    allocation: Mapping[int, int], rng: np.random.Generator
) -> tuple[int, ...]:
    """Expand an allocation into a shuffled, reproducible update schedule."""
    if any(task_id < 0 for task_id in allocation):
        raise ValueError("task ids must be non-negative")
    if any(updates < 0 for updates in allocation.values()):
        raise ValueError("task update counts must be non-negative")
    if any(allocation.values()):
        schedule = np.concatenate(
            [
                np.full(updates, task_id, dtype=np.int64)
                for task_id, updates in sorted(allocation.items())
                if updates
            ]
        )
    else:
        schedule = np.empty(0, dtype=np.int64)
    rng.shuffle(schedule)
    return tuple(int(task_id) for task_id in schedule)


@dataclass
class ActorCriticBank(Generic[T]):
    """Own independent actor-critic/optimizer bundles keyed by task identity."""

    artifact_kind: str = "moe_arrow_actor_critic_bank_inference_state"
    _entries: dict[int, T] = field(default_factory=dict)

    def __contains__(self, task_id: int) -> bool:
        return task_id in self._entries

    def task_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._entries))

    def get(self, task_id: int) -> T:
        try:
            return self._entries[task_id]
        except KeyError as exc:
            raise KeyError(
                f"Actor-critic task {task_id} has not been initialized"
            ) from exc

    def get_optional(self, task_id: int) -> Optional[T]:
        return self._entries.get(task_id)

    def ensure(
        self,
        task_id: int,
        factory: Callable[[int], T],
        *,
        warm_start_from: Optional[int] = None,
    ) -> T:
        """Create one task entry, optionally cloning only network/EMA state."""
        if task_id < 0:
            raise ValueError("task_id must be non-negative")
        existing = self._entries.get(task_id)
        if existing is not None:
            return existing

        entry = factory(task_id)
        if warm_start_from is not None:
            source = self.get(warm_start_from)
            target_ac = getattr(entry, "ac")
            source_ac = getattr(source, "ac")
            target_ac.load_state_dict(source_ac.state_dict())
            source_slow = getattr(source, "slow_critic", None)
            target_slow = getattr(entry, "slow_critic", None)
            if (source_slow is None) != (target_slow is None):
                raise ValueError("Warm-start actor-critics disagree on slow-critic state")
            if source_slow is not None:
                target_slow.load_state_dict(source_slow.state_dict())
            for name in ("return_scale_ema", "return_mean_ema"):
                value = getattr(source, name, None)
                copied = value.detach().clone() if value is not None else None
                setattr(entry, name, copied)
        self._entries[task_id] = entry
        return entry

    def activate(self, task_id: int) -> None:
        """Make one actor-critic plastic while preserving frozen old policies."""
        self.get(task_id)
        for entry_task_id, entry in self._entries.items():
            is_active = entry_task_id == task_id
            getattr(entry, "ac").requires_grad_(is_active)
            slow_critic = getattr(entry, "slow_critic", None)
            if slow_critic is not None:
                slow_critic.requires_grad_(is_active)

    def inference_state_dict(self) -> dict[str, object]:
        """Return CPU actor states without pretending optimizer/replay are resumable."""
        tasks: dict[str, dict[str, torch.Tensor]] = {}
        for task_id, entry in sorted(self._entries.items()):
            actor_critic = getattr(entry, "ac")
            tasks[str(task_id)] = {
                name: value.detach().cpu()
                for name, value in actor_critic.state_dict().items()
            }
        return {
            "schema_version": 1,
            "artifact_kind": self.artifact_kind,
            "resumable": False,
            "tasks": tasks,
        }
