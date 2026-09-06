"""Explicit, pilot-only Continual-Dreamer recipe on a pinned DreamerV3 core."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ContinualDreamerConfig:
    """Only these fields are user configurable; the protocol name is immutable.

    Native V3 architecture/optimizer defaults are resolved from its pinned YAML
    and saved alongside this recipe, not accepted as unchecked user overrides.
    """

    protocol: str = "CD-DV3-P2E-EpisodeSlots-v1"
    evidence_level: str = "pilot"
    method: Literal["arrow50", "rs"] = "arrow50"
    seed: int = 0
    task_count: int = 1
    decisions_per_task: int = 750_000
    prefill_decisions: int = 10_000
    initial_updates: int = 2
    train_every_decisions: int = 10
    batch_size: int = 16
    batch_length: int = 50
    episode_limit: int = 100
    replay_episode_slots: int = 20_000
    eval_every_decisions: int = 10_000
    eval_decisions_per_task: int = 1_000
    log_every_decisions: int = 1_000
    disagreement_models: int = 10
    disagreement_layers: int = 4
    disagreement_units: int = 400
    disagreement_lr: float = 3e-4
    disagreement_eps: float = 1e-5
    disagreement_clip: float = 100.0
    disagreement_weight_decay: float = 1e-6
    intrinsic_scale: float = 0.9
    extrinsic_scale: float = 0.9
    reward_norm_eps: float = 1e-8
    discount: float = 0.99
    actor_entropy: float = 3e-3
    device: str = "cuda:0"
    cpu_threads: int = 4

    def __post_init__(self) -> None:
        defaults = type(self).__dataclass_fields__
        flexible = {"method", "seed", "task_count", "decisions_per_task", "device", "cpu_threads"}
        for name, definition in defaults.items():
            value = getattr(self, name)
            if type(value) is not type(definition.default):
                raise ValueError(f"{name} must be {type(definition.default).__name__}")
            if name not in flexible and value != definition.default:
                raise ValueError(f"{name} is frozen by {self.protocol}; use a new protocol for changes")
        if self.method not in {"arrow50", "rs"}:
            raise ValueError("method must be arrow50 or rs")
        if not 0 <= self.seed < 2**32 - 100_000:
            raise ValueError("seed must be in [0, 2**32 - 100000)")
        if self.task_count not in {1, 3}:
            raise ValueError("task_count must be 1 (acquisition pilot) or 3 (continual pilot)")
        if self.decisions_per_task <= self.prefill_decisions or self.decisions_per_task % self.eval_every_decisions:
            raise ValueError("decisions_per_task must exceed prefill and be a multiple of eval_every_decisions")
        if self.device not in {"cpu", "cuda:0"} or self.cpu_threads < 1:
            raise ValueError("device must be cpu/cuda:0 and cpu_threads must be positive")

    @classmethod
    def from_json(cls, path: Path, **overrides) -> "ContinualDreamerConfig":
        values = json.loads(path.read_text())
        if not isinstance(values, dict):
            raise ValueError("Protocol JSON must be an object")
        unknown = set(values) - cls.__dataclass_fields__.keys()
        if unknown:
            raise ValueError(f"Unknown protocol keys: {sorted(unknown)}")
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def total_decisions(self) -> int:
        return self.decisions_per_task * self.task_count

    @property
    def expected_updates(self) -> int:
        # Source Every() fires on the first post-prefill action, then every 10.
        return self.initial_updates + (self.total_decisions - self.prefill_decisions + self.train_every_decisions - 1) // self.train_every_decisions


@dataclass
class Counters:
    environment_decisions: int = 0
    raw_environment_frames: int = 0
    collected_transitions: int = 0
    reset_observations: int = 0
    world_model_updates: int = 0
    task_actor_updates: int = 0
    task_critic_updates: int = 0
    exploration_actor_updates: int = 0
    exploration_critic_updates: int = 0
    ensemble_updates: int = 0
    evaluation_decisions: int = 0


def should_update(decisions: int, config: ContinualDreamerConfig) -> bool:
    return decisions > config.prefill_decisions and (decisions - config.prefill_decisions - 1) % config.train_every_decisions == 0
