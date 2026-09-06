"""Declarative protocol and scheduling for the official-code Atari transfer.

Learning is deliberately NOT implemented here. The integration invokes the
unchanged NM512 Dreamer and Nijjer ``tunnel_update`` at recorded source pins.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, fields
from typing import Any, Callable, ClassVar, Iterator


ATARI_TASKS = (
    "ALE/MsPacman-v5", "ALE/Boxing-v5", "ALE/CrazyClimber-v5",
    "ALE/Frostbite-v5", "ALE/Seaquest-v5", "ALE/Enduro-v5",
)
METHOD = "Dream-Rehearsal-OfficialCode-v1-Atari"
MEMORY_PAIR_METHOD = "Dream-Rehearsal-MemoryPair-v1-Atari"
ARROW_TRANSITION_CAPACITY = 1024 * 512


@dataclass(frozen=True)
class OfficialDreamRehearsalConfig:
    """All project-level, behavior-affecting settings for this named protocol.

    Model/optimizer defaults come from the hash-checked upstream configs and
    substrate SETUP preset, not from the ARROW port. Algorithm overrides are
    rejected: tuning them requires a separately named method.
    """

    _protocol_name: ClassVar[str] = METHOD
    protocol: str = METHOD
    classification: str = "pilot"
    seed: int = 123456789
    tasks: tuple[str, ...] = ATARI_TASKS
    task_total_decisions: int = 90 * 32 * 512
    prefill_decisions: int = 2500
    chunk_decisions: int = 2000
    rehearsal_updates: int = 50
    batch_size: int = 16
    batch_length: int = 64
    imag_horizon: int = 15
    top_fraction: float = 0.25
    realized_threshold: float = 0.3
    realized_bonus: float = 10.0
    train_ratio: int = 512
    pretrain_updates: int = 100
    eval_episodes: int = 10
    final_average_rounds: int = 3
    image_size: int = 64
    action_repeat: int = 4
    noop_max: int = 30
    sticky_probability: float = 0.0
    episode_limit_raw_frames: int = 108000
    cpu_threads: int = 8
    device: str = "cuda:0"

    def __post_init__(self) -> None:
        for name in (
            "seed", "task_total_decisions", "prefill_decisions", "chunk_decisions",
            "rehearsal_updates", "batch_size", "batch_length", "imag_horizon",
            "train_ratio", "pretrain_updates", "eval_episodes", "final_average_rounds",
            "image_size", "action_repeat", "noop_max", "episode_limit_raw_frames",
            "cpu_threads",
        ):
            value = getattr(self, name)
            minimum = 0 if name in {"seed", "noop_max"} else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be a positive integer (seed/noops allow zero)")
        if self.protocol != self._protocol_name:
            raise ValueError(f"This runner only implements {self._protocol_name}")
        if self.seed >= 2**32:
            raise ValueError("seed must be smaller than 2**32 for NumPy and PYTHONHASHSEED")
        for name in ("top_fraction", "realized_threshold", "realized_bonus", "sticky_probability"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric, not a boolean or string")
        if self.classification not in {"pilot", "smoke"}:
            raise ValueError("Only pilot or smoke are supported; no reproduction claim yet")
        expected_tasks = ATARI_TASKS[:2] if self.classification == "smoke" else ATARI_TASKS
        if self.tasks != expected_tasks:
            raise ValueError("Task order is frozen for this named Atari protocol")
        fixed = {
            "chunk_decisions": 2000, "rehearsal_updates": 50,
            "batch_size": 16, "batch_length": 64, "imag_horizon": 15,
            "top_fraction": 0.25, "realized_threshold": 0.3, "realized_bonus": 10.0,
            "train_ratio": 512, "pretrain_updates": 100,
            "prefill_decisions": 2500, "final_average_rounds": 3,
            "image_size": 64, "action_repeat": 4, "noop_max": 30,
            "sticky_probability": 0.0,
            "eval_episodes": 2 if self.classification == "smoke" else 10,
            "task_total_decisions": 5000 if self.classification == "smoke" else 1474560,
            "episode_limit_raw_frames": 800 if self.classification == "smoke" else 108000,
        }
        different = {k: getattr(self, k) for k, v in fixed.items() if getattr(self, k) != v}
        if different:
            raise ValueError(f"Changing the official-code preset requires a new protocol: {different}")
        if self.device != "cpu" and self.device != "cuda:0":
            raise ValueError("device must be cpu or cuda:0 (one visible accelerator)")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OfficialDreamRehearsalConfig":
        if not isinstance(data, dict):
            raise ValueError("Config must be a JSON object")
        unknown = sorted(set(data) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"Unknown Dream Rehearsal config keys: {unknown}")
        values = dict(data)
        if "tasks" in values:
            if not isinstance(values["tasks"], (list, tuple)):
                raise ValueError("tasks must be a list of task IDs")
            values["tasks"] = tuple(values["tasks"])
        return cls(**values)

    @classmethod
    def smoke(cls, **overrides: Any) -> "OfficialDreamRehearsalConfig":
        return cls(classification="smoke", tasks=ATARI_TASKS[:2],
                   task_total_decisions=5000, episode_limit_raw_frames=800,
                   eval_episodes=2, **overrides)

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["tasks"] = list(self.tasks)
        return result

    def projected_budgets(self) -> dict[str, Any]:
        phases = len(self.tasks)
        online = self.task_total_decisions - self.prefill_decisions
        chunks = online // self.chunk_decisions
        rehearsal_by_phase = [j * chunks * self.rehearsal_updates for j in range(phases)]
        # NM512 Once(pretrain=100), then Every(16*64/512=2) at actor step 1,3,...
        base_updates = self.pretrain_updates + (phases * online) // 2
        return {
            "agent_decisions": phases * self.task_total_decisions,
            "prefill_agent_decisions": phases * self.prefill_decisions,
            "online_agent_decisions": phases * online,
            "nominal_raw_frames_excluding_reset_noops": (
                phases * self.task_total_decisions * self.action_repeat
            ),
            "world_model_updates": base_updates,
            "base_actor_updates": base_updates,
            "base_critic_updates": base_updates,
            "rehearsal_updates": sum(rehearsal_by_phase),
            "rehearsal_updates_by_phase": rehearsal_by_phase,
            "total_actor_updates": base_updates + sum(rehearsal_by_phase),
            "starts_per_rehearsal_update": self.batch_size * self.batch_length,
            "selected_dreams_per_update": int(self.batch_size * self.batch_length * self.top_fraction),
            "full_chunks_per_phase": chunks,
            "tail_decisions_per_phase": online % self.chunk_decisions,
            "evaluation_rounds_per_phase": 1 + len(list(phase_chunks(self))),
            "evaluation_episodes_upper_bound": (
                (1 + len(list(phase_chunks(self)))) * self.eval_episodes * phases * (phases + 1) // 2
            ),
            "evaluation_frames_are_separate": True,
        }


@dataclass(frozen=True)
class MemoryPairConfig(OfficialDreamRehearsalConfig):
    """One algorithm/config; the ONLY arm-specific option is history capacity.

    Both arms use the same episode-preserving block store, original sampler,
    model, optimizer, collection, evaluation and rehearsal schedule. Blocks
    contain 512 actual transitions; reset/context observations are accounted
    separately and are never additional replay transitions.
    """

    _protocol_name: ClassVar[str] = MEMORY_PAIR_METHOD
    protocol: str = MEMORY_PAIR_METHOD
    history_capacity_transitions: int | None = None
    retention_block_transitions: int = 512

    def __post_init__(self) -> None:
        super().__post_init__()
        capacity = self.history_capacity_transitions
        if capacity is not None and (type(capacity) is not int or capacity != ARROW_TRANSITION_CAPACITY):
            raise ValueError("The bounded arm must use ARROW's 524288-transition capacity")
        if type(self.retention_block_transitions) is not int or self.retention_block_transitions != 512:
            raise ValueError("Both arms use identical 512-transition retention blocks")

    @property
    def history_arm(self) -> str:
        return "full" if self.history_capacity_transitions is None else "bounded"


@dataclass(frozen=True)
class TrainingChunk:
    agent_decisions: int
    completed_online_decisions: int
    rehearsal_due: bool


def phase_chunks(config: OfficialDreamRehearsalConfig) -> Iterator[TrainingChunk]:
    """Never overshoot the named task budget; no batched catch-up updates.

    The last <2000-decision fragment is Atari fixed-budget bookkeeping, not an
    extra rehearsal event. The official MiniGrid all-good early stop is NOT
    used: its common return threshold has no meaning across Atari scores.
    """
    total = config.task_total_decisions - config.prefill_decisions
    completed = 0
    while completed < total:
        size = min(config.chunk_decisions, total - completed)
        completed += size
        yield TrainingChunk(size, completed, size == config.chunk_decisions)


def run_phase_chunks(
    config: OfficialDreamRehearsalConfig,
    prior_phases: tuple[int, ...],
    *,
    train: Callable[[int], None],
    rehearse: Callable[[int, int], None],
    evaluate: Callable[[int], None],
) -> None:
    if prior_phases != tuple(range(len(prior_phases))):
        raise ValueError("Prior phases must be the complete ordered history")
    evaluate(0)
    for chunk in phase_chunks(config):
        train(chunk.agent_decisions)
        if chunk.rehearsal_due:
            for phase_id in prior_phases:
                rehearse(phase_id, config.rehearsal_updates)
        evaluate(chunk.completed_online_decisions)


class ReplayLibraries:
    """One growing buffer, with non-copying snapshots for prior-phase dreams.

    ``dataset_factory`` is the pinned ``dreamer.make_dataset``. No task keys are
    injected into observations. Ordinary training ALWAYS samples the same
    full-history dictionary. Per-phase views are ONLY for ``tunnel_update``.
    """

    def __init__(self, dataset_factory: Callable[[OrderedDict], Any]) -> None:
        self.episodes: OrderedDict = OrderedDict()
        self.phase_episodes: dict[int, OrderedDict] = {}
        self.rehearsal_datasets: dict[int, Any] = {}
        self._dataset_factory = dataset_factory

    def ordinary_dataset(self) -> Any:
        return self._dataset_factory(self.episodes)

    def finish_phase(self, phase_id: int, keys_before_online: set[str]) -> None:
        if phase_id != len(self.phase_episodes):
            raise ValueError("Phases must be finalized exactly once in order")
        own = OrderedDict((k, v) for k, v in self.episodes.items() if k not in keys_before_online)
        if len(own) < 2:
            raise RuntimeError(
                f"Phase {phase_id} needs at least two own-phase episodes for reference rehearsal; "
                "refusing to silently skip a prior task"
            )
        self.phase_episodes[phase_id] = own
        self.rehearsal_datasets[phase_id] = self._dataset_factory(own)
