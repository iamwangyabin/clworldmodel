"""CPU episode replay: source window sampling, explicitly named slot capacity.

NOT a literal port of Continual-Dreamer's variable-transition-cap RS code.
Each slot holds one complete, unpadded eligible episode. LTDM is Algorithm R
over episodes. A sampled minibatch comes entirely from one selected subbuffer.
No environment names, task labels, reward transformation, or torch dependency.
"""

from __future__ import annotations

import sys
from typing import Literal

import numpy as np

Episode = dict[str, np.ndarray]
FIELDS = {"image", "action", "reward", "is_first", "is_last", "is_terminal"}


class EpisodeSlotsReplay:
    def __init__(
        self, capacity: int, *, method: Literal["arrow50", "rs"],
        min_decisions: int, max_decisions: int, sequence_length: int, seed: int,
    ) -> None:
        if capacity < 1 or (method == "arrow50" and capacity % 2):
            raise ValueError("capacity must be positive and even for ARROW-50")
        if method not in {"arrow50", "rs"}:
            raise ValueError("method must be arrow50 or rs")
        if not 1 <= sequence_length <= min_decisions <= max_decisions:
            raise ValueError("Require 1 <= sequence_length <= min_decisions <= max_decisions")
        self.capacity, self.method = capacity, method
        self.min_decisions, self.max_decisions = min_decisions, max_decisions
        self.sequence_length = sequence_length
        seeds = np.random.SeedSequence(seed).spawn(3)
        self.retention_rng, self.selection_rng, self.sampling_rng = [np.random.default_rng(s) for s in seeds]
        self.fifo: list[Episode] = []
        self.reservoir: list[Episode] = []
        self.fifo_cursor = self.eligible_seen = self.rejected_short = 0
        self.fifo_selections = self.reservoir_selections = 0
        self._schema: dict | None = None

    def add(self, episode: Episode) -> bool:
        if set(episode) != FIELDS or any(not isinstance(x, np.ndarray) for x in episode.values()):
            raise ValueError(f"Episode must contain exactly numpy arrays {sorted(FIELDS)}")
        rows = len(episode["reward"])
        if any(len(x) != rows for x in episode.values()) or not 2 <= rows <= self.max_decisions + 1:
            raise ValueError("Episode arrays must share a valid complete-episode length")
        if episode["image"].dtype != np.uint8 or episode["image"].ndim != 4:
            raise ValueError("Replay images must be uint8 [time, height, width, channels]")
        if episode["action"].dtype != np.float32 or episode["action"].ndim != 2 or episode["reward"].shape != (rows,) or episode["reward"].dtype != np.float32:
            raise ValueError("Actions/rewards must be float32 [time, actions]/[time]")
        for key in ("is_first", "is_last", "is_terminal"):
            if episode[key].dtype != np.bool_ or episode[key].shape != (rows,):
                raise ValueError(f"{key} must be bool [time]")
        if not episode["is_first"][0] or episode["is_first"][1:].any() or not episode["is_last"][-1] or episode["is_last"][:-1].any():
            raise ValueError("Replay accepts complete episodes with exactly one reset and one last row")
        if (episode["is_terminal"] & ~episode["is_last"]).any():
            raise ValueError("A terminal row must also be the episode's last row")
        action = episode["action"]
        if action[0].any() or episode["reward"][0] != 0 or not np.all((action[1:] == 0) | (action[1:] == 1)) or not np.all(action[1:].sum(-1) == 1) or not np.isfinite(episode["reward"]).all():
            raise ValueError("Require zero reset action/reward and finite rewards with one-hot incoming actions")
        schema = {k: (v.shape[1:], v.dtype.str) for k, v in episode.items()}
        if self._schema is not None and schema != self._schema:
            raise ValueError("Replay episode shape/dtype schema changed")
        self._schema = schema
        if rows - 1 < self.min_decisions:
            self.rejected_short += 1
            return False
        self.eligible_seen += 1
        size = self.capacity // 2 if self.method == "arrow50" else self.capacity
        # Separate physical copies make ARROW's duplicate occupancy explicit.
        if self.method == "arrow50":
            entry = {k: v.copy() for k, v in episode.items()}
            if len(self.fifo) < size:
                self.fifo.append(entry)
            else:
                self.fifo[self.fifo_cursor] = entry
            self.fifo_cursor = (self.fifo_cursor + 1) % size
        if len(self.reservoir) < size:
            self.reservoir.append({k: v.copy() for k, v in episode.items()})
        else:
            index = int(self.retention_rng.integers(self.eligible_seen))
            if index < size:
                self.reservoir[index] = {k: v.copy() for k, v in episode.items()}
        return True

    def sample(self, batch_size: int) -> Episode:
        if batch_size < 1 or not self.reservoir:
            raise ValueError("Sampling needs a positive batch size and at least one eligible episode")
        fifo = self.method == "arrow50" and self.selection_rng.random() < 0.5
        buffer = self.fifo if fifo else self.reservoir
        self.fifo_selections += int(fifo)
        self.reservoir_selections += int(not fifo)
        samples = []
        for _ in range(batch_size):
            episode = buffer[int(self.sampling_rng.integers(len(buffer)))]
            last_start = len(episode["reward"]) - self.sequence_length
            # Released source: prioritize_ends adds minlen to the draw range,
            # then clamps to the final valid start (it is NOT uniform starts).
            start = min(int(self.sampling_rng.integers(last_start + 1 + self.min_decisions)), last_start)
            samples.append({k: v[start:start + self.sequence_length] for k, v in episode.items()})
        result = {k: np.stack([sample[k] for sample in samples]) for k in FIELDS}
        result["is_first"][:] = False
        result["is_first"][:, 0] = True
        return result

    def accounting(self) -> dict:
        buffers = (self.fifo, self.reservoir)
        episodes = [ep for buffer in buffers for ep in buffer]
        payload = sum(array.nbytes for ep in episodes for array in ep.values())
        # Array/object/list overhead is CPython-specific, not GPU allocator or
        # process RSS. Shared key strings and RNG objects are counted once.
        overhead = sum(sys.getsizeof(buffer) for buffer in buffers)
        overhead += sum(sys.getsizeof(ep) + sum(sys.getsizeof(a) - a.nbytes for a in ep.values()) for ep in episodes)
        overhead += sum(sys.getsizeof(k) for k in FIELDS)
        return dict(
            storage_device="cpu", image_dtype="uint8", compression="none",
            capacity_episode_slots=self.capacity,
            capacity_action_upper_bound=self.capacity * self.max_decisions,
            fifo_episodes=len(self.fifo), reservoir_episodes=len(self.reservoir),
            retained_action_occupancy=sum(len(ep["reward"]) - 1 for ep in episodes),
            eligible_episodes_seen=self.eligible_seen, rejected_short_episodes=self.rejected_short,
            payload_bytes=payload, container_and_array_overhead_bytes=overhead,
            accounted_bytes=payload + overhead,
            fifo_minibatches=self.fifo_selections, reservoir_minibatches=self.reservoir_selections,
        )
