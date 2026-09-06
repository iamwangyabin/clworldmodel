"""Episode-preserving history with a single optional uniform-reservoir cap.

Retention units are non-overlapping blocks of K environment transitions, not
overlapping sampled minibatches. All fields live in one fixed mmap per field;
phase libraries contain views only. Evicted views cannot retain old pixels.
Uncapped consecutive blocks reconstruct the original episode exactly, without
inserting artificial RSSM resets at block boundaries.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Any, Callable

import numpy as np


@dataclass(eq=False)
class Fragment:
    slot: int
    block: int
    offset: int
    first_transition: int
    count: int = 0


@dataclass(eq=False)
class Episode:
    key: str
    phase: int
    online: bool
    transitions: int = 0
    fragments: list[Fragment] = field(default_factory=list)
    published_keys: list[str] = field(default_factory=list)


class EpisodeSeries:
    """Read-only array-like view accepted by the unchanged NM512 sampler.

    It owns only fragment indices, never observation arrays. Slices materialize
    just the requested minibatch context. The leading context row of each
    subsequent contiguous fragment is omitted to recover the source episode.
    """

    def __init__(self, history: EpisodeHistory, name: str, fragments: list[Fragment]):
        self.history, self.name, self.fragments = history, name, tuple(fragments)

    def __len__(self) -> int:
        return 1 + sum(f.count for f in self.fragments)

    def __getitem__(self, index):
        if isinstance(index, int):
            pos = index if index >= 0 else len(self) + index
            if pos < 0 or pos >= len(self):
                raise IndexError(index)
            return self[pos:pos + 1][0]
        if not isinstance(index, slice):
            raise TypeError("Episode series supports integer or slice indexing")
        start, stop, stride = index.indices(len(self))
        if stride != 1:
            raise ValueError("Replay sequence slices must be contiguous")
        parts, position = [], 0
        storage = self.history.arrays[self.name]
        for i, fragment in enumerate(self.fragments):
            if self.history.slot_blocks[fragment.slot] != fragment.block:
                raise RuntimeError("Attempted to read an evicted replay fragment")
            count = fragment.count + (i == 0)
            lo, hi = max(start - position, 0), min(stop - position, count)
            if hi > lo:
                offset = fragment.offset + (i != 0)
                parts.append(storage[fragment.slot, offset + lo:offset + hi])
            position += count
        if not parts:
            return np.empty((0, *storage.shape[2:]), dtype=storage.dtype)
        # Always a copy: minibatch lifetime never leases writable replay slots.
        return np.concatenate(parts, axis=0)


class EpisodeHistory:
    """Global Algorithm-R reservoir, optionally uncapped, over started blocks.

    A slot is selected when a block's first transition arrives. If rejected,
    none of that block's transitions enters history. If accepted, it fills
    incrementally. This keeps retained transitions <= capacity even during
    collection. At a block-aligned endpoint every retained block is complete
    and each observed block has inclusion probability slots / blocks_seen.

    A fragment prefix stores only its starting observation (zero action/reward),
    not another transition. Max two stored observation rows per real transition
    covers even one-step episodes. The current reset row is transient collector
    context, contains zero transitions, and is reported separately.
    """

    def __init__(self, directory: Path, *, total_decisions: int,
                 capacity_transitions: int | None, block_length: int = 512,
                 rng: random.Random, dataset_factory: Callable[[OrderedDict], Any]):
        for name, value in (("total_decisions", total_decisions), ("block_length", block_length)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if capacity_transitions is not None and (
            type(capacity_transitions) is not int or capacity_transitions < block_length
            or capacity_transitions % block_length
        ):
            raise ValueError("Capacity must be a positive integral number of transition blocks")
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        self.total_decisions, self.capacity = total_decisions, capacity_transitions
        self.block_length, self.rng = block_length, rng
        self.slots = ((total_decisions + block_length - 1) // block_length
                      if capacity_transitions is None else capacity_transitions // block_length)
        self.arrays: dict[str, np.memmap] = {}
        self.slot_blocks = [-1] * self.slots
        self.slot_rows = [0] * self.slots
        self.slot_fragments: list[list[tuple[Episode, Fragment]]] = [[] for _ in range(self.slots)]
        self.episodes: OrderedDict = OrderedDict()
        self.phase_episodes: dict[int, OrderedDict] = {}
        self.rehearsal_datasets: dict[int, Any] = {}
        self._dataset_factory = dataset_factory
        self._catalog: dict[str, Episode] = {}
        self._current: Episode | None = None
        self._prefix: dict[str, np.ndarray] | None = None
        self._zero_episode_key: str | None = None
        self._active_slot: int | None = None
        self._active_fragment: Fragment | None = None
        self.transitions_seen = self.blocks_seen = self.rejected_blocks = self.evicted_blocks = 0

    @staticmethod
    def context_row(row: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        result = {k: np.array(v, copy=True) for k, v in row.items()}
        for name in ("reward", "action", "logprob", "is_terminal", "is_last"):
            result[name] = np.zeros_like(result[name])
        result["is_first"] = np.ones_like(result["is_first"])
        result["discount"] = np.ones_like(result["discount"])
        return result

    def _validate_row(self, row) -> None:
        required = {"image", "is_first", "is_last", "is_terminal", "reward", "discount", "action", "logprob"}
        if set(row) != required:
            raise ValueError(f"Reference replay row fields differ: {set(row) ^ required}")
        if np.asarray(row["image"]).dtype != np.uint8:
            raise ValueError("Replay image dtype must remain uint8")
        if not self.arrays:
            for name, value in row.items():
                value = np.asarray(value)
                self.arrays[name] = np.memmap(
                    self.directory / f"{name}.mmap", mode="w+", dtype=value.dtype,
                    shape=(self.slots, 2 * self.block_length, *value.shape),
                )
        for name, value in row.items():
            value, target = np.asarray(value), self.arrays[name]
            if value.shape != target.shape[2:] or value.dtype != target.dtype:
                raise ValueError(f"Replay row shape/dtype changed for {name}")

    def begin_episode(self, key: str, row: dict[str, np.ndarray], *, phase: int, online: bool) -> None:
        self._validate_row(row)
        if key in self._catalog or key in self.episodes:
            raise ValueError("Episode identities must be unique")
        if self._zero_episode_key is not None:
            self.episodes.pop(self._zero_episode_key, None)
        self._current = Episode(key, phase, online)
        self._active_fragment = None
        self._prefix = self.context_row(row)
        # Same one-row episode visible to the source sampler before collection;
        # it is ineligible (length < 2), exactly like tools.add_to_cache(reset).
        self.episodes[key] = {k: v[None].copy() for k, v in self._prefix.items()}
        self._zero_episode_key = key

    def _refresh_episode(self, episode: Episode) -> None:
        phase_view = self.phase_episodes.get(episode.phase) if episode.online else None
        for key in episode.published_keys:
            self.episodes.pop(key, None)
            if phase_view is not None:
                phase_view.pop(key, None)
        episode.published_keys.clear()
        fragments = sorted(episode.fragments, key=lambda f: f.first_transition)
        groups: list[list[Fragment]] = []
        for fragment in fragments:
            if groups and groups[-1][-1].first_transition + groups[-1][-1].count == fragment.first_transition:
                groups[-1].append(fragment)
            else:
                groups.append([fragment])
        for group in groups:
            first = group[0].first_transition
            key = episode.key if first == 1 else f"{episode.key}/retained-{first}"
            value = {name: EpisodeSeries(self, name, group) for name in self.arrays}
            self.episodes[key] = value
            episode.published_keys.append(key)
            if phase_view is not None:
                phase_view[key] = value
        if fragments:
            self._catalog[episode.key] = episode
        else:
            self._catalog.pop(episode.key, None)

    def _start_block(self) -> None:
        self.blocks_seen += 1
        if self.capacity is None or self.blocks_seen <= self.slots:
            slot = self.blocks_seen - 1
        else:
            slot = self.rng.randrange(self.blocks_seen)
            if slot >= self.slots:
                self._active_slot, self._active_fragment = None, None
                self.rejected_blocks += 1
                return
        if slot >= self.slots:
            raise RuntimeError("Declared full-history interaction budget exceeded")
        affected = set()
        if self.slot_blocks[slot] >= 0:
            self.evicted_blocks += 1
        for episode, fragment in self.slot_fragments[slot]:
            episode.fragments.remove(fragment)
            affected.add(episode)
        self.slot_fragments[slot].clear()
        self.slot_rows[slot], self.slot_blocks[slot] = 0, self.blocks_seen - 1
        for episode in affected:
            self._refresh_episode(episode)
        self._active_slot, self._active_fragment = slot, None

    def _write_row(self, slot: int, row) -> None:
        offset = self.slot_rows[slot]
        if offset >= 2 * self.block_length:
            raise RuntimeError("Replay reset/context row bound exceeded")
        for name, value in row.items():
            self.arrays[name][slot, offset] = value
        self.slot_rows[slot] += 1

    def add_transition(self, key: str, row: dict[str, np.ndarray]) -> None:
        self._validate_row(row)
        episode = self._current
        if episode is None or key != episode.key:
            raise ValueError("Transition does not belong to the current episode")
        if self.transitions_seen >= self.total_decisions:
            raise RuntimeError("Declared interaction budget exceeded; refusing extra replay data")
        if self.transitions_seen % self.block_length == 0:
            self._start_block()
        if self._zero_episode_key is not None:
            self.episodes.pop(self._zero_episode_key, None)
            self._zero_episode_key = None
        slot = self._active_slot
        if slot is not None:
            if self._active_fragment is None:
                fragment = Fragment(slot, self.blocks_seen - 1, self.slot_rows[slot], episode.transitions + 1)
                self._write_row(slot, self._prefix)
                self._active_fragment = fragment
                self.slot_fragments[slot].append((episode, fragment))
                episode.fragments.append(fragment)
            self._write_row(slot, row)
            self._active_fragment.count += 1
            self._refresh_episode(episode)
        episode.transitions += 1
        self.transitions_seen += 1
        self._prefix = self.context_row(row)

    def ordinary_dataset(self):
        return self._dataset_factory(self.episodes)

    def finish_phase(self, phase_id: int) -> None:
        if phase_id != len(self.phase_episodes):
            raise ValueError("Phases must finish exactly once in order")
        own = [e for e in self._catalog.values() if e.phase == phase_id and e.online]
        if len(own) < 2:
            raise RuntimeError(f"Phase {phase_id} needs at least two retained online episodes")
        view = OrderedDict((key, self.episodes[key]) for e in own for key in e.published_keys)
        self.phase_episodes[phase_id] = view
        self.rehearsal_datasets[phase_id] = self._dataset_factory(view)

    def require_rehearsal_phase(self, phase_id: int) -> None:
        if not self.phase_episodes.get(phase_id):
            raise RuntimeError(f"No retained data for prior phase {phase_id}; refusing hidden backup or skipped rehearsal")

    def flush(self) -> None:
        for array in self.arrays.values():
            array.flush()

    def accounting(self, *, include_index: bool = False) -> dict:
        retained = sum(f.count for entries in self.slot_fragments for _, f in entries)
        rows = sum(self.slot_rows)
        per_row_bytes = sum(a.dtype.itemsize * int(np.prod(a.shape[2:])) for a in self.arrays.values())
        stats = [Path(a.filename).stat() for a in self.arrays.values()]
        result = {
            "collected_transitions": self.transitions_seen,
            "collected_transitions_retained": retained,
            "transition_capacity": self.capacity,
            "retention_block_transitions": self.block_length,
            "blocks_seen": self.blocks_seen,
            "retained_blocks": sum(b >= 0 for b in self.slot_blocks),
            "rejected_blocks": self.rejected_blocks, "evicted_blocks": self.evicted_blocks,
            "stored_rows_including_context": rows,
            "context_rows_not_additional_transitions": rows - retained,
            "transient_reset_rows": int(self._zero_episode_key is not None),
            "collector_context_tensor_bytes": sum(v.nbytes for v in (self._prefix or {}).values()),
            "reset_episode_tensor_bytes": (
                sum(v.nbytes for v in self.episodes[self._zero_episode_key].values())
                if self._zero_episode_key is not None else 0
            ),
            "logical_used_tensor_bytes": rows * per_row_bytes,
            "logical_allocated_tensor_bytes": sum(a.nbytes for a in self.arrays.values()),
            "filesystem_allocated_bytes": sum(s.st_blocks * 512 for s in stats),
            "episodes_or_retained_segments": len(self.episodes),
            "phase_library_segments": {str(k): len(v) for k, v in self.phase_episodes.items()},
            "phase_libraries_copy_observations": False,
            "unbounded_training_npz_backup": False,
            "ordinary_training_sampling": "shared_retained_history",
            "python_and_filesystem_metadata_overhead_included": False,
            "upstream_minibatch_and_optimizer_workspace_included": False,
        }
        if include_index:
            result["fields"] = {name: {"shape": list(a.shape), "dtype": str(a.dtype),
                                       "file": Path(a.filename).name} for name, a in self.arrays.items()}
            result["retained_index"] = [
                {"episode": e.key, "phase": e.phase, "online": e.online,
                 "slot": f.slot, "block": f.block, "offset": f.offset,
                 "first_transition": f.first_transition, "count": f.count}
                for entries in self.slot_fragments for e, f in entries
            ]
        return result
