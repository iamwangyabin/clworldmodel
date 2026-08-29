import hashlib
import random
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from sortedcontainers import SortedList

from rssm import ActionT, ContT, ImageT, ResetT
from wm import RewardT


_OBSERVATION_DTYPES = {
    "float32": torch.float32,
    "uint8": torch.uint8,
}


class TaskReplayBatch(Sequence[torch.Tensor]):
    """Five Dreamer tensors plus explicit homogeneous task metadata."""

    def __init__(self, values: tuple[torch.Tensor, ...], task_id: int) -> None:
        if len(values) != 5:
            raise ValueError("A task replay batch requires five model tensors")
        self._values = values
        sequences = values[0].shape[1]
        self.task_ids = torch.full(
            (sequences,),
            task_id,
            dtype=torch.long,
            device=values[0].device,
        )

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter(self._values)

    def __getitem__(self, index):
        return self._values[index]


class Replay:
    def __init__(self) -> None:
        self.n_valid = 0
        self.valid_slots_by_task: dict[int, torch.Tensor] = {}

    def add(
        self,
        acts: ActionT,
        obss: ImageT,
        rews: RewardT,
        conts: ContT,
        resets: ResetT,
        task_id: Optional[int] = None,
    ) -> list[int]:
        raise NotImplementedError

    def minibatch(
        self,
        mb_t: int,
        mb_n: int,
        mb_device: str = "cuda",
        task_id: Optional[int] = None,
    ) -> tuple[ActionT, ImageT, RewardT, ContT, ResetT]:
        return self.minibatch_with_metadata(
            mb_t, mb_n, mb_device, task_id=task_id
        )[:5]

    def minibatch_with_metadata(
        self,
        mb_t: int,
        mb_n: int,
        mb_device: str = "cuda",
        task_id: Optional[int] = None,
    ) -> tuple[ActionT, ImageT, RewardT, ContT, ResetT, np.ndarray, np.ndarray]:
        """Sample a minibatch plus source time and sequence indices.

        The public ``minibatch`` behavior is unchanged. The metadata-only
        variant is used by the R2-Dreamer adapter to read and refresh its
        sidecar posterior-state cache without altering ARROW retention or
        buffer-selection semantics.
        """
        # Data [ T N ... ]
        # Sample minibatches in [ t_size n_size ... ]
        t_size = min(mb_t, self.t)
        t_starts = np.random.randint(0, self.t - t_size + 1, size=mb_n)
        t_stops = t_starts + t_size
        if task_id is None:
            # Preserve upstream sampling and RNG consumption exactly.
            ns = np.random.randint(0, self.n_valid, size=mb_n)
        else:
            eligible = self._eligible_sequence_indices(task_id)
            if eligible.size == 0:
                raise ValueError(f"Replay contains no sequences for task {task_id}")
            ns = np.random.choice(eligible, size=mb_n, replace=True)

        mb_acts = torch.stack(
            [self.acts[t_start:t_stop, it] for t_start, t_stop, it in zip(t_starts, t_stops, ns)],
            dim=1,
        )
        mb_obss = torch.stack(
            [self.obss[t_start:t_stop, it] for t_start, t_stop, it in zip(t_starts, t_stops, ns)],
            dim=1,
        )
        mb_rews = torch.stack(
            [self.rews[t_start:t_stop, it] for t_start, t_stop, it in zip(t_starts, t_stops, ns)],
            dim=1,
        )
        mb_conts = torch.stack(
            [self.conts[t_start:t_stop, it] for t_start, t_stop, it in zip(t_starts, t_stops, ns)],
            dim=1,
        )
        mb_resets = torch.stack(
            [
                self.resets[t_start:t_stop, it]
                for t_start, t_stop, it in zip(t_starts, t_stops, ns)
            ],
            dim=1,
        )
        return (
            mb_acts.to(mb_device),
            self._decode_observations(
                mb_obss.to(mb_device), self.obss.dtype
            ),
            mb_rews.to(mb_device),
            mb_conts.to(mb_device),
            mb_resets.to(mb_device),
            t_starts,
            ns,
        )

    def minibatch_for_task(
        self,
        task_id: int,
        sequence_length: int,
        sequences: int,
        source: Literal["fifo", "ltdm", "mixed"] = "mixed",
        mb_device: str = "cuda",
    ) -> TaskReplayBatch:
        """Sample a task-homogeneous batch without rejection sampling."""

        if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0:
            raise ValueError("task_id must be a non-negative integer")
        if sequence_length < 1 or sequences < 1:
            raise ValueError("Replay sequence length and count must be positive")
        if source not in {"fifo", "ltdm", "mixed"}:
            raise ValueError(f"Unknown replay source: {source!r}")
        if isinstance(self, FifoReplay) and source == "ltdm":
            raise ValueError("A FIFO replay cannot provide an LTDM sample")
        if isinstance(self, LongTermReplay) and source == "fifo":
            raise ValueError("An LTDM replay cannot provide a FIFO sample")
        return TaskReplayBatch(
            self.minibatch(
                sequence_length,
                sequences,
                mb_device,
                task_id=task_id,
            ),
            task_id,
        )

    def _refresh_valid_slots_by_task(self) -> None:
        """Synchronize exact task-to-slot tensors after retention/overwrite."""

        task_ids = getattr(self, "task_ids", None)
        if task_ids is None or self.n_valid == 0:
            self.valid_slots_by_task = {}
            return
        valid = task_ids[: self.n_valid]
        refreshed: dict[int, torch.Tensor] = {}
        for value in valid.unique().tolist():
            task_id = int(value)
            if task_id < 0:
                continue
            refreshed[task_id] = torch.nonzero(
                valid == task_id, as_tuple=False
            ).flatten()
        self.valid_slots_by_task = refreshed

    def _trajectory_state_dict(self) -> dict[str, object]:
        """Serialize valid trajectories and mmap provenance for exact resume."""

        state: dict[str, object] = {
            "replay_type": type(self).__name__,
            "t": self.t,
            "n": self.n,
            "n_valid": self.n_valid,
            "acts": self.acts[:, : self.n_valid].detach().cpu().clone(),
            "rews": self.rews[:, : self.n_valid].detach().cpu().clone(),
            "conts": self.conts[:, : self.n_valid].detach().cpu().clone(),
            "resets": self.resets[:, : self.n_valid].detach().cpu().clone(),
            "task_ids": (
                None
                if self.task_ids is None
                else self.task_ids[: self.n_valid].detach().cpu().clone()
            ),
            "observation_dtype": self.observation_dtype,
        }
        if self.observation_storage_path is None:
            state["observations"] = {
                "kind": "tensor",
                "values": self.obss[:, : self.n_valid].detach().cpu().clone(),
            }
        else:
            state["observations"] = {
                "kind": "mmap",
                "path": str(self.observation_storage_path),
                "shape": list(self.obss.shape),
                "dtype": self.observation_dtype,
                "byte_size": self.observation_storage_path.stat().st_size,
            }
        return state

    def _load_trajectory_state_dict(self, state: dict[str, object]) -> None:
        if state.get("replay_type") != type(self).__name__:
            raise ValueError("Replay checkpoint type does not match target replay")
        if (int(state["t"]), int(state["n"])) != (self.t, self.n):
            raise ValueError("Replay checkpoint capacity does not match target replay")
        n_valid = int(state["n_valid"])
        if not 0 <= n_valid <= self.n:
            raise ValueError("Replay checkpoint n_valid is outside capacity")
        observations = state.get("observations")
        if not isinstance(observations, dict):
            raise ValueError("Replay checkpoint is missing observation state")
        if observations.get("kind") == "mmap":
            from clworldmodel.replay.mapped_tensor import open_file_backed_tensor

            observation_path = Path(str(observations["path"])).expanduser().resolve()
            expected_sha256 = observations.get("sha256")
            if expected_sha256 is not None:
                digest = hashlib.sha256()
                with observation_path.open("rb") as file:
                    for chunk in iter(lambda: file.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != expected_sha256:
                    raise ValueError(
                        f"Mapped replay checksum mismatch: {observation_path}"
                    )
            dtype = _OBSERVATION_DTYPES[str(observations["dtype"])]
            mapped = open_file_backed_tensor(
                observation_path,
                tuple(int(value) for value in observations["shape"]),
                dtype=dtype,
            )
            if tuple(mapped.shape) != tuple(self.obss.shape):
                raise ValueError("Mapped replay shape does not match target replay")
            if mapped.dtype != self.obss.dtype:
                raise ValueError("Mapped replay dtype does not match target replay")
            # Checkpoint assets are immutable provenance.  Copy their contents
            # into the target replay's independently constructed working
            # storage instead of mapping and later mutating the checkpoint.
            self.obss[:, :n_valid].copy_(mapped[:, :n_valid])
        elif observations.get("kind") == "tensor":
            values = observations.get("values")
            if not isinstance(values, torch.Tensor):
                raise ValueError("Replay tensor observations are missing")
            self.obss[:, :n_valid].copy_(values.to(self.obss.device))
        else:
            raise ValueError("Unknown replay observation checkpoint kind")

        for name in ("acts", "rews", "conts", "resets"):
            values = state.get(name)
            target = getattr(self, name)
            if not isinstance(values, torch.Tensor):
                raise ValueError(f"Replay checkpoint is missing {name}")
            target[:, :n_valid].copy_(values.to(target.device))
        task_ids = state.get("task_ids")
        if self.task_ids is None:
            if task_ids is not None:
                raise ValueError("Task-aware replay state cannot load into unlabelled replay")
        else:
            if not isinstance(task_ids, torch.Tensor):
                raise ValueError("Task-aware replay checkpoint is missing task IDs")
            self.task_ids.fill_(-1)
            self.task_ids[:n_valid].copy_(task_ids.to(self.task_ids.device))
        self.n_valid = n_valid
        self._refresh_valid_slots_by_task()

    def _eligible_sequence_indices(self, task_id: int) -> np.ndarray:
        if task_id < 0:
            raise ValueError("task_id must be non-negative")
        if self.task_ids is None:
            raise ValueError("Replay was constructed without task-id storage")
        eligible = self.valid_slots_by_task.get(task_id)
        if eligible is None:
            return np.empty(0, dtype=np.int64)
        return eligible.detach().cpu().numpy()

    def available_task_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.valid_slots_by_task))

    @staticmethod
    def _encode_observations(
        observations: ImageT, storage_dtype: torch.dtype
    ) -> ImageT:
        if storage_dtype == torch.float32:
            return observations
        if storage_dtype != torch.uint8:
            raise TypeError(f"Unsupported observation storage dtype: {storage_dtype}")
        if observations.dtype == torch.uint8:
            return observations
        if not observations.is_floating_point():
            raise TypeError(
                "uint8 replay observations must be uint8 pixels or floating-point "
                "values in [0, 1]"
            )
        minimum, maximum = torch.aminmax(observations)
        if not (
            bool(torch.isfinite(minimum)) and bool(torch.isfinite(maximum))
        ):
            raise ValueError("Replay observations must contain only finite values")
        if minimum.item() < 0.0 or maximum.item() > 1.0:
            raise ValueError(
                "Floating-point observations for uint8 replay must lie in [0, 1]"
            )
        return observations.mul(255).round_().to(torch.uint8)

    @staticmethod
    def _decode_observations(
        observations: ImageT, storage_dtype: torch.dtype
    ) -> ImageT:
        if storage_dtype == torch.float32:
            return observations
        if storage_dtype != torch.uint8:
            raise TypeError(f"Unsupported observation storage dtype: {storage_dtype}")
        return observations.float().div_(255)


def _validated_task_id(
    task_ids: Optional[torch.Tensor], task_id: Optional[int]
) -> Optional[int]:
    if task_ids is None:
        if task_id is not None:
            raise ValueError("Replay was constructed without task-id storage")
        return None
    if task_id is None:
        raise ValueError("Task-aware replay requires task_id on every add")
    if isinstance(task_id, bool) or not isinstance(task_id, int):
        raise TypeError("task_id must be an integer")
    if task_id < 0:
        raise ValueError("task_id must be non-negative")
    return task_id


def _observation_storage(
    t: int,
    n: int,
    store_device: str,
    storage_path: Optional[str | Path],
    observation_dtype: str,
) -> ImageT:
    try:
        dtype = _OBSERVATION_DTYPES[observation_dtype]
    except KeyError as exc:
        raise ValueError(
            f"Unknown replay observation dtype: {observation_dtype!r}"
        ) from exc
    shape = (t, n, 3, 64, 64)
    if storage_path is None:
        return torch.zeros(*shape, dtype=dtype, device=store_device)
    if torch.device(store_device).type != "cpu":
        raise ValueError("Mapped replay observations require CPU storage")
    from clworldmodel.replay.mapped_tensor import create_file_backed_tensor

    return create_file_backed_tensor(
        storage_path,
        shape,
        dtype=dtype,
    )


class FifoReplay(Replay):
    def __init__(
        self,
        t: int,
        n: int,
        n_acts: int,
        store_device: str = "cpu",
        *,
        store_task_ids: bool = False,
        observation_storage_path: Optional[str | Path] = None,
        observation_dtype: str = "float32",
    ) -> None:
        super().__init__()

        self.t = t
        self.n = n
        self.n_idx = 0
        self.n_valid = 0
        self.acts: ActionT = torch.zeros(t, n, n_acts).to(store_device)
        self.obss: ImageT = _observation_storage(
            t,
            n,
            store_device,
            observation_storage_path,
            observation_dtype,
        )
        self.observation_dtype = observation_dtype
        self.observation_storage_path = (
            None
            if observation_storage_path is None
            else Path(observation_storage_path).expanduser().resolve()
        )
        self.rews: RewardT = torch.zeros(t, n, 1).to(store_device)
        self.conts: ContT = torch.zeros(t, n, 1).to(store_device)
        self.resets: ResetT = torch.zeros(t, n, 1).to(store_device)
        self.task_ids = (
            torch.full((n,), -1, dtype=torch.long, device=store_device)
            if store_task_ids
            else None
        )

    def add(
        self,
        acts: ActionT,
        obss: ImageT,
        rews: RewardT,
        conts: ContT,
        resets: ResetT,
        task_id: Optional[int] = None,
    ) -> list[int]:
        # Incoming shapes [ T N ... ]
        assert acts.shape[0] == self.t
        data_n = acts.shape[1]
        stored_task_id = _validated_task_id(self.task_ids, task_id)
        slots = [int((self.n_idx + offset) % self.n) for offset in range(data_n)]
        stored_obss = self._encode_observations(obss, self.obss.dtype)

        if self.n_idx + data_n <= self.n:
            self.acts[:, self.n_idx : self.n_idx + data_n] = acts
            self.obss[:, self.n_idx : self.n_idx + data_n] = stored_obss
            self.rews[:, self.n_idx : self.n_idx + data_n] = rews
            self.conts[:, self.n_idx : self.n_idx + data_n] = conts
            self.resets[:, self.n_idx : self.n_idx + data_n] = resets
            if self.task_ids is not None:
                self.task_ids[self.n_idx : self.n_idx + data_n] = stored_task_id
        else:
            n1 = self.n - self.n_idx
            n2 = data_n - n1
            self.acts[:, self.n_idx :] = acts[:, :n1]
            self.obss[:, self.n_idx :] = stored_obss[:, :n1]
            self.rews[:, self.n_idx :] = rews[:, :n1]
            self.conts[:, self.n_idx :] = conts[:, :n1]
            self.resets[:, self.n_idx :] = resets[:, :n1]
            if self.task_ids is not None:
                self.task_ids[self.n_idx :] = stored_task_id

            self.acts[:, :n2] = acts[:, -n2:]
            self.obss[:, :n2] = stored_obss[:, -n2:]
            self.rews[:, :n2] = rews[:, -n2:]
            self.conts[:, :n2] = conts[:, -n2:]
            self.resets[:, :n2] = resets[:, -n2:]
            if self.task_ids is not None:
                self.task_ids[:n2] = stored_task_id

        self.n_idx = (self.n_idx + data_n) % self.n
        self.n_valid = min(self.n_valid + data_n, self.n)
        self._refresh_valid_slots_by_task()
        return slots

    def state_dict(self) -> dict[str, object]:
        return {**self._trajectory_state_dict(), "n_idx": self.n_idx}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self._load_trajectory_state_dict(state)
        n_idx = int(state["n_idx"])
        if not 0 <= n_idx < self.n:
            raise ValueError("FIFO replay checkpoint n_idx is outside capacity")
        self.n_idx = n_idx


class LongTermReplay(Replay):
    Priority = float
    NIndex = int

    def __init__(
        self,
        t: int,
        n: int,
        n_acts: int,
        store_device: str = "cpu",
        *,
        store_task_ids: bool = False,
        observation_storage_path: Optional[str | Path] = None,
        observation_dtype: str = "float32",
    ) -> None:
        super().__init__()

        self.t = t
        self.n = n
        self.acts: ActionT = torch.zeros(t, n, n_acts).to(store_device)
        self.obss: ImageT = _observation_storage(
            t,
            n,
            store_device,
            observation_storage_path,
            observation_dtype,
        )
        self.observation_dtype = observation_dtype
        self.observation_storage_path = (
            None
            if observation_storage_path is None
            else Path(observation_storage_path).expanduser().resolve()
        )
        self.rews: RewardT = torch.zeros(t, n, 1).to(store_device)
        self.conts: ContT = torch.zeros(t, n, 1).to(store_device)
        self.resets: ResetT = torch.zeros(t, n, 1).to(store_device)
        self.task_ids = (
            torch.full((n,), -1, dtype=torch.long, device=store_device)
            if store_task_ids
            else None
        )

        self.collection = SortedList([(float("-inf"), _n) for _n in range(n)])
        self.n_valid = 0

    def add(
        self,
        acts: ActionT,
        obss: ImageT,
        rews: RewardT,
        conts: ContT,
        resets: ResetT,
        task_id: Optional[int] = None,
    ) -> list[int]:
        assert acts.shape[0] == self.t
        data_n = acts.shape[1]
        stored_task_id = _validated_task_id(self.task_ids, task_id)
        slots = [-1 for _ in range(data_n)]
        stored_obss = self._encode_observations(obss, self.obss.dtype)

        for n in range(data_n):
            least_prio, least_index = self.collection[0]
            rand_prio = np.random.randn()
            if rand_prio > least_prio:
                del self.collection[0]
                self.collection.add((rand_prio, least_index))
                self.n_valid = min(self.n, self.n_valid + 1)

                self.acts[:, least_index] = acts[:, n]
                self.obss[:, least_index] = stored_obss[:, n]
                self.rews[:, least_index] = rews[:, n]
                self.conts[:, least_index] = conts[:, n]
                self.resets[:, least_index] = resets[:, n]
                if self.task_ids is not None:
                    self.task_ids[least_index] = stored_task_id
                slots[n] = least_index
        self._refresh_valid_slots_by_task()
        return slots

    def state_dict(self) -> dict[str, object]:
        return {
            **self._trajectory_state_dict(),
            "collection": list(self.collection),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self._load_trajectory_state_dict(state)
        collection = state.get("collection")
        if not isinstance(collection, list) or len(collection) != self.n:
            raise ValueError("LTDM replay checkpoint has an invalid key collection")
        self.collection = SortedList(
            (float(priority), int(index)) for priority, index in collection
        )


class MultiTypeReplay(Replay):
    def __init__(
        self,
        *replays: Replay,
        sampling_weights: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        self.replays = replays
        if sampling_weights is None:
            self.sampling_weights = tuple(1.0 / len(replays) for _ in replays)
        else:
            assert len(sampling_weights) == len(replays)
            self.sampling_weights = sampling_weights

    @property
    def n_valid(self) -> int:
        return self.replays[0].n_valid

    @n_valid.setter
    def n_valid(self, _: int) -> None:
        return

    def add(
        self,
        acts: ActionT,
        obss: ImageT,
        rews: RewardT,
        conts: ContT,
        resets: ResetT,
        task_id: Optional[int] = None,
    ) -> tuple[list[int], ...]:
        observation_dtypes = {
            replay.obss.dtype
            for replay in self.replays
            if hasattr(replay, "obss")
        }
        if len(observation_dtypes) == 1:
            obss = self._encode_observations(obss, observation_dtypes.pop())
        return tuple(
            replay.add(acts, obss, rews, conts, resets, task_id=task_id)
            for replay in self.replays
        )

    def minibatch_with_metadata(
        self,
        mb_t: int,
        mb_n: int,
        mb_device: str = "cuda",
        task_id: Optional[int] = None,
    ) -> tuple[ActionT, ImageT, RewardT, ContT, ResetT, int, np.ndarray, np.ndarray]:
        """Sample with the selected sub-buffer and its source indices."""
        if task_id is None:
            # Preserve upstream sub-buffer selection and RNG consumption exactly.
            replay = random.choices(self.replays, weights=self.sampling_weights, k=1)[0]
            replay_index = self.replays.index(replay)
        else:
            eligible_indices = [
                index
                for index, replay in enumerate(self.replays)
                if task_id in replay.available_task_ids()
            ]
            if not eligible_indices:
                raise ValueError(f"ARROW replay contains no sequences for task {task_id}")
            replay_index = random.choices(
                eligible_indices,
                weights=[self.sampling_weights[index] for index in eligible_indices],
                k=1,
            )[0]
            replay = self.replays[replay_index]
        acts, obss, rews, conts, resets, t_starts, ns = replay.minibatch_with_metadata(
            mb_t, mb_n, mb_device, task_id=task_id
        )
        return acts, obss, rews, conts, resets, replay_index, t_starts, ns

    def minibatch_for_task(
        self,
        task_id: int,
        sequence_length: int,
        sequences: int,
        source: Literal["fifo", "ltdm", "mixed"] = "mixed",
        mb_device: str = "cuda",
    ) -> TaskReplayBatch:
        """Select the requested ARROW sub-buffer, then sample cached task slots."""

        if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0:
            raise ValueError("task_id must be a non-negative integer")
        if sequence_length < 1 or sequences < 1:
            raise ValueError("Replay sequence length and count must be positive")
        if source == "mixed":
            return TaskReplayBatch(
                self.minibatch(
                    sequence_length,
                    sequences,
                    mb_device,
                    task_id=task_id,
                ),
                task_id,
            )
        replay_type = (
            FifoReplay
            if source == "fifo"
            else LongTermReplay
            if source == "ltdm"
            else None
        )
        if replay_type is None:
            raise ValueError(f"Unknown replay source: {source!r}")
        candidates = [
            replay
            for replay in self.replays
            if isinstance(replay, replay_type)
            and task_id in replay.valid_slots_by_task
        ]
        if not candidates:
            raise ValueError(
                f"ARROW {source} replay contains no sequences for task {task_id}"
            )
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected one eligible {source} replay, found {len(candidates)}"
            )
        return candidates[0].minibatch_for_task(
            task_id,
            sequence_length,
            sequences,
            source=source,
            mb_device=mb_device,
        )

    def available_task_ids(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    task_id
                    for replay in self.replays
                    for task_id in replay.available_task_ids()
                }
            )
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "replay_type": type(self).__name__,
            "sampling_weights": self.sampling_weights,
            "replays": [replay.state_dict() for replay in self.replays],
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("replay_type") != type(self).__name__:
            raise ValueError("Replay checkpoint type does not match MultiTypeReplay")
        replay_states = state.get("replays")
        if not isinstance(replay_states, list) or len(replay_states) != len(self.replays):
            raise ValueError("Replay checkpoint sub-buffer count does not match")
        weights = tuple(float(value) for value in state["sampling_weights"])
        if weights != tuple(self.sampling_weights):
            raise ValueError("Replay checkpoint sampling weights do not match")
        for replay, replay_state in zip(self.replays, replay_states):
            replay.load_state_dict(replay_state)
