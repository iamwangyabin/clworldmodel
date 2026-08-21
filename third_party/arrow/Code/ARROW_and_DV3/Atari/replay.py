import random
from typing import Optional

import numpy as np
import torch
from sortedcontainers import SortedList

from rssm import ActionT, ContT, ImageT, ResetT
from wm import RewardT


class Replay:
    def __init__(self) -> None:
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
            mb_obss.to(mb_device),
            mb_rews.to(mb_device),
            mb_conts.to(mb_device),
            mb_resets.to(mb_device),
            t_starts,
            ns,
        )

    def _eligible_sequence_indices(self, task_id: int) -> np.ndarray:
        if task_id < 0:
            raise ValueError("task_id must be non-negative")
        if self.task_ids is None:
            raise ValueError("Replay was constructed without task-id storage")
        return np.flatnonzero(
            self.task_ids[: self.n_valid].detach().cpu().numpy() == task_id
        )

    def available_task_ids(self) -> tuple[int, ...]:
        if self.n_valid == 0 or self.task_ids is None:
            return ()
        task_ids = self.task_ids[: self.n_valid].detach().cpu().unique().tolist()
        return tuple(sorted(int(task_id) for task_id in task_ids if task_id >= 0))


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


class FifoReplay(Replay):
    def __init__(
        self,
        t: int,
        n: int,
        n_acts: int,
        store_device: str = "cpu",
        *,
        store_task_ids: bool = False,
    ) -> None:
        super().__init__()

        self.t = t
        self.n = n
        self.n_idx = 0
        self.n_valid = 0
        self.acts: ActionT = torch.zeros(t, n, n_acts).to(store_device)
        self.obss: ImageT = torch.zeros(t, n, 3, 64, 64).to(store_device)
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

        if self.n_idx + data_n <= self.n:
            self.acts[:, self.n_idx : self.n_idx + data_n] = acts
            self.obss[:, self.n_idx : self.n_idx + data_n] = obss
            self.rews[:, self.n_idx : self.n_idx + data_n] = rews
            self.conts[:, self.n_idx : self.n_idx + data_n] = conts
            self.resets[:, self.n_idx : self.n_idx + data_n] = resets
            if self.task_ids is not None:
                self.task_ids[self.n_idx : self.n_idx + data_n] = stored_task_id
        else:
            n1 = self.n - self.n_idx
            n2 = data_n - n1
            self.acts[:, self.n_idx :] = acts[:, :n1]
            self.obss[:, self.n_idx :] = obss[:, :n1]
            self.rews[:, self.n_idx :] = rews[:, :n1]
            self.conts[:, self.n_idx :] = conts[:, :n1]
            self.resets[:, self.n_idx :] = resets[:, :n1]
            if self.task_ids is not None:
                self.task_ids[self.n_idx :] = stored_task_id

            self.acts[:, :n2] = acts[:, -n2:]
            self.obss[:, :n2] = obss[:, -n2:]
            self.rews[:, :n2] = rews[:, -n2:]
            self.conts[:, :n2] = conts[:, -n2:]
            self.resets[:, :n2] = resets[:, -n2:]
            if self.task_ids is not None:
                self.task_ids[:n2] = stored_task_id

        self.n_idx = (self.n_idx + data_n) % self.n
        self.n_valid = min(self.n_valid + data_n, self.n)
        return slots


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
    ) -> None:
        super().__init__()

        self.t = t
        self.n = n
        self.acts: ActionT = torch.zeros(t, n, n_acts).to(store_device)
        self.obss: ImageT = torch.zeros(t, n, 3, 64, 64).to(store_device)
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

        for n in range(data_n):
            least_prio, least_index = self.collection[0]
            rand_prio = np.random.randn()
            if rand_prio > least_prio:
                del self.collection[0]
                self.collection.add((rand_prio, least_index))
                self.n_valid = min(self.n, self.n_valid + 1)

                self.acts[:, least_index] = acts[:, n]
                self.obss[:, least_index] = obss[:, n]
                self.rews[:, least_index] = rews[:, n]
                self.conts[:, least_index] = conts[:, n]
                self.resets[:, least_index] = resets[:, n]
                if self.task_ids is not None:
                    self.task_ids[least_index] = stored_task_id
                slots[n] = least_index
        return slots


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
