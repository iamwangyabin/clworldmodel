"""Bridge ARROW trajectory replay to native R2-Dreamer training batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from clworldmodel.r2dreamer.agent import R2ReplayBatch, R2UpdateResult
from clworldmodel.r2dreamer.config import R2DreamerConfig


class _ArrowSubReplay(Protocol):
    t: int
    n: int


class _ArrowReplay(Protocol):
    n_valid: int
    replays: tuple[_ArrowSubReplay, ...]

    def add(
        self,
        actions: torch.Tensor,
        observations: torch.Tensor,
        rewards: torch.Tensor,
        continues: torch.Tensor,
        resets: torch.Tensor,
    ) -> tuple[list[int], ...]: ...

    def minibatch_with_metadata(
        self, mb_t: int, mb_n: int, mb_device: str
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        object,
        object,
    ]: ...


@dataclass(frozen=True)
class ArrowR2SampleReference:
    """Source slots and time offsets for one R2 replay minibatch."""

    replay_index: int
    time_starts: torch.Tensor
    sequence_indices: torch.Tensor


@dataclass(frozen=True)
class ArrowR2Sample:
    """A native R2 batch and the ARROW source locations behind it."""

    batch: R2ReplayBatch
    reference: ArrowR2SampleReference


class ArrowR2ReplayAdapter:
    """ARROW replay with R2's one-step context and latent-state sidecar.

    ARROW remains responsible for FIFO/LTDM retention and its 50/50 whole-
    minibatch selection. This adapter adds the state cache required by the
    upstream R2-Dreamer replay contract: a sampled context transition supplies
    the posterior state at time ``t``; R2 trains on observations ``t+1...``
    with actions ``t...`` and writes its refreshed posterior states back to the
    corresponding ARROW storage positions.

    The cache is project-owned metadata rather than an ARROW retention field.
    Its device and byte footprint are explicit in the run artifact.
    """

    def __init__(
        self,
        replay: _ArrowReplay,
        config: R2DreamerConfig,
        *,
        state_device: str | torch.device = "cpu",
    ) -> None:
        self._replay = replay
        self.config = config
        self._state_device = torch.device(state_device)
        self._sub_replays = tuple(replay.replays)
        if not self._sub_replays:
            raise ValueError("ARROW replay must contain at least one sub-buffer")
        if any(sub_replay.t < self.sequence_length_with_context for sub_replay in self._sub_replays):
            raise ValueError("ARROW replay trajectories are shorter than the R2 sample context")

        self._stoch_states = tuple(
            torch.zeros(
                sub_replay.t,
                sub_replay.n,
                config.stoch,
                config.discrete,
                dtype=torch.float32,
                device=self._state_device,
            )
            for sub_replay in self._sub_replays
        )
        self._deter_states = tuple(
            torch.zeros(
                sub_replay.t,
                sub_replay.n,
                config.deter,
                dtype=torch.float32,
                device=self._state_device,
            )
            for sub_replay in self._sub_replays
        )
        self._last_states = tuple(
            torch.zeros(
                sub_replay.t,
                sub_replay.n,
                dtype=torch.bool,
                device=self._state_device,
            )
            for sub_replay in self._sub_replays
        )

    @property
    def sequence_length_with_context(self) -> int:
        return self.config.batch_length + 1

    @property
    def state_device(self) -> torch.device:
        return self._state_device

    @property
    def latent_state_storage_bytes(self) -> int:
        return sum(
            states.numel() * states.element_size()
            for states in (*self._stoch_states, *self._deter_states)
        )

    @property
    def transition_metadata_storage_bytes(self) -> int:
        """Bytes for R2's `is_last` labels that ARROW does not store itself."""
        return sum(states.numel() * states.element_size() for states in self._last_states)

    def storage_accounting(self) -> dict[str, object]:
        """Return explicit transition and R2 sidecar byte accounting."""
        transition_tensors = ("acts", "obss", "rews", "conts", "resets")
        transition_bytes = 0
        sub_buffers = []
        for index, sub_replay in enumerate(self._sub_replays):
            bytes_for_sub_buffer = sum(
                getattr(sub_replay, name).numel() * getattr(sub_replay, name).element_size()
                for name in transition_tensors
            )
            transition_bytes += bytes_for_sub_buffer
            sub_buffers.append(
                {
                    "index": index,
                    "type": type(sub_replay).__name__,
                    "trajectory_length": sub_replay.t,
                    "trajectory_slots": sub_replay.n,
                    "transition_storage_bytes": bytes_for_sub_buffer,
                    "latent_state_storage_bytes": (
                        self._stoch_states[index].numel() * self._stoch_states[index].element_size()
                        + self._deter_states[index].numel() * self._deter_states[index].element_size()
                    ),
                    "transition_metadata_storage_bytes": (
                        self._last_states[index].numel() * self._last_states[index].element_size()
                    ),
                }
            )
        return {
            "schema_version": 1,
            "arrow_transition_storage_bytes": transition_bytes,
            "r2_latent_state_storage_bytes": self.latent_state_storage_bytes,
            "r2_transition_metadata_storage_bytes": self.transition_metadata_storage_bytes,
            "r2_sidecar_storage_bytes": (
                self.latent_state_storage_bytes + self.transition_metadata_storage_bytes
            ),
            "combined_storage_bytes": (
                transition_bytes
                + self.latent_state_storage_bytes
                + self.transition_metadata_storage_bytes
            ),
            "latent_state_dtype": "float32",
            "latent_state_device": str(self._state_device),
            "transition_metadata_dtype": "bool",
            "sub_buffers": sub_buffers,
        }

    def add(
        self,
        actions: torch.Tensor,
        observations: torch.Tensor,
        rewards: torch.Tensor,
        continues: torch.Tensor,
        resets: torch.Tensor,
        is_last: torch.Tensor,
        stoch_states: torch.Tensor,
        deter_states: torch.Tensor,
    ) -> None:
        """Add ARROW transitions and posterior states using identical slots."""
        if stoch_states.shape != (
            actions.shape[0],
            actions.shape[1],
            self.config.stoch,
            self.config.discrete,
        ):
            raise ValueError("collected posterior stochastic states do not match ARROW data")
        if deter_states.shape != (
            actions.shape[0],
            actions.shape[1],
            self.config.deter,
        ):
            raise ValueError("collected posterior deterministic states do not match ARROW data")
        if is_last.shape != (actions.shape[0], actions.shape[1], 1):
            raise ValueError("collected is_last labels do not match ARROW data")
        write_slots = self._replay.add(actions, observations, rewards, continues, resets)
        if len(write_slots) != len(self._sub_replays):
            raise RuntimeError("ARROW replay did not report one write map per sub-buffer")

        stoch_states = stoch_states.detach().to(self._state_device, dtype=torch.float32)
        deter_states = deter_states.detach().to(self._state_device, dtype=torch.float32)
        is_last = is_last.detach().to(self._state_device, dtype=torch.bool).squeeze(-1)
        for replay_index, slots in enumerate(write_slots):
            if len(slots) != actions.shape[1]:
                raise RuntimeError("ARROW replay write map has an unexpected sequence count")
            for incoming_index, stored_index in enumerate(slots):
                if stored_index < 0:
                    continue
                self._stoch_states[replay_index][:, stored_index].copy_(
                    stoch_states[:, incoming_index]
                )
                self._deter_states[replay_index][:, stored_index].copy_(
                    deter_states[:, incoming_index]
                )
                self._last_states[replay_index][:, stored_index].copy_(
                    is_last[:, incoming_index]
                )

    def sample(self) -> ArrowR2Sample:
        if self._replay.n_valid < 1:
            raise RuntimeError("Cannot sample R2-Dreamer before ARROW replay has data")
        (
            actions,
            observations,
            rewards,
            continues,
            resets,
            replay_index,
            time_starts,
            sequence_indices,
        ) = self._replay.minibatch_with_metadata(
            self.sequence_length_with_context,
            self.config.batch_size,
            self.config.device,
        )
        expected_time = self.sequence_length_with_context
        if actions.shape[:2] != (expected_time, self.config.batch_size):
            raise RuntimeError("ARROW replay returned an unexpected time/batch shape")
        if replay_index < 0 or replay_index >= len(self._sub_replays):
            raise RuntimeError("ARROW replay returned an invalid sub-buffer index")

        time_starts = torch.as_tensor(time_starts, dtype=torch.long, device=self._state_device)
        sequence_indices = torch.as_tensor(
            sequence_indices,
            dtype=torch.long,
            device=self._state_device,
        )
        if time_starts.shape != (self.config.batch_size,) or sequence_indices.shape != (
            self.config.batch_size,
        ):
            raise RuntimeError("ARROW replay returned unexpected source-index shapes")
        if torch.any(time_starts < 0) or torch.any(
            time_starts + self.config.batch_length >= self._sub_replays[replay_index].t
        ):
            raise RuntimeError("ARROW replay returned out-of-range source time offsets")
        initial_stoch = self._stoch_states[replay_index][time_starts, sequence_indices].to(
            self.config.device,
            non_blocking=self._state_device.type == "cpu",
        )
        initial_deter = self._deter_states[replay_index][time_starts, sequence_indices].to(
            self.config.device,
            non_blocking=self._state_device.type == "cpu",
        )
        time_offsets = torch.arange(
            1,
            self.config.batch_length + 1,
            device=self._state_device,
            dtype=torch.long,
        )
        time_indices = time_starts[:, None] + time_offsets[None, :]
        sampled_sequences = sequence_indices[:, None].expand_as(time_indices)
        is_last = self._last_states[replay_index][time_indices, sampled_sequences].to(
            self.config.device,
            non_blocking=self._state_device.type == "cpu",
        )
        batch = R2ReplayBatch(
            images=observations[1:].swapaxes(0, 1).movedim(2, -1).contiguous(),
            actions=actions[:-1].swapaxes(0, 1).contiguous(),
            rewards=rewards[1:].swapaxes(0, 1).contiguous(),
            continues=continues[1:].swapaxes(0, 1).contiguous(),
            is_first=resets[1:].swapaxes(0, 1).squeeze(-1).to(dtype=torch.bool),
            is_last=is_last,
            initial_stoch=initial_stoch,
            initial_deter=initial_deter,
        )
        batch.validate(self.config)
        return ArrowR2Sample(
            batch=batch,
            reference=ArrowR2SampleReference(
                replay_index=replay_index,
                time_starts=time_starts,
                sequence_indices=sequence_indices,
            ),
        )

    def update_latent_states(
        self,
        reference: ArrowR2SampleReference,
        update: R2UpdateResult,
    ) -> None:
        """Write the pre-optimizer posterior rollout back to the source slots."""
        if update.posterior_stoch.shape != (
            self.config.batch_size,
            self.config.batch_length,
            self.config.stoch,
            self.config.discrete,
        ):
            raise ValueError("R2 posterior stochastic states do not match the sampled batch")
        if update.posterior_deter.shape != (
            self.config.batch_size,
            self.config.batch_length,
            self.config.deter,
        ):
            raise ValueError("R2 posterior deterministic states do not match the sampled batch")
        time_offsets = torch.arange(
            1,
            self.config.batch_length + 1,
            device=self._state_device,
            dtype=torch.long,
        )
        time_indices = reference.time_starts[:, None] + time_offsets[None, :]
        sequence_indices = reference.sequence_indices[:, None].expand_as(time_indices)
        stoch = update.posterior_stoch.detach().to(self._state_device, dtype=torch.float32)
        deter = update.posterior_deter.detach().to(self._state_device, dtype=torch.float32)
        self._stoch_states[reference.replay_index][
            time_indices.reshape(-1), sequence_indices.reshape(-1)
        ] = stoch.reshape(-1, self.config.stoch, self.config.discrete)
        self._deter_states[reference.replay_index][
            time_indices.reshape(-1), sequence_indices.reshape(-1)
        ] = deter.reshape(-1, self.config.deter)
