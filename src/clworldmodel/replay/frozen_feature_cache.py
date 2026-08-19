"""Sidecar replay storage for deterministic frozen visual features."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch


class ArrowFrozenFeatureCache:
    """Keep frozen features aligned with ARROW FIFO/LTDM slot decisions."""

    def __init__(
        self,
        replay: Any,
        feature_dim: int,
        *,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if dtype not in {torch.float16, torch.float32}:
            raise ValueError("feature cache dtype must be float16 or float32")
        self.replay = replay
        self.feature_dim = feature_dim
        self.dtype = dtype
        self._sub_replays = tuple(replay.replays)
        if not self._sub_replays:
            raise ValueError("ARROW replay must contain at least one sub-buffer")
        self._features = tuple(
            torch.zeros(
                sub_replay.t,
                sub_replay.n,
                feature_dim,
                dtype=dtype,
                device=sub_replay.obss.device,
            )
            for sub_replay in self._sub_replays
        )

    @property
    def storage_bytes(self) -> int:
        return sum(values.numel() * values.element_size() for values in self._features)

    def storage_accounting(self) -> dict[str, object]:
        buffers = []
        for index, (sub_replay, features) in enumerate(
            zip(self._sub_replays, self._features)
        ):
            buffers.append(
                {
                    "index": index,
                    "type": type(sub_replay).__name__,
                    "device": str(features.device),
                    "trajectory_length": sub_replay.t,
                    "trajectory_slots": sub_replay.n,
                    "feature_storage_bytes": features.numel() * features.element_size(),
                }
            )
        return {
            "schema_version": 1,
            "feature_dim": self.feature_dim,
            "dtype": str(self.dtype).removeprefix("torch."),
            "storage_bytes": self.storage_bytes,
            "retention_semantics": "sidecar follows existing ARROW write maps",
            "sampling_semantics": "sidecar follows existing ARROW sampled indices",
            "buffers": buffers,
        }

    def record(
        self,
        write_slots: Sequence[Sequence[int]],
        features: torch.Tensor,
    ) -> None:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                "features must have shape [time, sequences, feature_dim], "
                f"got {tuple(features.shape)}"
            )
        if len(write_slots) != len(self._features):
            raise ValueError("ARROW write map count does not match feature buffers")
        if any(len(slots) != features.shape[1] for slots in write_slots):
            raise ValueError("ARROW write maps do not match incoming sequence count")

        detached = features.detach()
        for replay_index, slots in enumerate(write_slots):
            target = self._features[replay_index]
            for incoming_index, stored_index in enumerate(slots):
                if stored_index < 0:
                    continue
                target[:, stored_index].copy_(
                    detached[:, incoming_index].to(
                        device=target.device,
                        dtype=self.dtype,
                    )
                )

    def minibatch(
        self,
        mb_t: int,
        mb_n: int,
        mb_device: str | torch.device = "cuda",
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        sample = self.replay.minibatch_with_metadata(mb_t, mb_n, str(mb_device))
        if len(sample) != 8:
            raise RuntimeError("Frozen feature cache requires ARROW MultiTypeReplay metadata")
        (
            actions,
            observations,
            rewards,
            continues,
            resets,
            replay_index,
            time_starts,
            sequence_indices,
        ) = sample
        if replay_index < 0 or replay_index >= len(self._features):
            raise RuntimeError("ARROW returned an invalid sub-buffer index")
        feature_values = self._gather(
            self._features[replay_index],
            time_starts,
            sequence_indices,
            actions.shape[0],
        ).to(mb_device, dtype=torch.float32)
        return actions, observations, feature_values, rewards, continues, resets

    @staticmethod
    def _gather(
        source: torch.Tensor,
        time_starts: np.ndarray,
        sequence_indices: np.ndarray,
        time_size: int,
    ) -> torch.Tensor:
        if len(time_starts) != len(sequence_indices):
            raise ValueError("Sampled time and sequence metadata lengths must match")
        return torch.stack(
            [
                source[int(start) : int(start) + time_size, int(sequence)]
                for start, sequence in zip(time_starts, sequence_indices)
            ],
            dim=1,
        )
