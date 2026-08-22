"""Sidecar replay storage for deterministic frozen visual features."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .mapped_tensor import create_file_backed_tensor


class ArrowFrozenFeatureCache:
    """Keep frozen features aligned with ARROW FIFO/LTDM slot decisions."""

    requires_recording = True

    def __init__(
        self,
        replay: Any,
        feature_dim: int,
        *,
        dtype: torch.dtype = torch.float16,
        storage_directory: str | Path | None = None,
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
        storage_root = (
            None
            if storage_directory is None
            else Path(storage_directory).expanduser().resolve()
        )
        if storage_root is not None:
            storage_root.mkdir(parents=True, exist_ok=True)
        storage_paths = []
        features = []
        for index, sub_replay in enumerate(self._sub_replays):
            shape = (sub_replay.t, sub_replay.n, feature_dim)
            if storage_root is None:
                storage_path = None
                values = torch.zeros(
                    *shape,
                    dtype=dtype,
                    device=sub_replay.obss.device,
                )
            else:
                if sub_replay.obss.device.type != "cpu":
                    raise ValueError("Mapped feature storage requires CPU replay")
                dtype_name = str(dtype).removeprefix("torch.")
                storage_path = storage_root / (
                    f"{index}_{type(sub_replay).__name__}_features.{dtype_name}.mmap"
                )
                values = create_file_backed_tensor(
                    storage_path,
                    shape,
                    dtype=dtype,
                )
            storage_paths.append(storage_path)
            features.append(values)
        self._storage_paths = tuple(storage_paths)
        self._features = tuple(features)

    @property
    def storage_paths(self) -> tuple[Path | None, ...]:
        return self._storage_paths

    @staticmethod
    def _allocated_file_bytes(path: Path | None) -> int | None:
        if path is None:
            return None
        stat = path.stat()
        return stat.st_blocks * 512

    @property
    def storage_backend(self) -> str:
        return (
            "file_mmap"
            if any(path is not None for path in self._storage_paths)
            else "anonymous_cpu"
        )

    @property
    def storage_bytes(self) -> int:
        return sum(values.numel() * values.element_size() for values in self._features)

    def storage_accounting(self) -> dict[str, object]:
        buffers = []
        for index, (sub_replay, features, storage_path) in enumerate(
            zip(self._sub_replays, self._features, self._storage_paths)
        ):
            buffers.append(
                {
                    "index": index,
                    "type": type(sub_replay).__name__,
                    "device": str(features.device),
                    "trajectory_length": sub_replay.t,
                    "trajectory_slots": sub_replay.n,
                    "feature_storage_bytes": features.numel() * features.element_size(),
                    "storage_backend": (
                        "file_mmap" if storage_path is not None else "anonymous_cpu"
                    ),
                    "storage_path": (
                        str(storage_path) if storage_path is not None else None
                    ),
                    "allocated_file_bytes_at_accounting": (
                        self._allocated_file_bytes(storage_path)
                    ),
                }
            )
        return {
            "schema_version": 1,
            "feature_dim": self.feature_dim,
            "dtype": str(self.dtype).removeprefix("torch."),
            "storage_bytes": self.storage_bytes,
            "storage_backend": self.storage_backend,
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
        *,
        task_id: int | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if task_id is None:
            sample = self.replay.minibatch_with_metadata(
                mb_t, mb_n, str(mb_device)
            )
        else:
            sample = self.replay.minibatch_with_metadata(
                mb_t,
                mb_n,
                str(mb_device),
                task_id=task_id,
            )
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


class ArrowOnTheFlyFeatureSource:
    """Recompute frozen features from sampled ARROW observations on the GPU."""

    requires_recording = False

    def __init__(
        self,
        replay: Any,
        encoder: torch.nn.Module,
        feature_dim: int,
        *,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if dtype not in {torch.float16, torch.float32}:
            raise ValueError("feature replay dtype must be float16 or float32")
        self.replay = replay
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.dtype = dtype

    @property
    def storage_backend(self) -> str:
        return "on_the_fly"

    @property
    def storage_bytes(self) -> int:
        return 0

    def storage_accounting(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "feature_dim": self.feature_dim,
            "dtype": str(self.dtype).removeprefix("torch."),
            "storage_bytes": 0,
            "storage_backend": self.storage_backend,
            "source": "sampled ARROW observations",
            "retention_semantics": "no separately retained feature values",
            "sampling_semantics": "features use the existing ARROW sampled observations",
            "quantization_semantics": (
                "encoder outputs round through the configured replay dtype before "
                "RSSM float32 consumption"
            ),
            "buffers": [],
        }

    @torch.no_grad()
    def minibatch(
        self,
        mb_t: int,
        mb_n: int,
        mb_device: str | torch.device = "cuda",
        *,
        task_id: int | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if task_id is None:
            sample = self.replay.minibatch_with_metadata(
                mb_t, mb_n, str(mb_device)
            )
        else:
            sample = self.replay.minibatch_with_metadata(
                mb_t,
                mb_n,
                str(mb_device),
                task_id=task_id,
            )
        if len(sample) != 8:
            raise RuntimeError(
                "On-the-fly features require ARROW MultiTypeReplay metadata"
            )
        actions, observations, rewards, continues, resets = sample[:5]
        time, sequences = observations.shape[:2]
        flat_observations = observations.reshape(-1, *observations.shape[-3:])
        flat_features = self.encoder(flat_observations).detach()
        if flat_features.shape != (time * sequences, self.feature_dim):
            raise RuntimeError(
                "Frozen encoder returned unexpected replay feature shape: "
                f"expected {(time * sequences, self.feature_dim)}, "
                f"got {tuple(flat_features.shape)}"
            )
        features = flat_features.to(self.dtype).to(torch.float32).view(
            time, sequences, self.feature_dim
        )
        return actions, observations, features, rewards, continues, resets
