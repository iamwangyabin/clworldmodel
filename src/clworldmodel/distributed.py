"""Synchronous multi-GPU helpers for fixed-global-batch ARROW training."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

SUPPORTED_DATA_PARALLEL_WORLD_SIZES = (1, 2, 4)


def local_sequence_count(global_sequences: int, world_size: int) -> int:
    """Return the equal per-rank sequence count for a fixed global batch."""
    if world_size not in SUPPORTED_DATA_PARALLEL_WORLD_SIZES:
        raise ValueError(
            "data-parallel world size must be one of "
            f"{SUPPORTED_DATA_PARALLEL_WORLD_SIZES}, got {world_size}"
        )
    if global_sequences < 1:
        raise ValueError("global sequence count must be positive")
    if global_sequences % world_size:
        raise ValueError(
            f"global sequence count {global_sequences} must be divisible by "
            f"world size {world_size}"
        )
    return global_sequences // world_size


def split_sequence_tensor(
    tensor: torch.Tensor, world_size: int
) -> tuple[torch.Tensor, ...]:
    """Split a `[T, N, ...]` tensor along the independent sequence axis."""
    if tensor.ndim < 2:
        raise ValueError("sequence tensors must contain [T, N] axes")
    local_n = local_sequence_count(tensor.shape[1], world_size)
    return tuple(
        tensor[:, rank * local_n : (rank + 1) * local_n].contiguous()
        for rank in range(world_size)
    )


@dataclass(frozen=True)
class DistributedContext:
    """Own one process's torch.distributed topology and collective helpers."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str | None

    @classmethod
    def initialize(cls, expected_world_size: int) -> DistributedContext:
        if expected_world_size not in SUPPORTED_DATA_PARALLEL_WORLD_SIZES:
            raise ValueError(
                "data-parallel world size must be one of "
                f"{SUPPORTED_DATA_PARALLEL_WORLD_SIZES}"
            )

        observed_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if observed_world_size != expected_world_size:
            launcher = (
                f"run the launcher with --devices {expected_world_size}"
                if expected_world_size > 1
                else "remove the torchrun multi-process launch"
            )
            raise RuntimeError(
                "Configured data_parallel_world_size does not match torchrun: "
                f"configured={expected_world_size} observed={observed_world_size}; "
                f"{launcher}"
            )

        if expected_world_size == 1:
            device = (
                torch.device("cuda", torch.cuda.current_device())
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            return cls(0, 0, 1, device, None)

        if not torch.cuda.is_available():
            raise RuntimeError("multi-GPU data parallelism requires CUDA")
        if not dist.is_nccl_available():
            raise RuntimeError("multi-GPU data parallelism requires the NCCL backend")
        try:
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
        except KeyError as exc:
            raise RuntimeError(
                "multi-GPU data parallelism must be launched with torchrun"
            ) from exc
        if local_rank < 0 or local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} is not available; visible CUDA devices="
                f"{torch.cuda.device_count()}"
            )

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=expected_world_size,
            device=torch.device("cuda", local_rank),
            backend="nccl",
        )

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def local_sequences(self, global_sequences: int) -> int:
        return local_sequence_count(global_sequences, self.world_size)

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def close(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()

    def seed_local_torch_stream(self, base_seed: int) -> int:
        """Give each rank independent sampling noise after identical initialization."""
        local_seed = base_seed + self.rank * 1_000_003
        torch.manual_seed(local_seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(local_seed)
        return local_seed

    def wrap_module(self, module: nn.Module) -> nn.Module:
        """Wrap a currently active, static trainable route in native PyTorch DDP."""
        if not self.enabled:
            return module
        from torch.nn.parallel import DistributedDataParallel

        return DistributedDataParallel(
            module,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )

    def all_gather_sequence_batch(
        self, tensor: torch.Tensor, *, sequence_dim: int = 1
    ) -> torch.Tensor:
        """Reconstruct a small global tensor such as actor return statistics."""
        if not self.enabled:
            return tensor
        if tensor.ndim <= sequence_dim:
            raise ValueError("gathered tensor is missing its sequence axis")
        gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor.contiguous())
        return torch.cat(gathered, dim=sequence_dim)

    def mean_float_mapping(self, values: Mapping[str, float]) -> dict[str, float]:
        """Average already-detached scalar metrics with one collective."""
        if not self.enabled:
            return dict(values)
        names = tuple(values)
        packed = torch.tensor(
            [values[name] for name in names], dtype=torch.float64, device=self.device
        )
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        packed.div_(self.world_size)
        return {name: float(value) for name, value in zip(names, packed.cpu().tolist())}

    def mean_tensor_mapping(
        self, values: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Average detached scalar tensors for rank-zero logging."""
        if not self.enabled:
            return dict(values)
        names = tuple(values)
        packed = torch.stack(
            [values[name].detach().float().reshape(()) for name in names]
        )
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        packed.div_(self.world_size)
        return {name: value for name, value in zip(names, packed.unbind())}

    def combine_sparse_task_values(
        self, values: torch.Tensor, present: torch.Tensor
    ) -> torch.Tensor:
        """Combine evaluation results produced by exactly one rank per task."""
        if values.shape != present.shape:
            raise ValueError("task values and presence masks must have equal shapes")
        if self.enabled:
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            dist.all_reduce(present, op=dist.ReduceOp.SUM)
        if not torch.equal(present, torch.ones_like(present)):
            raise RuntimeError(
                "each evaluation task must be assigned to exactly one rank"
            )
        return values


class DistributedReplaySampler:
    """Sample one global ARROW minibatch on rank zero and scatter its N axis."""

    def __init__(
        self,
        context: DistributedContext,
        authoritative_replay: Any,
        *,
        action_space: int,
        num_tasks: int,
        observation_shape: Sequence[int],
    ) -> None:
        if not context.enabled:
            raise ValueError("DistributedReplaySampler requires world_size > 1")
        if context.is_primary and authoritative_replay is None:
            raise ValueError("rank zero requires the authoritative replay")
        if not context.is_primary and authoritative_replay is not None:
            raise ValueError("only rank zero may own the authoritative replay")
        if action_space < 1 or num_tasks < 1:
            raise ValueError("action space and task count must be positive")
        self.context = context
        self.authoritative_replay = authoritative_replay
        self.action_space = action_space
        self.num_tasks = num_tasks
        self.observation_shape = tuple(int(value) for value in observation_shape)

    @property
    def n_valid(self) -> int:
        if not self.context.is_primary:
            raise RuntimeError("only rank zero owns replay validity state")
        return int(self.authoritative_replay.n_valid)

    def add(self, *args: Any, **kwargs: Any) -> Any:
        if not self.context.is_primary:
            raise RuntimeError("only rank zero may write replay")
        return self.authoritative_replay.add(*args, **kwargs)

    def available_task_ids(self) -> tuple[int, ...]:
        task_ids = torch.full(
            (self.num_tasks,), -1, dtype=torch.int64, device=self.context.device
        )
        if self.context.is_primary:
            available = self.authoritative_replay.available_task_ids()
            if len(available) > self.num_tasks:
                raise RuntimeError(
                    "replay returned more task IDs than the schedule owns"
                )
            if available:
                task_ids[: len(available)] = torch.tensor(
                    available, dtype=torch.int64, device=self.context.device
                )
        dist.broadcast(task_ids, src=0)
        return tuple(int(value) for value in task_ids.cpu().tolist() if value >= 0)

    def minibatch(
        self,
        mb_t: int,
        mb_n: int,
        mb_device: str | torch.device = "cuda",
        task_id: int | None = None,
    ) -> tuple[torch.Tensor, ...]:
        return self.minibatch_with_metadata(mb_t, mb_n, mb_device, task_id=task_id)[:5]

    def minibatch_with_metadata(
        self,
        mb_t: int,
        mb_n: int,
        mb_device: str | torch.device = "cuda",
        task_id: int | None = None,
    ) -> tuple[Any, ...]:
        requested_device = torch.device(mb_device)
        if requested_device.type != self.context.device.type:
            raise ValueError(
                "distributed replay batches must target the local training device"
            )
        global_n = mb_n * self.context.world_size
        source_tensors: tuple[torch.Tensor, ...] | None = None
        if self.context.is_primary:
            sample = self.authoritative_replay.minibatch_with_metadata(
                mb_t,
                global_n,
                str(self.context.device),
                task_id=task_id,
            )
            source_tensors = tuple(sample[:5])

        tails = (
            (self.action_space,),
            self.observation_shape,
            (1,),
            (1,),
            (1,),
        )
        local_tensors = []
        for index, tail in enumerate(tails):
            output = torch.empty(
                (mb_t, mb_n, *tail),
                dtype=torch.float32,
                device=self.context.device,
            )
            scatter_list = None
            if source_tensors is not None:
                source = source_tensors[index]
                expected_shape = (mb_t, global_n, *tail)
                if source.shape != expected_shape or source.dtype != torch.float32:
                    raise RuntimeError(
                        "authoritative replay returned an unexpected distributed "
                        f"tensor: expected shape={expected_shape} dtype=float32, "
                        f"got shape={tuple(source.shape)} dtype={source.dtype}"
                    )
                scatter_list = list(
                    split_sequence_tensor(source, self.context.world_size)
                )
            dist.scatter(output, scatter_list=scatter_list, src=0)
            local_tensors.append(output)

        # On-the-fly DINO consumes only the first five entries. Metadata remains
        # deliberately unavailable because it belongs to the rank-zero global draw.
        empty_indices = np.empty(mb_n, dtype=np.int64)
        return (*local_tensors, -1, empty_indices, empty_indices.copy())
