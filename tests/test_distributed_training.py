"""CPU contracts for fixed-global-batch 2/4-GPU execution."""

from __future__ import annotations

import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - GPU image supplies PyTorch
    torch = None

if torch is not None:
    from clworldmodel.distributed import (
        DistributedContext,
        DistributedReplaySampler,
        local_sequence_count,
        split_sequence_tensor,
    )


def _run_cuda_distributed_smoke() -> None:
    """Exercise NCCL replay scatter, BF16 DDP, and gradient reduction."""
    if torch is None:
        raise RuntimeError("distributed smoke requires PyTorch")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    context = DistributedContext.initialize(world_size)

    class Replay:
        def available_task_ids(self):
            return (0,)

        def minibatch_with_metadata(self, mb_t, mb_n, mb_device, task_id=None):
            if task_id != 0:
                raise ValueError("smoke replay expects task 0")
            device = torch.device(mb_device)
            tails = ((3,), (1, 2, 2), (1,), (1,), (1,))
            tensors = tuple(
                torch.arange(
                    mb_t * mb_n * int(torch.tensor(tail).prod()),
                    dtype=torch.float32,
                    device=device,
                ).reshape(mb_t, mb_n, *tail)
                for tail in tails
            )
            indices = torch.arange(mb_n).cpu().numpy()
            return (*tensors, 0, indices, indices.copy())

    try:
        replay = Replay() if context.is_primary else None
        sampler = DistributedReplaySampler(
            context,
            replay,
            action_space=3,
            num_tasks=1,
            observation_shape=(1, 2, 2),
        )
        if sampler.available_task_ids() != (0,):
            raise RuntimeError("task-ID broadcast failed")
        local_n = 2
        actions, observations, *_ = sampler.minibatch(
            2, local_n, context.device, task_id=0
        )
        if observations.shape != (2, local_n, 1, 2, 2):
            raise RuntimeError("replay scatter returned the wrong observation shape")
        gathered_actions = context.all_gather_sequence_batch(actions)
        expected_actions = torch.arange(
            2 * local_n * world_size * 3,
            dtype=torch.float32,
            device=context.device,
        ).reshape(2, local_n * world_size, 3)
        if not torch.equal(gathered_actions, expected_actions):
            raise RuntimeError("replay scatter did not reconstruct the global draw")

        module = torch.nn.Sequential(
            torch.nn.Linear(3, 8),
            torch.nn.SiLU(),
            torch.nn.Linear(8, 1),
        ).to(context.device)
        distributed_module = context.wrap_module(module)
        initial_parameters = [
            parameter.detach().clone() for parameter in module.parameters()
        ]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = distributed_module(actions.mean(dim=0))
        if prediction.dtype != torch.bfloat16:
            raise RuntimeError("BF16 autocast did not reach the DDP module")
        loss = prediction.float().square().mean()
        loss.backward()

        for initial, parameter in zip(initial_parameters, module.parameters()):
            if not torch.equal(initial, parameter):
                raise RuntimeError("smoke must not update model parameters")
            if parameter.grad is None:
                raise RuntimeError("DDP left a trainable parameter without a gradient")
            peer_gradients = [
                torch.empty_like(parameter.grad) for _ in range(context.world_size)
            ]
            torch.distributed.all_gather(peer_gradients, parameter.grad)
            if not all(
                torch.equal(peer_gradients[0], gradient)
                for gradient in peer_gradients[1:]
            ):
                raise RuntimeError("DDP gradients differ across ranks")

        context.barrier()
        if context.is_primary:
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "world_size": context.world_size,
                        "backend": context.backend,
                        "bf16": True,
                        "global_replay_sequences": local_n * context.world_size,
                        "parameters_updated": False,
                        "peak_memory_bytes": torch.cuda.max_memory_allocated(
                            context.device
                        ),
                    }
                )
            )
    finally:
        context.close()


@unittest.skipIf(torch is None, "requires the pinned PyTorch experiment environment")
class DistributedTrainingTests(unittest.TestCase):
    def test_fixed_global_sequence_counts_for_two_and_four_ranks(self) -> None:
        expected = {
            2: {16: 8, 128: 64},
            4: {16: 4, 128: 32},
        }

        for world_size, batches in expected.items():
            for global_sequences, local_sequences in batches.items():
                with self.subTest(
                    world_size=world_size, global_sequences=global_sequences
                ):
                    self.assertEqual(
                        local_sequence_count(global_sequences, world_size),
                        local_sequences,
                    )

    def test_sequence_scatter_preserves_global_draw_order(self) -> None:
        global_batch = torch.arange(3 * 16 * 2).reshape(3, 16, 2)

        for world_size in (2, 4):
            with self.subTest(world_size=world_size):
                rank_batches = split_sequence_tensor(global_batch, world_size)
                self.assertEqual(len(rank_batches), world_size)
                self.assertTrue(
                    torch.equal(torch.cat(rank_batches, dim=1), global_batch)
                )
                self.assertTrue(all(batch.is_contiguous() for batch in rank_batches))

    def test_invalid_world_sizes_and_indivisible_batches_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be one of"):
            local_sequence_count(16, 3)
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            local_sequence_count(15, 4)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            local_sequence_count(0, 2)
        with self.assertRaisesRegex(ValueError, r"\[T, N\]"):
            split_sequence_tensor(torch.zeros(4), 2)

    def test_rank_zero_draws_one_global_batch_before_scatter(self) -> None:
        class Replay:
            def __init__(self) -> None:
                self.call = None

            def minibatch_with_metadata(self, mb_t, mb_n, mb_device, task_id=None):
                self.call = (mb_t, mb_n, mb_device, task_id)
                tails = ((3,), (1, 2, 2), (1,), (1,), (1,))
                tensors = tuple(
                    torch.arange(mb_t * mb_n * torch.tensor(tail).prod().item())
                    .reshape(mb_t, mb_n, *tail)
                    .float()
                    for tail in tails
                )
                return (*tensors, 0, torch.arange(mb_n), torch.arange(mb_n))

        replay = Replay()
        context = SimpleNamespace(
            enabled=True,
            is_primary=True,
            world_size=2,
            device=torch.device("cpu"),
        )
        sampler = DistributedReplaySampler(
            context,
            replay,
            action_space=3,
            num_tasks=2,
            observation_shape=(1, 2, 2),
        )

        def copy_rank_zero(output, scatter_list, src):
            self.assertEqual(src, 0)
            self.assertEqual(len(scatter_list), 2)
            output.copy_(scatter_list[0])

        with mock.patch(
            "clworldmodel.distributed.dist.scatter", side_effect=copy_rank_zero
        ):
            sample = sampler.minibatch_with_metadata(3, 2, "cpu", task_id=1)

        self.assertEqual(replay.call, (3, 4, "cpu", 1))
        self.assertEqual(sample[0].shape, (3, 2, 3))
        self.assertEqual(sample[1].shape, (3, 2, 1, 2, 2))
        self.assertTrue(all(tensor.dtype == torch.float32 for tensor in sample[:5]))


if __name__ == "__main__":
    if "--distributed-smoke" in sys.argv:
        _run_cuda_distributed_smoke()
    else:
        unittest.main()
