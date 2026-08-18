"""Parity contracts for ARROW replay metadata used by native R2-Dreamer."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VENDORED_ATARI = ROOT / "third_party" / "arrow" / "Code" / "ARROW_and_DV3" / "Atari"

try:
    import torch
    import sortedcontainers  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - exercised in the GPU environment
    torch = None

if torch is not None:
    sys.path.insert(0, str(VENDORED_ATARI))
    from replay import FifoReplay


@unittest.skipIf(torch is None, "requires the pinned PyTorch experiment environment")
class ArrowReplayMetadataParityTests(unittest.TestCase):
    @staticmethod
    def _trajectory(offset: float) -> tuple[torch.Tensor, ...]:
        actions = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3) + offset
        observations = torch.arange(4 * 2 * 3 * 64 * 64, dtype=torch.float32).reshape(4, 2, 3, 64, 64)
        rewards = torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2, 1) + offset
        continues = torch.ones(4, 2, 1)
        resets = torch.zeros(4, 2, 1)
        return actions, observations, rewards, continues, resets

    def test_metadata_path_preserves_fifo_minibatch_values(self) -> None:
        replay = FifoReplay(t=4, n=3, n_acts=3, store_device="cpu")
        first = self._trajectory(0.0)
        second = self._trajectory(100.0)
        self.assertEqual(replay.add(*first), [0, 1])
        self.assertEqual(replay.add(*second), [2, 0])

        np.random.seed(31)
        plain = replay.minibatch(3, 2, "cpu")
        np.random.seed(31)
        metadata = replay.minibatch_with_metadata(3, 2, "cpu")

        self.assertEqual(len(plain), 5)
        for expected, observed in zip(plain, metadata[:5]):
            torch.testing.assert_close(expected, observed, rtol=0, atol=0)
        starts, sequences = metadata[-2:]
        self.assertEqual(starts.shape, (2,))
        self.assertEqual(sequences.shape, (2,))
        self.assertTrue(np.all(starts >= 0))
        self.assertTrue(np.all(starts <= 1))
        self.assertTrue(np.all(sequences >= 0))
        self.assertTrue(np.all(sequences < replay.n_valid))


if __name__ == "__main__":
    unittest.main()
