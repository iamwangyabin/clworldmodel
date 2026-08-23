"""Value and retention parity for the opt-in uint8 observation replay."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = ROOT / "src"
VENDORED_ATARI = ROOT / "third_party" / "arrow" / "Code" / "ARROW_and_DV3" / "Atari"

try:
    import torch
    import sortedcontainers  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - exercised in experiment envs
    torch = None

if torch is not None:
    sys.path.insert(0, str(PROJECT_SRC))
    sys.path.insert(0, str(VENDORED_ATARI))
    from replay import FifoReplay, LongTermReplay


@unittest.skipIf(torch is None, "requires PyTorch")
class Uint8ReplayTests(unittest.TestCase):
    @staticmethod
    def _batch(
        time: int, sequences: int, *, offset: int
    ) -> tuple[torch.Tensor, ...]:
        action_indices = (
            torch.arange(time * sequences).reshape(time, sequences) + offset
        ).remainder(4)
        actions = torch.nn.functional.one_hot(action_indices, 4).float()
        count = time * sequences * 3 * 64 * 64
        pixels = (
            torch.arange(count, dtype=torch.int64) + offset
        ).remainder(256).to(torch.uint8)
        observations = pixels.reshape(time, sequences, 3, 64, 64).float().div(255)
        rewards = torch.arange(time * sequences, dtype=torch.float32).reshape(
            time, sequences, 1
        )
        continues = torch.ones(time, sequences, 1)
        resets = torch.zeros(time, sequences, 1)
        return actions, observations, rewards, continues, resets

    def assert_samples_equal(self, left: tuple, right: tuple) -> None:
        for left_value, right_value in zip(left[:5], right[:5]):
            torch.testing.assert_close(left_value, right_value, rtol=0, atol=0)
        np.testing.assert_array_equal(left[-2], right[-2])
        np.testing.assert_array_equal(left[-1], right[-1])

    def test_fifo_uint8_round_trip_matches_float32_through_wraparound(self) -> None:
        float_replay = FifoReplay(3, 3, 4, "cpu")
        uint8_replay = FifoReplay(
            3, 3, 4, "cpu", observation_dtype="uint8"
        )

        for offset in (0, 73):
            batch = self._batch(3, 2, offset=offset)
            self.assertEqual(float_replay.add(*batch), uint8_replay.add(*batch))

        self.assertEqual(float_replay.n_idx, uint8_replay.n_idx)
        self.assertEqual(float_replay.n_valid, uint8_replay.n_valid)
        self.assertEqual(float_replay.obss.dtype, torch.float32)
        self.assertEqual(uint8_replay.obss.dtype, torch.uint8)

        np.random.seed(19)
        float_sample = float_replay.minibatch_with_metadata(2, 8, "cpu")
        np.random.seed(19)
        uint8_sample = uint8_replay.minibatch_with_metadata(2, 8, "cpu")
        self.assertEqual(uint8_sample[1].dtype, torch.float32)
        self.assert_samples_equal(float_sample, uint8_sample)

    def test_ltdm_uint8_preserves_random_key_retention_and_samples(self) -> None:
        float_replay = LongTermReplay(2, 3, 4, "cpu")
        uint8_replay = LongTermReplay(
            2, 3, 4, "cpu", observation_dtype="uint8"
        )
        batch = self._batch(2, 5, offset=41)

        np.random.seed(23)
        float_slots = float_replay.add(*batch)
        np.random.seed(23)
        uint8_slots = uint8_replay.add(*batch)

        self.assertEqual(float_slots, uint8_slots)
        self.assertEqual(list(float_replay.collection), list(uint8_replay.collection))
        self.assertEqual(float_replay.n_valid, uint8_replay.n_valid)

        np.random.seed(29)
        float_sample = float_replay.minibatch_with_metadata(2, 7, "cpu")
        np.random.seed(29)
        uint8_sample = uint8_replay.minibatch_with_metadata(2, 7, "cpu")
        self.assert_samples_equal(float_sample, uint8_sample)

    def test_uint8_mmap_uses_one_byte_per_pixel_and_decodes_on_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observations.uint8.mmap"
            replay = FifoReplay(
                2,
                2,
                4,
                "cpu",
                observation_storage_path=path,
                observation_dtype="uint8",
            )
            batch = self._batch(2, 2, offset=11)
            replay.add(*batch)

            self.assertEqual(path.stat().st_size, 2 * 2 * 3 * 64 * 64)
            self.assertEqual(replay.obss.dtype, torch.uint8)
            np.random.seed(31)
            sample = replay.minibatch_with_metadata(2, 2, "cpu")
            np.random.seed(31)
            reference = FifoReplay(2, 2, 4, "cpu")
            reference.add(*batch)
            reference_sample = reference.minibatch_with_metadata(2, 2, "cpu")
            self.assert_samples_equal(reference_sample, sample)

    def test_uint8_replay_rejects_non_pixel_float_values(self) -> None:
        replay = FifoReplay(1, 1, 4, "cpu", observation_dtype="uint8")
        batch = list(self._batch(1, 1, offset=0))
        batch[1] = torch.full_like(batch[1], 1.01)
        with self.assertRaisesRegex(ValueError, r"must lie in \[0, 1\]"):
            replay.add(*batch)


if __name__ == "__main__":
    unittest.main()
