"""Storage and sampling parity for opt-in CoinRun uint8 replay."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VENDORED_COINRUN = (
    ROOT / "third_party" / "arrow" / "Code" / "ARROW_and_DV3" / "CoinRun"
)

try:
    import torch
    import sortedcontainers  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - exercised in experiment envs
    torch = None


def _load_coinrun_replay():
    """Load CoinRun replay without colliding with Atari's top-level modules."""

    module_names = ("replay", "rssm", "wm", "vae")
    saved_modules = {name: sys.modules.pop(name, None) for name in module_names}
    sys.path.insert(0, str(VENDORED_COINRUN))
    try:
        spec = importlib.util.spec_from_file_location(
            "coinrun_uint8_replay_under_test", VENDORED_COINRUN / "replay.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load vendored CoinRun replay")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(VENDORED_COINRUN))
        for name in module_names:
            sys.modules.pop(name, None)
            if saved_modules[name] is not None:
                sys.modules[name] = saved_modules[name]


if torch is not None:
    COINRUN_REPLAY = _load_coinrun_replay()
    FifoReplay = COINRUN_REPLAY.FifoReplay
    LongTermReplay = COINRUN_REPLAY.LongTermReplay
    MultiTypeReplay = COINRUN_REPLAY.MultiTypeReplay


@unittest.skipIf(torch is None, "requires PyTorch")
class CoinRunUint8ReplayTests(unittest.TestCase):
    @staticmethod
    def _batch(
        time: int, sequences: int, *, offset: int
    ) -> tuple[torch.Tensor, ...]:
        action_indices = (
            torch.arange(time * sequences).reshape(time, sequences) + offset
        ).remainder(15)
        actions = torch.nn.functional.one_hot(action_indices, 15).float()
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
        for left_value, right_value in zip(left, right):
            torch.testing.assert_close(left_value, right_value, rtol=0, atol=0)

    def test_default_float32_fifo_behavior_is_unchanged(self) -> None:
        replay = FifoReplay(2, 3, 15, "cpu")
        batch = self._batch(2, 2, offset=7)
        replay.add(*batch)

        self.assertEqual(replay.obss.dtype, torch.float32)
        np.random.seed(13)
        sample = replay.minibatch(2, 4, "cpu")
        self.assertEqual(sample[1].dtype, torch.float32)

    def test_fifo_uint8_round_trip_matches_float32_after_wraparound(self) -> None:
        float_replay = FifoReplay(3, 3, 15, "cpu")
        uint8_replay = FifoReplay(
            3, 3, 15, "cpu", observation_dtype="uint8"
        )

        for offset in (0, 73):
            batch = self._batch(3, 2, offset=offset)
            float_replay.add(*batch)
            uint8_replay.add(*batch)

        self.assertEqual(float_replay.n_idx, uint8_replay.n_idx)
        self.assertEqual(float_replay.n_valid, uint8_replay.n_valid)
        self.assertEqual(uint8_replay.obss.dtype, torch.uint8)
        self.assertEqual(
            uint8_replay.obss.numel() * uint8_replay.obss.element_size(),
            3 * 3 * 3 * 64 * 64,
        )

        np.random.seed(19)
        float_sample = float_replay.minibatch(2, 8, "cpu")
        np.random.seed(19)
        uint8_sample = uint8_replay.minibatch(2, 8, "cpu")
        self.assert_samples_equal(float_sample, uint8_sample)

    def test_ltdm_uint8_preserves_retention_and_sample_values(self) -> None:
        float_replay = LongTermReplay(2, 3, 15, "cpu")
        uint8_replay = LongTermReplay(
            2, 3, 15, "cpu", observation_dtype="uint8"
        )
        batch = self._batch(2, 5, offset=41)

        np.random.seed(23)
        float_replay.add(*batch)
        np.random.seed(23)
        uint8_replay.add(*batch)

        self.assertEqual(list(float_replay.collection), list(uint8_replay.collection))
        self.assertEqual(float_replay.n_valid, uint8_replay.n_valid)

        np.random.seed(29)
        float_sample = float_replay.minibatch(2, 7, "cpu")
        np.random.seed(29)
        uint8_sample = uint8_replay.minibatch(2, 7, "cpu")
        self.assert_samples_equal(float_sample, uint8_sample)

    def test_mixed_uint8_replay_encodes_once_and_preserves_subbuffers(self) -> None:
        fifo = FifoReplay(2, 2, 15, "cpu", observation_dtype="uint8")
        ltdm = LongTermReplay(2, 2, 15, "cpu", observation_dtype="uint8")
        replay = MultiTypeReplay(fifo, ltdm, sampling_weights=(0.5, 0.5))
        batch = self._batch(2, 2, offset=17)

        np.random.seed(37)
        replay.add(*batch)

        expected = batch[1].mul(255).round().to(torch.uint8)
        torch.testing.assert_close(fifo.obss, expected, rtol=0, atol=0)
        torch.testing.assert_close(ltdm.obss, expected, rtol=0, atol=0)
        self.assertEqual(replay.sampling_weights, (0.5, 0.5))

    def test_uint8_replay_rejects_out_of_range_pixels(self) -> None:
        replay = FifoReplay(1, 1, 15, "cpu", observation_dtype="uint8")
        batch = list(self._batch(1, 1, offset=0))
        batch[1] = torch.full_like(batch[1], 1.01)
        with self.assertRaisesRegex(ValueError, r"must lie in \[0, 1\]"):
            replay.add(*batch)


if __name__ == "__main__":
    unittest.main()
