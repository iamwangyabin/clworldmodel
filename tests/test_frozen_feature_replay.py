"""Parity checks for cached and on-the-fly frozen replay features."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = ROOT / "src"
VENDORED_ATARI = ROOT / "third_party" / "arrow" / "Code" / "ARROW_and_DV3" / "Atari"

try:
    import torch
    import torch.nn as nn
    import sortedcontainers  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - exercised in the GPU environment
    torch = None
    nn = None

if torch is not None:
    sys.path.insert(0, str(PROJECT_SRC))
    sys.path.insert(0, str(VENDORED_ATARI))
    from clworldmodel.replay import (
        ArrowFrozenFeatureCache,
        ArrowOnTheFlyFeatureSource,
    )
    from replay import FifoReplay, LongTermReplay, MultiTypeReplay

    class _Encoder(nn.Module):
        output_size = 4

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            flat = images.flatten(1).float()
            return torch.stack(
                (
                    flat.mean(1),
                    flat[:, 0] * 0.12345,
                    flat[:, 1] * 0.33333,
                    flat[:, -1] * 0.98765,
                ),
                dim=-1,
            )


@unittest.skipIf(torch is None, "requires PyTorch")
class FrozenFeatureReplayTests(unittest.TestCase):
    @staticmethod
    def _replay_batch() -> tuple[MultiTypeReplay, tuple[list[int], ...], torch.Tensor]:
        replay = MultiTypeReplay(
            FifoReplay(4, 2, 3, "cpu"),
            LongTermReplay(4, 2, 3, "cpu"),
            sampling_weights=(0.5, 0.5),
        )
        actions = torch.zeros(4, 2, 3)
        observations = torch.arange(
            4 * 2 * 3 * 64 * 64, dtype=torch.float32
        ).view(4, 2, 3, 64, 64)
        rewards = torch.zeros(4, 2, 1)
        continues = torch.ones(4, 2, 1)
        resets = torch.zeros(4, 2, 1)
        write_slots = replay.add(
            actions, observations, rewards, continues, resets
        )
        return replay, write_slots, observations

    def test_on_the_fly_features_match_float16_sidecar_samples(self) -> None:
        replay, write_slots, observations = self._replay_batch()
        encoder = _Encoder()
        cached = ArrowFrozenFeatureCache(
            replay, encoder.output_size, dtype=torch.float16
        )
        encoded = encoder(observations.flatten(0, 1)).view(4, 2, -1)
        cached.record(write_slots, encoded)
        on_the_fly = ArrowOnTheFlyFeatureSource(
            replay,
            encoder,
            encoder.output_size,
            dtype=torch.float16,
        )

        random.seed(7)
        np.random.seed(11)
        cached_sample = cached.minibatch(3, 2, mb_device="cpu")
        random.seed(7)
        np.random.seed(11)
        recomputed_sample = on_the_fly.minibatch(3, 2, mb_device="cpu")

        for cached_value, recomputed_value in zip(
            cached_sample, recomputed_sample
        ):
            torch.testing.assert_close(
                cached_value, recomputed_value, rtol=0, atol=0
            )
        self.assertFalse(on_the_fly.requires_recording)
        self.assertEqual(on_the_fly.storage_bytes, 0)
        self.assertEqual(
            on_the_fly.storage_accounting()["storage_backend"], "on_the_fly"
        )

    def test_bfloat16_on_the_fly_features_avoid_the_float32_round_trip(self) -> None:
        replay, _, _ = self._replay_batch()
        source = ArrowOnTheFlyFeatureSource(
            replay,
            _Encoder(),
            _Encoder.output_size,
            dtype=torch.bfloat16,
            consumer_dtype=torch.bfloat16,
        )

        sample = source.minibatch(3, 2, mb_device="cpu")
        features = sample[2]
        accounting = source.storage_accounting()

        self.assertEqual(features.dtype, torch.bfloat16)
        self.assertEqual(accounting["quantization_dtype"], "bfloat16")
        self.assertEqual(accounting["consumer_dtype"], "bfloat16")
        self.assertEqual(
            accounting["quantization_semantics"],
            "encoder output is retained without a dtype round trip",
        )


if __name__ == "__main__":
    unittest.main()
