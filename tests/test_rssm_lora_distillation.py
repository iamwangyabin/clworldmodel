"""Focused contracts for the posthoc trained RSSM LoRA probe."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - minimal host environments omit torch.
    torch = None
    nn = None

if torch is not None:
    from train_cnn_fullbank_rssm_lora_distill import (
        TrajectoryCohort,
        _cpu_state_dict,
        _parameterize_affines,
    )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class RssmLoraDistillationTests(unittest.TestCase):
    def test_zero_effect_lora_preserves_linear_and_gru_outputs(self) -> None:
        class Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(5, 4)
                self.norm = nn.LayerNorm(4)
                self.gru = nn.GRUCell(4, 3)

            def forward(self, x, h):
                x = self.norm(self.linear(x))
                return self.gru(x, h)

        torch.manual_seed(4)
        block = Block()
        x = torch.randn(7, 5)
        h = torch.randn(7, 3)
        expected = block(x, h).detach()
        block.requires_grad_(False)

        report = _parameterize_affines(block, rank=2)
        actual = block(x, h)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        trainable = [parameter for parameter in block.parameters() if parameter.requires_grad]
        self.assertTrue(trainable)
        self.assertEqual(
            sum(parameter.numel() for parameter in trainable),
            report["trainable_parameters"],
        )
        actual.square().mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in trainable))
        originals = [
            parameter
            for name, parameter in block.named_parameters()
            if name.endswith(".original")
        ]
        self.assertTrue(originals)
        self.assertTrue(all(not parameter.requires_grad for parameter in originals))

    def test_worker_major_sampling_never_crosses_workers(self) -> None:
        workers, frames = 2, 6
        observations = torch.zeros(workers, frames, 3, 1, 1, dtype=torch.uint8)
        observations[0, :, 0, 0, 0] = torch.arange(frames)
        observations[1, :, 0, 0, 0] = 100 + torch.arange(frames)
        cohort = TrajectoryCohort(
            observations=observations,
            action_indices=torch.zeros(workers, frames, dtype=torch.uint8),
            resets=torch.zeros(workers, frames, dtype=torch.bool),
        )

        _, sampled, _ = cohort.sample(
            sequence_length=4,
            batch_size=32,
            action_space=18,
            generator=torch.Generator().manual_seed(8),
            device=torch.device("cpu"),
        )

        values = (sampled[:, :, 0, 0, 0] * 255).round().to(torch.int64)
        self.assertTrue(torch.equal(values[1:] - values[:-1], torch.ones_like(values[1:])))
        for batch in range(values.shape[1]):
            self.assertTrue(bool((values[:, batch] < 10).all() or (values[:, batch] >= 100).all()))

    def test_cpu_state_dict_is_detached_from_trainable_actor(self) -> None:
        actor = nn.Linear(3, 2)
        saved = _cpu_state_dict(actor)
        with torch.no_grad():
            actor.weight.add_(1)

        self.assertEqual(saved["weight"].device.type, "cpu")
        self.assertFalse(torch.equal(saved["weight"], actor.weight.cpu()))


if __name__ == "__main__":
    unittest.main()
