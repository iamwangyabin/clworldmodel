"""Focused contracts for the shared DINO patch convolution adapter."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = ROOT / "src"

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - exercised in the GPU environment
    torch = None
    nn = None

if torch is not None:
    sys.path.insert(0, str(PROJECT_SRC))
    from clworldmodel.models import ChannelLayerNorm, DinoPatchConvAdapter


@unittest.skipIf(torch is None, "requires PyTorch")
class DinoV3AdapterTests(unittest.TestCase):
    def test_exact_single_convolution_contract(self) -> None:
        adapter = DinoPatchConvAdapter()
        layers = list(adapter.adapter)

        self.assertIsInstance(layers[0], nn.Conv2d)
        self.assertEqual(layers[0].in_channels, 384)
        self.assertEqual(layers[0].out_channels, 64)
        self.assertEqual(layers[0].kernel_size, (3, 3))
        self.assertEqual(layers[0].stride, (2, 2))
        self.assertEqual(layers[0].padding, (1, 1))
        self.assertEqual(layers[0].groups, 1)
        self.assertIsInstance(layers[1], ChannelLayerNorm)
        self.assertEqual(layers[1].norm.normalized_shape, (64,))
        self.assertEqual(layers[1].norm.eps, 1e-3)
        self.assertIsInstance(layers[2], nn.SiLU)
        self.assertIsInstance(layers[3], nn.Flatten)
        self.assertEqual(adapter.input_size, 98_304)
        self.assertEqual(adapter.output_grid_size, 8)
        self.assertEqual(adapter.output_size, 4_096)
        self.assertEqual(
            sum(parameter.numel() for parameter in adapter.parameters()),
            221_376,
        )

    def test_forward_preserves_batch_and_trains_adapter_only(self) -> None:
        torch.manual_seed(17)
        adapter = DinoPatchConvAdapter()
        patches = torch.randn(3, 98_304, requires_grad=True)

        output = adapter(patches.detach())
        self.assertEqual(output.shape, (3, 4_096))
        output.square().mean().backward()

        self.assertIsNone(patches.grad)
        self.assertGreater(adapter.adapter[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(adapter.adapter[1].norm.weight.grad.abs().sum().item(), 0)

    def test_rejects_non_flattened_or_wrong_width_features(self) -> None:
        adapter = DinoPatchConvAdapter()
        with self.assertRaisesRegex(ValueError, "flattened_patches"):
            adapter(torch.zeros(1, 16, 16, 384))
        with self.assertRaisesRegex(ValueError, "expected 98304"):
            adapter(torch.zeros(1, 4_096))


if __name__ == "__main__":
    unittest.main()
