"""CPU contracts for the opt-in BF16 execution profile."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock


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
    from clworldmodel.precision import (
        autocast_context,
        require_cuda_compute_support,
        validate_compute_dtype,
    )
    from ac import replay_lambda_returns, rew_symlog_to_2hot
    from rssm import straight_through_one_hot
    from wm import WorldModel, categorical_kl, symexp, symlog


@unittest.skipIf(torch is None, "requires PyTorch")
class MixedPrecisionTests(unittest.TestCase):
    def test_sensitive_probability_and_return_math_promotes_bfloat16_to_float32(
        self,
    ) -> None:
        logits = torch.tensor([[[1.0, 0.0, -1.0]]], dtype=torch.bfloat16)
        log_probs, sample = straight_through_one_hot(logits, stochastic=False)
        kl = categorical_kl(logits, logits + torch.tensor(0.5, dtype=torch.bfloat16))
        transformed = symlog(torch.tensor([[-2.0], [3.0]], dtype=torch.bfloat16))
        restored = symexp(transformed.to(torch.bfloat16))
        two_hot = rew_symlog_to_2hot(
            torch.tensor([[[-1.25]], [[2.5]]], dtype=torch.bfloat16)
        )
        returns = replay_lambda_returns(
            torch.ones(3, 2, 1, dtype=torch.bfloat16),
            torch.ones(3, 2, 1, dtype=torch.bfloat16),
            torch.ones(3, 2, 1, dtype=torch.bfloat16),
            discount=0.997,
            lam=0.95,
        )

        for value in (log_probs, sample, kl, transformed, restored, two_hot, returns):
            self.assertEqual(value.dtype, torch.float32)
            self.assertTrue(torch.isfinite(value).all())
        torch.testing.assert_close(sample.sum(-1), torch.ones_like(sample.sum(-1)))
        torch.testing.assert_close(two_hot.sum(-1), torch.ones_like(two_hot.sum(-1)))

    def test_cpu_context_is_noop_and_model_parameters_stay_float32(self) -> None:
        class Embedder(nn.Module):
            output_size = 4

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return images.flatten(1)[:, : self.output_size]

        with autocast_context("cpu", "bfloat16"):
            result = torch.ones(2, 2) @ torch.ones(2, 2)
        model = WorldModel(
            3,
            (2, 3),
            4,
            5,
            cnn_depth=2,
            mlp_features=8,
            mlp_layers=2,
            image_embedder=Embedder(),
            compute_dtype="bfloat16",
        )

        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(model.compute_dtype, "bfloat16")
        self.assertTrue(
            all(parameter.dtype == torch.float32 for parameter in model.parameters())
        )
        with self.assertRaisesRegex(ValueError, "Unknown compute dtype"):
            validate_compute_dtype("float16")

    def test_continuation_probability_and_loss_stay_full_precision(self) -> None:
        class Embedder(nn.Module):
            output_size = 4

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return images.flatten(1)[:, : self.output_size]

        model = WorldModel(
            3,
            (2, 3),
            4,
            5,
            cnn_depth=2,
            mlp_features=8,
            mlp_layers=2,
            image_embedder=Embedder(),
            compute_dtype="bfloat16",
        )
        linear_layers = [
            module for module in model.continue_fc.modules()
            if isinstance(module, nn.Linear)
        ]
        with torch.no_grad():
            for layer in linear_layers:
                layer.weight.zero_()
                layer.bias.zero_()
            linear_layers[-1].bias.fill_(10.0)

        model_state = torch.zeros(2, 1, model.zh_transform.out_features)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits = model.predict_continue_logits(model_state)
            probabilities = model.predict_continue(model_state)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits.float(), torch.zeros_like(logits, dtype=torch.float32)
        )

        self.assertEqual(probabilities.dtype, torch.float32)
        self.assertTrue((probabilities < 1.0).all())
        self.assertGreater(loss.item(), 9.0)
        self.assertLess(loss.item(), 11.0)
        loss.backward()
        self.assertTrue(torch.isfinite(linear_layers[-1].bias.grad).all())
        self.assertGreater(linear_layers[-1].bias.grad.item(), 0.9)

    @unittest.skipUnless(
        torch is not None
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported(),
        "requires a CUDA device with BF16 support",
    )
    def test_cuda_bfloat16_continuation_does_not_saturate(self) -> None:
        class Embedder(nn.Module):
            output_size = 4

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return images.flatten(1)[:, : self.output_size]

        model = WorldModel(
            3,
            (2, 3),
            4,
            5,
            cnn_depth=2,
            mlp_features=8,
            mlp_layers=2,
            image_embedder=Embedder(),
            compute_dtype="bfloat16",
        ).cuda()
        linear_layers = [
            module for module in model.continue_fc.modules()
            if isinstance(module, nn.Linear)
        ]
        with torch.no_grad():
            for layer in linear_layers:
                layer.weight.zero_()
                layer.bias.zero_()
            linear_layers[-1].bias.fill_(10.0)

        model_state = torch.zeros(
            2, 1, model.zh_transform.out_features, device="cuda"
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model.predict_continue_logits(model_state)
            probabilities = model.predict_continue(model_state)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits.float(), torch.zeros_like(logits, dtype=torch.float32)
        )

        self.assertEqual(logits.dtype, torch.bfloat16)
        self.assertEqual(probabilities.dtype, torch.float32)
        self.assertTrue((probabilities < 1.0).all())
        self.assertGreater(loss.item(), 9.0)
        self.assertLess(loss.item(), 11.0)
        loss.backward()
        self.assertTrue(torch.isfinite(linear_layers[-1].bias.grad).all())
        self.assertGreater(linear_layers[-1].bias.grad.item(), 0.9)

    def test_bfloat16_profile_fails_before_allocation_without_cuda_support(self) -> None:
        require_cuda_compute_support("float32")
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "available CUDA"):
                require_cuda_compute_support("bfloat16")
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "is_bf16_supported", return_value=False),
        ):
            with self.assertRaisesRegex(RuntimeError, "does not support BF16"):
                require_cuda_compute_support("bfloat16")


if __name__ == "__main__":
    unittest.main()
