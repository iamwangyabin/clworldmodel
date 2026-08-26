"""Focused checks for the oracle low-rank route diagnostic."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import torch
except ImportError:  # pragma: no cover - exercised only in minimal host envs.
    torch = None

from evaluate_cnn_fullbank_oracle_lora import (
    _install_oracle_state_delta,
    _truncated_delta,
)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class OracleLoraEvaluationTests(unittest.TestCase):
    def test_rank_one_projection_is_best_svd_delta(self) -> None:
        base = torch.zeros(2, 2)
        target = torch.diag(torch.tensor([3.0, 1.0]))

        reconstructed, captured = _truncated_delta(torch, base, target, rank=1)

        torch.testing.assert_close(
            reconstructed, torch.diag(torch.tensor([3.0, 0.0]))
        )
        self.assertAlmostEqual(captured, 0.9)

    def test_small_vectors_are_preserved_exactly(self) -> None:
        base = {
            "weight": torch.zeros(2, 2),
            "bias": torch.zeros(2),
        }
        target = {
            "weight": torch.diag(torch.tensor([3.0, 1.0])),
            "bias": torch.tensor([2.0, -1.0]),
        }

        report = _install_oracle_state_delta(torch, base, target, rank=1)

        torch.testing.assert_close(target["bias"], torch.tensor([2.0, -1.0]))
        torch.testing.assert_close(
            target["weight"], torch.diag(torch.tensor([3.0, 0.0]))
        )
        self.assertAlmostEqual(
            report["delta_energy_capture_including_exact_vectors"], 14.0 / 15.0
        )


if __name__ == "__main__":
    unittest.main()
