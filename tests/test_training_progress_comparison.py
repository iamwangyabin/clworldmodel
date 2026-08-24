"""Tests for the predeclared early ARROW progress diagnostic."""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compare_arrow_training_progress import compare_progress  # noqa: E402


class TrainingProgressComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference = json.loads(
            (
                ROOT
                / "tests"
                / "fixtures"
                / "arrow_ar50_original_s0_early_metrics.json"
            ).read_text(encoding="utf-8")
        )

    def _scaled_current(self, scale: float, points: int = 5) -> dict:
        current = {}
        for tag, reference_points in self.reference["metrics"].items():
            selected = [point for point in reference_points if point["step"] > 0]
            current[tag] = [
                {"step": point["step"], "value": point["value"] * scale}
                for point in selected[:points]
            ]
        return current

    def test_comparable_finite_progress_passes(self) -> None:
        result = compare_progress(self._scaled_current(2.0), self.reference)

        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["failures"])
        self.assertAlmostEqual(
            result["metrics"]["Loss/recon"]["median_absolute_ratio"], 2.0
        )

    def test_order_of_magnitude_continuation_divergence_fails(self) -> None:
        current = self._scaled_current(1.0)
        current["Loss/cont"] = [
            {"step": point["step"], "value": 0.25}
            for point in self.reference["metrics"]["Loss/cont"]
            if point["step"] in {1000, 2000, 3000, 4000, 5000}
        ]

        result = compare_progress(current, self.reference)

        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["metrics"]["Loss/cont"]["status"], "fail")
        self.assertTrue(any("Loss/cont" in message for message in result["failures"]))

    def test_non_finite_required_metric_fails(self) -> None:
        current = self._scaled_current(1.0)
        current["Loss/kl"][2]["value"] = math.nan

        result = compare_progress(current, self.reference)

        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["metrics"]["Loss/kl"]["non_finite_steps"], [3000])

    def test_guard_waits_for_three_aligned_points(self) -> None:
        result = compare_progress(
            self._scaled_current(1.0, points=2), self.reference
        )

        self.assertEqual(result["status"], "insufficient_data")
        self.assertTrue(result["insufficient_data"])


if __name__ == "__main__":
    unittest.main()
