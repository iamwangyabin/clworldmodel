"""Hand-computed tests for the versioned continual evaluation metrics."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clworldmodel.evaluation import (  # noqa: E402
    forward_transfer,
    median_iqr,
    normalize_return_matrix,
    sample_efficiency,
    single_pass_metrics,
    two_cycle_metrics,
)


class ContinualMetricTests(unittest.TestCase):
    def test_arrow_normalization_is_unclipped(self) -> None:
        normalized = normalize_return_matrix(
            [[10.0, 100.0], [25.0, -50.0]],
            random_returns=[10.0, 0.0],
            single_task_returns=[20.0, 100.0],
        )

        self.assertEqual(normalized[0], [0.0, 1.0])
        self.assertEqual(normalized[1], [1.5, -0.5])

    def test_single_pass_metrics_match_hand_computation(self) -> None:
        scores = [
            [1.0, 0.0, 0.0],
            [0.8, 0.5, 0.0],
            [0.6, 1.2, 0.0],
            [0.4, 1.1, 0.5],
            [0.5, 0.9, 1.5],
        ]

        result = single_pass_metrics(scores, task_end_rows=[0, 2, 4])

        self.assertAlmostEqual(result["forgetting"], 0.8 / 3.0)
        for observed, expected in zip(
            result["per_task_forgetting"], [0.5, 0.3, 0.0]
        ):
            self.assertAlmostEqual(observed, expected)
        self.assertAlmostEqual(result["acc"], 2.9 / 3.0)
        self.assertAlmostEqual(result["min_acc"], 0.65)
        self.assertAlmostEqual(result["wc_acc"], 1.5 / 3.0 + 2.0 / 3.0 * 0.65)

        first, second, third = result["boundaries"]
        self.assertIsNone(first["min_acc"])
        self.assertEqual(first["wc_acc"], 1.0)
        self.assertAlmostEqual(second["acc"], 0.9)
        self.assertAlmostEqual(second["min_acc"], 0.6)
        self.assertAlmostEqual(second["wc_acc"], 0.9)
        self.assertEqual(third["old_task_minima"], [0.4, 0.9])

    def test_forward_transfer_uses_aligned_curve_averages(self) -> None:
        result = forward_transfer(
            continual_task_curves=[[0.0, 1.0], [0.0, 2.0]],
            single_task_curves=[[0.0, 0.5], [0.0, 1.0]],
        )

        self.assertEqual(result["continual_areas"], [0.5, 1.0])
        self.assertEqual(result["single_task_areas"], [0.25, 0.5])
        self.assertEqual(result["per_task_forward_transfer"], [1.0, 1.0])
        self.assertEqual(result["forward_transfer"], 1.0)

    def test_two_cycle_metrics_match_hand_computation(self) -> None:
        result = two_cycle_metrics(
            first_exposure_end=[1.0, 2.0],
            before_second_exposure=[0.4, 1.0],
            second_exposure_end=[1.2, 1.5],
        )

        self.assertEqual(result["per_task_max_forgetting"], [0.6, 1.0])
        self.assertAlmostEqual(result["max_forgetting"], 0.8)
        self.assertEqual(result["per_task_recovery"], [1.2, 0.75])
        self.assertAlmostEqual(result["recovery"], 0.975)

    def test_sample_efficiency_uses_shared_global_maximum(self) -> None:
        result = sample_efficiency(
            {
                "method-a": [(0, 0.1), (10, 0.85), (20, 1.0)],
                "method-b": [(0, 0.2), (5, 0.9), (15, 2.0)],
            }
        )

        self.assertEqual(result["global_cross_method_maximum"], 2.0)
        self.assertEqual(result["threshold"], 1.7)
        self.assertIsNone(result["frames_to_threshold"]["method-a"])
        self.assertEqual(result["frames_to_threshold"]["method-b"], 15)

    def test_median_iqr_uses_linear_quantiles(self) -> None:
        self.assertEqual(
            median_iqr([5.0, 1.0, 4.0, 2.0, 3.0]),
            {"median": 3.0, "q25": 2.0, "q75": 4.0},
        )

    def test_invalid_inputs_fail_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical"):
            normalize_return_matrix([[1.0]], [1.0], [1.0])
        with self.assertRaisesRegex(ValueError, "rectangular"):
            single_pass_metrics([[1.0], [1.0, 2.0]], [0])
        with self.assertRaisesRegex(ValueError, "finite"):
            normalize_return_matrix([[math.nan]], [0.0], [1.0])
        with self.assertRaisesRegex(ValueError, "zero"):
            two_cycle_metrics([0.0], [0.0], [1.0])


if __name__ == "__main__":
    unittest.main()
