"""Focused tests for offline task-region and RBF-support diagnostics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clworldmodel.evaluation import analyze_task_regions, task_support_overlap


class LatentRegionMetricTests(unittest.TestCase):
    def test_separated_tasks_are_decodable_on_held_out_samples(self) -> None:
        rng = np.random.default_rng(7)
        task_arrays = {
            "game_a": rng.normal((-5.0, 0.0, 0.0), 0.4, size=(160, 3)),
            "game_b": rng.normal((0.0, 5.0, 0.0), 0.4, size=(160, 3)),
            "game_c": rng.normal((0.0, 0.0, 5.0), 0.4, size=(160, 3)),
        }
        result = analyze_task_regions(
            task_arrays,
            seed=11,
            max_samples_per_task=120,
            permutation_repetitions=200,
        )
        self.assertTrue(result["region_separation_detected"])
        self.assertGreater(result["nearest_centroid"]["accuracy"], 0.95)
        self.assertGreater(result["knn"]["accuracy"], 0.95)
        self.assertLessEqual(result["nearest_centroid"]["permutation_p_value"], 0.01)

    def test_identical_task_distributions_do_not_establish_separation(self) -> None:
        rng = np.random.default_rng(17)
        shared = rng.normal(size=(240, 8))
        result = analyze_task_regions(
            {"game_a": shared.copy(), "game_b": shared.copy()},
            seed=19,
            max_samples_per_task=200,
            permutation_repetitions=100,
        )
        self.assertFalse(result["region_separation_detected"])

    def test_weighted_support_overlap_distinguishes_local_regions(self) -> None:
        overlap = task_support_overlap(
            {
                "game_a": np.asarray([[1.0, 0.0, 0.0]]),
                "game_b": np.asarray([[0.0, 0.0, 1.0]]),
                "game_c": np.asarray([[0.9, 0.1, 0.0]]),
            }
        )
        matrix = np.asarray(overlap["weighted_jaccard_matrix"])
        self.assertEqual(matrix[0, 1], 0.0)
        self.assertGreater(matrix[0, 2], 0.8)


if __name__ == "__main__":
    unittest.main()
