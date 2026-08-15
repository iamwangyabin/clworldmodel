"""CPU-only checks for checkpoint-audit metric definitions."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from component_forgetting_audit import _event_candidate, _natural_candidates, _write_metrics_npz
from component_audit_metrics import (
    discounted_returns,
    linear_cka,
    mean_and_episode_bootstrap_ci,
    normalized_rmse,
    orthogonal_procrustes_residual,
    paired_episode_bootstrap_difference,
    symmetric_kl_from_log_probs,
)
from summarize_component_audit import MetricSpec, _paired_row
from component_swap_audit import _merged_state_dict
from input_fixed_module_forgetting_audit import METRICS, _assert_baseline_invariance


class ComponentAuditMetricTests(unittest.TestCase):
    def test_input_fixed_baseline_allows_only_svd_scale_residual(self) -> None:
        values = {metric.name: np.zeros(2, dtype=np.float64) for metric in METRICS}
        values["encoder.linear_cka"] = np.ones(2, dtype=np.float64)
        values["actor_head.top1_agreement"] = np.ones(2, dtype=np.float64)
        values["encoder.procrustes_residual"] = np.asarray([0.0, 5e-8])
        _assert_baseline_invariance(values)

        values["encoder.linear_cka"] = np.asarray([1.0, np.nan])
        values["encoder.procrustes_residual"] = np.asarray([0.0, np.nan])
        _assert_baseline_invariance(values)

        values["encoder.procrustes_residual"] = np.asarray([0.0, 1e-4])
        with self.assertRaises(RuntimeError):
            _assert_baseline_invariance(values)

    def test_linear_cka_is_one_for_identical_centered_features(self) -> None:
        features = np.asarray([[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]])
        self.assertAlmostEqual(linear_cka(features, features), 1.0, places=12)

    def test_procrustes_residual_removes_global_rotation(self) -> None:
        reference = np.asarray([[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]])
        rotation = np.asarray([[0.0, -1.0], [1.0, 0.0]])
        comparison = reference @ rotation * 3.0
        self.assertAlmostEqual(
            orthogonal_procrustes_residual(reference, comparison), 0.0, places=12
        )

    def test_normalized_rmse_is_zero_for_identical_values(self) -> None:
        values = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        self.assertAlmostEqual(normalized_rmse(values, values), 0.0, places=12)

    def test_symmetric_kl_is_zero_for_identical_policies(self) -> None:
        log_probs = np.log(np.asarray([[0.25, 0.75], [0.5, 0.5]]))
        values = symmetric_kl_from_log_probs(log_probs, log_probs)
        np.testing.assert_allclose(values, np.zeros(2), atol=1e-12)

    def test_discounted_returns_respect_terminal_continue(self) -> None:
        rewards = np.asarray([[[1.0], [2.0], [3.0]]])
        continues = np.asarray([[[1.0], [0.0], [1.0]]])
        expected = np.asarray([[[2.0], [2.0], [3.0]]])
        np.testing.assert_allclose(discounted_returns(rewards, continues, 0.5), expected)

    def test_episode_bootstrap_counts_clusters_not_timesteps(self) -> None:
        result = mean_and_episode_bootstrap_ci(
            np.asarray([1.0, 3.0, 10.0]),
            np.asarray([7, 7, 8]),
            seed=4,
            repetitions=100,
        )
        self.assertEqual(result["n_chunks"], 3)
        self.assertEqual(result["n_episodes"], 2)
        self.assertAlmostEqual(result["mean"], 14.0 / 3.0)

    def test_paired_episode_bootstrap_retains_difference_direction(self) -> None:
        result = paired_episode_bootstrap_difference(
            np.asarray([1.0, 3.0, 10.0]),
            np.asarray([2.0, 4.0, 5.0]),
            np.asarray([7, 7, 8]),
            seed=9,
            repetitions=100,
        )
        self.assertAlmostEqual(result["baseline_mean"], 14.0 / 3.0)
        self.assertAlmostEqual(result["comparison_mean"], 11.0 / 3.0)
        self.assertAlmostEqual(result["comparison_minus_baseline"], -1.0)
        self.assertEqual(result["n_chunks"], 3)
        self.assertEqual(result["n_episodes"], 2)

    def test_higher_is_better_summary_preserves_raw_delta_ci(self) -> None:
        row = _paired_row(
            MetricSpec("actor.top1_agreement", "actor agreement", "higher_is_better"),
            task_index=0,
            baseline_label="C1_e89",
            comparison_label="C6_e539",
            episode_ids=np.asarray([4, 5]),
            raw_metrics={
                (0, "natural", "C1_e89", "actor.top1_agreement"): np.asarray([1.0, 1.0]),
                (0, "natural", "C6_e539", "actor.top1_agreement"): np.asarray([0.0, 0.0]),
            },
            summary_rows={},
            bootstrap_seed=3,
        )
        self.assertAlmostEqual(row["comparison_minus_baseline"], -1.0)
        self.assertAlmostEqual(row["comparison_minus_baseline_ci_low"], -1.0)
        self.assertAlmostEqual(row["comparison_minus_baseline_ci_high"], -1.0)
        self.assertAlmostEqual(row["degradation_score"], 1.0)

    def test_component_swap_replaces_only_requested_parameter_group(self) -> None:
        restored = _merged_state_dict(
            {"rssm.transition.weight": 1, "decoder.weight": 2, "reward_fc.weight": 3},
            {"rssm.transition.weight": 10, "decoder.weight": 20, "reward_fc.weight": 30},
            ("rssm.",),
        )
        self.assertEqual(restored["rssm.transition.weight"], 10)
        self.assertEqual(restored["decoder.weight"], 2)
        self.assertEqual(restored["reward_fc.weight"], 3)

    def test_raw_metric_bundle_allows_different_dataset_sizes(self) -> None:
        with TemporaryDirectory() as temporary:
            digest = _write_metrics_npz(
                Path(temporary) / "metrics.npz",
                [{"dataset_role": "natural"}, {"dataset_role": "event"}],
                [np.asarray([1.0, 2.0]), np.asarray([3.0])],
            )
            self.assertEqual(len(digest), 64)
            with np.load(Path(temporary) / "metrics.npz", allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["offsets"], np.asarray([0, 2, 3]))
                np.testing.assert_array_equal(archive["values"], np.asarray([1.0, 2.0, 3.0]))

    def test_capped_nonterminal_segment_is_natural_only(self) -> None:
        length = 128
        episode = {
            "actions": np.zeros((length, 2), dtype=np.uint8),
            "observations": np.zeros((length, 3, 64, 64), dtype=np.uint8),
            "raw_rewards": np.zeros((length, 1), dtype=np.float32),
            "scaled_rewards": np.zeros((length, 1), dtype=np.float32),
            "continues": np.ones((length, 1), dtype=np.uint8),
            "resets": np.zeros((length, 1), dtype=np.uint8),
            "terminated": np.zeros((length, 1), dtype=np.uint8),
            "truncated": np.zeros((length, 1), dtype=np.uint8),
        }
        episode["resets"][0, 0] = 1
        candidates = _natural_candidates(
            episode,
            chunk_length=64,
            episode_id=3,
            rng=np.random.default_rng(2),
        )
        self.assertGreaterEqual(len(candidates), 1)
        self.assertIsNone(_event_candidate(episode, chunk_length=64, episode_id=3))


if __name__ == "__main__":
    unittest.main()
