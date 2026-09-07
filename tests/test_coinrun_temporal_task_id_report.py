# SPDX-License-Identifier: Apache-2.0
import unittest
import numpy as np
from report_coinrun_temporal_task_id import capped_correct, cluster_delta


class TaskIDReportTests(unittest.TestCase):
    def test_padding_never_changes_ended_episode_classification(self):
        pred = np.array([[[0, 1, 0], [1, 1, 0]]])
        labels = np.array([1, 1]); valid = np.array([[1, 1, 0], [1, 1, 1]], bool)
        np.testing.assert_array_equal(capped_correct(pred, labels, valid), [[[0, 1, 1], [1, 1, 0]]])

    def test_paired_groups_not_head_replicas_are_bootstrap_units(self):
        result = cluster_delta(np.ones(6), np.zeros(6), np.array([0, 1, 0, 1, 0, 1]), draws=32)
        self.assertEqual((result["paired_delta"], result["ci95"], result["seed_group_count"]), (1., [1., 1.], 2))
        same = cluster_delta(np.arange(6) / 6, np.arange(6) / 6, np.array([0, 1, 0, 1, 0, 1]), draws=32)
        self.assertEqual(same["ci95"], [0., 0.])
        with self.assertRaises(ValueError): cluster_delta(np.ones(3), np.zeros(3), [0, 0, 1])


if __name__ == "__main__": unittest.main()
