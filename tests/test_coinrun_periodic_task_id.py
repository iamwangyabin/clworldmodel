# SPDX-License-Identifier: Apache-2.0
"""Sparse routing, state ownership and task-ID denominators; no simulator."""
import unittest
from unittest import mock
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from test_coinrun_prefix_refinement import CannedEnv, models
from eval_coinrun_periodic_task_id import EvaluationConfig, evaluate_episode, restore_route_state, audit_pair, summarize, report
from artifact_io import sha256_file, write_json_atomic
from clworldmodel.evaluation.metrics import task_id_trace_accuracy


class PeriodicRoutingTests(unittest.TestCase):
    def test_sparse_checks_hold_routes_and_only_restore_on_switch(self):
        wm, actors = models()
        cfg = EvaluationConfig(intervals=(2, 3), window_frames=3)
        env = CannedEnv(6)
        row, frames = evaluate_episode(wm, actors, env, env_seed=8, interval=2,
                                      cfg=cfg, max_decisions=9, dummy=0)
        self.assertEqual(row["route_trace"], [0, 0, 1, 1, 1, 1])
        self.assertEqual(row["actions"], row["route_trace"])
        self.assertEqual([e["after_agent_decisions"] for e in row["rechecks"]], [2, 4])
        self.assertEqual([e["window_start"] for e in row["rechecks"]], [0, 2])
        self.assertEqual([e["switched"] for e in row["rechecks"]], [True, False])
        self.assertEqual(row["rssm_calls"], {"initial_probe": 2, "ordinary_filter": 5,
                         "window_probe": 12, "selected_history_restore": 3, "total": 22})
        self.assertEqual(len(wm.rssm.calls), 22)
        self.assertEqual(wm.rssm.calls[-1][2].item(), 15)  # Full incumbent state, not window's h=9.
        self.assertEqual(frames.flatten().tolist(), [0] + [255] * 5)
        self.assertTrue(env.closed)
        baseline, baseline_frames = evaluate_episode(*models(), CannedEnv(6), env_seed=8,
            interval=0, cfg=cfg, max_decisions=9, dummy=0)
        audit_pair(baseline, baseline_frames, row, frames)

    def test_switch_after_truncated_window_restores_full_actual_history(self):
        wm, actors = models()
        cfg = EvaluationConfig(intervals=(4,), window_frames=3)
        with mock.patch("eval_coinrun_periodic_task_id.restore_route_state", wraps=restore_route_state) as restore:
            row, _ = evaluate_episode(wm, actors, CannedEnv(6), env_seed=8, interval=4,
                                      cfg=cfg, max_decisions=9, dummy=0)
        self.assertEqual(row["route_trace"], [0, 0, 0, 0, 1, 1])
        self.assertEqual(row["rechecks"][0]["window_start"], 2)
        self.assertEqual(restore.call_args.args[1].shape[0], 5)
        self.assertEqual(restore.call_args.args[2].tolist(), [0] * 4)
        self.assertEqual(wm.rssm.calls[-1][2].item(), 15)  # 5 frames under route 1, once each.
        self.assertEqual(row["rssm_calls"]["total"], len(wm.rssm.calls))

    def test_early_terminal_and_no_switch_preserve_baseline(self):
        cfg = EvaluationConfig(intervals=(2,), window_frames=3)
        row, frames = evaluate_episode(*models(), CannedEnv(2), env_seed=8, interval=2,
                                      cfg=cfg, max_decisions=4, dummy=0)
        baseline, reference = evaluate_episode(*models(), CannedEnv(2), env_seed=8, interval=0,
                                             cfg=cfg, max_decisions=4, dummy=0)
        self.assertEqual(row["rechecks"], [])
        self.assertEqual(row["rssm_calls"]["total"], 4)
        audit_pair(baseline, reference, row, frames)
        bad = dict(row, actions=[1, 0])
        with self.assertRaises(RuntimeError): audit_pair(baseline, reference, bad, frames)
        env = CannedEnv(9)
        with self.assertRaisesRegex(RuntimeError, "no partial"):
            evaluate_episode(*models(), env, env_seed=8, interval=2, cfg=cfg, max_decisions=1, dummy=0)
        self.assertTrue(env.closed)

    def test_accuracy_scores_all_held_decisions_not_only_rechecks(self):
        result = task_id_trace_accuracy([[0, 0, 1, 1], [1]], [0, 1])
        self.assertEqual(result["episode_accuracy"], [.5, 1.])
        self.assertEqual(result["episode_mean_accuracy"], .75)
        self.assertEqual(result["decision_weighted_accuracy"], .6)
        self.assertEqual(result["initial_accuracy"], 1.)
        self.assertEqual(result["final_accuracy"], .5)
        for traces, labels in (([], []), ([[]], [0]), ([[0]], []), ([[float("nan")]], [0])):
            with self.assertRaises(ValueError): task_id_trace_accuracy(traces, labels)

    def test_report_recomputes_complete_cohort_from_raw_traces(self):
        cfg = EvaluationConfig(episodes_per_task=2, intervals=(2,), window_frames=3)
        with TemporaryDirectory() as directory:
            output, rows = Path(directory), []
            for episode in range(2):
                for interval in (0, 2):
                    row, frames = evaluate_episode(*models(), CannedEnv(5), env_seed=episode, interval=interval,
                                                  cfg=cfg, max_decisions=6, dummy=0)
                    path = output / f"{episode}_{interval}.npz"
                    np.savez_compressed(path, observations=frames, actions=row["actions"],
                                        rewards=row["raw_rewards"], routes=row["route_trace"])
                    row.update(task_index_for_audit_only=0, episode_index=episode,
                               trajectory_file=path.name, trajectory_sha256=sha256_file(path))
                    rows.append(row)
            write_json_atomic(output / "manifest.json", {"resolved_evaluation_config": asdict(cfg),
                "task_names": ["a", "b"], "eligible_routes": [0, 1], "frozen_state_sha256_before": "same"})
            # Include the second task without giving its label to the policy.
            rows += [dict(row, task_index_for_audit_only=1) for row in rows]
            result = {"complete": True, "frozen_state_sha256_after": "same", "actual_agent_decisions": 40,
                      **summarize(rows, ["a", "b"], cfg)}
            write_json_atomic(output / "episodes.json", {"episodes": rows})
            write_json_atomic(output / "results.json", result)
            self.assertEqual(report(output), result)
            write_json_atomic(output / "episodes.json", {"episodes": rows[:-1]})
            with self.assertRaisesRegex(ValueError, "Incomplete"): report(output)


if __name__ == "__main__":
    unittest.main()
