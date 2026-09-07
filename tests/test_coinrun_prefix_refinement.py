# SPDX-License-Identifier: Apache-2.0
"""One-time refinement invariants using fixed tensors and a canned environment."""
import unittest
import numpy as np
import torch
from test_coinrun_sequence_routing_probe import ActionFixture
from clworldmodel.routing import RoutedActorBank
from clworldmodel.evaluation.metrics import paired_raw_return_summary
from eval_coinrun_prefix_refinement import EvaluationConfig, evaluate_episode, recheck_prefix


class CountingRSSM(ActionFixture):
    def __call__(self, *args, **kwargs):
        logits, z, h = super().__call__(*args, **kwargs)
        return logits, z, h + kwargs["task_id"] + 1


def models():
    wm = torch.nn.Module()
    wm.register_parameter("anchor", torch.nn.Parameter(torch.zeros(1)))
    wm.rssm, wm.a_dim, wm.compute_dtype = CountingRSSM(), 2, "float32"
    wm.decoder_for = lambda route: lambda zh: zh.new_full((len(zh), 1, 1, 1), float(route))
    actors = {}
    for route in (0, 1):
        actor = torch.nn.Linear(3, 2)
        with torch.no_grad():
            actor.weight.zero_(); actor.bias.zero_(); actor.bias[route] = 1
        actors[route] = actor
    return wm, RoutedActorBank(actors)


class CannedEnv:
    def __init__(self, decisions):
        self.decisions, self.count, self.closed = decisions, 0, False
    def reset(self, *, seed):
        return np.zeros((1, 1, 1), np.uint8), {}
    def step(self, action):
        self.count += 1
        done = self.count == self.decisions
        return np.full((1, 1, 1), 99 if done else 255, np.uint8), 10 if done else 0, done, False, {}
    def close(self):
        self.closed = True


class RefinementTests(unittest.TestCase):
    def test_recheck_returns_new_routes_own_state_not_old_hidden(self):
        wm, _ = models()
        route, z, h, scores = recheck_prefix(wm, torch.tensor([0., 1., 1.]).reshape(3, 1, 1, 1),
                                            torch.tensor([0, 0]), (0, 1), 0)
        self.assertEqual(route, 1)
        self.assertEqual(h.item(), 9)
        self.assertLess(scores[1], scores[0])
        self.assertEqual([r[2].item() for r in wm.rssm.calls], [0, 2, 4, 0, 3, 6])
        self.assertTrue(all(not r[1] for r in wm.rssm.calls))  # No prior needed.

    def test_once_only_current_frame_not_double_advanced_and_terminal_not_replayed(self):
        wm, actors = models()
        env = CannedEnv(6)
        row, prefix = evaluate_episode(wm, actors, env, env_seed=8, arm="prefix_once",
                                      cfg=EvaluationConfig(refine_after_decisions=2), max_decisions=9, dummy=0)
        self.assertEqual(row["actions"], [0, 0, 1, 1, 1, 1])
        self.assertEqual((row["initial_route"], row["final_route"], row["raw_return"]), (0, 1, 10))
        self.assertEqual(row["refinement"]["after_agent_decisions"], 2)
        self.assertEqual(len(wm.rssm.calls), 13)  # 2 initial probes + 2 ordinary + 6 recheck + 3 later.
        self.assertEqual(wm.rssm.calls[-3][2].item(), 9)  # First later step starts from restored state.
        self.assertEqual(prefix.flatten().tolist(), [0, 255, 255])
        self.assertTrue(env.closed)
        other, baseline_prefix = evaluate_episode(*models(), CannedEnv(6), env_seed=8, arm="first_frame",
                                                  cfg=EvaluationConfig(refine_after_decisions=2), max_decisions=9, dummy=0)
        self.assertEqual(other["actions"], [0] * 6)
        self.assertEqual(row["prefix_sha256"], other["prefix_sha256"])
        np.testing.assert_array_equal(prefix, baseline_prefix)

    def test_early_terminal_skips_refinement_and_cap_never_returns_partial_score(self):
        wm, actors = models()
        row, prefix = evaluate_episode(wm, actors, CannedEnv(1), env_seed=8, arm="prefix_once",
                                      cfg=EvaluationConfig(), max_decisions=2, dummy=0)
        self.assertIsNone(row["refinement"])
        self.assertEqual((len(prefix), row["raw_return"]), (1, 10))
        env = CannedEnv(9)
        with self.assertRaisesRegex(RuntimeError, "no partial return"):
            evaluate_episode(wm, actors, env, env_seed=8, arm="first_frame", cfg=EvaluationConfig(), max_decisions=1, dummy=0)
        self.assertTrue(env.closed)

    def test_paired_raw_returns_keep_losses_and_no_normalization(self):
        report = paired_raw_return_summary([0, 10, 10], [10, 0, 10])
        self.assertEqual((report["wins"], report["ties"], report["losses"], report["mean_paired_delta"]), (1, 1, 1, 0))
        self.assertAlmostEqual(report["candidate_mean"], 20 / 3)
        with self.assertRaises(ValueError): paired_raw_return_summary([0], [])
        with self.assertRaises(ValueError): paired_raw_return_summary([float("nan")], [0])


if __name__ == "__main__":
    unittest.main()
