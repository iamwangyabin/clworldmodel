"""Deterministic collector fixtures and diagnostic launch contracts.

No real Gym environment or optimizer step is used by these tests.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "third_party/arrow/Code/ARROW_and_DV3/Atari"
HAS_RUNTIME = all(importlib.util.find_spec(name) is not None for name in (
    "torch", "gymnasium", "cv2", "sortedcontainers"
))


class DiagnosticLauncherTests(unittest.TestCase):
    def test_explicit_counter_and_unchanged_learning_budget(self):
        with TemporaryDirectory() as directory:
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts/run_arrow_minigrid_learning_audit.py"),
                "--dry-run", "--output-dir", str(Path(directory) / "run"),
            ], check=True, capture_output=True, text=True, cwd=ROOT)
        data = json.loads(result.stdout.split("\ncommand:", 1)[0])
        self.assertEqual(data["evidence_level"], "pilot")
        self.assertEqual(len(data["task_schedule"]["tasks"]), 1)
        self.assertFalse(data["evaluation"]["future_tasks_evaluated"])
        self.assertEqual(data["evaluation"]["periodic_every_regular_environment_decisions"], 9960)
        self.assertEqual(data["budgets"]["nominal_stored_rows"], 750000)
        self.assertEqual(data["budgets"]["actual_environment_actions"], 747000)
        self.assertEqual(data["budgets"]["world_model_updates"], 45750)
        self.assertEqual(data["budgets"]["actor_critic_updates"], 36309)
        cfg = data["resolved_config"]["values"]
        self.assertEqual(cfg["mb_t_size"], 32)
        self.assertEqual(cfg["steps_per_batch"], 61)
        self.assertEqual(cfg["collection_autoreset_mode"], "same_step")
        self.assertEqual(data["replay"]["transition_capacity"], 2000000)
        self.assertEqual(data["replay"]["storage_device"], "cpu")
        self.assertFalse(Path(directory, "run").exists())


@unittest.skipUnless(HAS_RUNTIME, "requires experiment runtime packages")
class CollectorFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(VENDOR))
        import generate_trajectory
        import torch
        import numpy as np
        import gymnasium as gym
        cls.collector, cls.torch, cls.np, cls.gym = generate_trajectory, torch, np, gym

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(VENDOR))

    def fixture(self, mode, *, with_diagnostics=True, policy=False):
        np, torch, gym = self.np, self.torch, self.gym
        created = []
        observed_flags = []

        class VectorFixture:
            """Return a known stream; no user/environment constructor is called."""
            def __init__(self, fns, *, autoreset_mode):
                self.mode = autoreset_mode
                self.single_action_space = gym.spaces.Discrete(7)
                self.phase, self.episode, self.pending, self.closed = 0, 0, False, False
                created.append(self)

            def observation(self):
                return np.full((1, 64, 64, 3), self.episode * 10 + self.phase, dtype=np.uint8)

            def reset(self, seed=None):
                self.phase = 0
                return self.observation(), {}

            def step(self, action):
                if self.pending:
                    self.episode += 1
                    self.phase, self.pending = 0, False
                    return self.observation(), np.array([0.]), np.array([False]), np.array([False]), {}
                self.phase += 1
                done = self.phase == 2
                if done:
                    if self.mode == gym.vector.AutoresetMode.SAME_STEP:
                        self.episode += 1
                        self.phase = 0
                    else:
                        self.pending = True
                return self.observation(), np.array([float(done)]), np.array([done]), np.array([False]), {}

            def close(self):
                self.closed = True

        class RSSMFixture:
            def initial_state(self, n):
                return torch.zeros(n, 1, 1), torch.zeros(n, 1)

            def __call__(self, z, a, h, obs, reset, **kwargs):
                observed_flags.append((round(obs[0, 0, 0, 0].item() * 255), bool(reset.item())))
                return None, z, h

        class WorldFixture:
            rssm = RSSMFixture()

        class ActorFixture:
            def actor(self, state):
                return torch.zeros(state.shape[0], 7)

        diagnostics = {} if with_diagnostics else None
        np.random.seed(123)
        torch.manual_seed(123)
        with patch.object(self.collector, "AsyncVectorEnv", VectorFixture):
            values = self.collector.generate_trajectories(
                7, 1, wm=WorldFixture() if policy else None,
                ac=ActorFixture() if policy else None,
                env_fns=[lambda: None], env_repeat=1, seed=8,
                action_space=7, autoreset_mode=mode, diagnostics=diagnostics,
                deterministic_policy=policy,
            )
        self.assertTrue(created[0].closed)
        return values, diagnostics, observed_flags

    def test_same_step_matches_reset_flags_and_real_action_count(self):
        values, diagnostic, flags = self.fixture("same_step", policy=True)
        self.assertEqual(flags, [(0, False), (1, False), (10, True), (11, False), (20, True), (21, False)])
        self.assertEqual(values[4].flatten().tolist(), [1, 0, 1, 0, 1, 0, 1])
        self.assertEqual(diagnostic["actual_environment_actions"], 6)
        self.assertEqual(diagnostic["ignored_reset_actions"], 0)
        self.assertEqual(diagnostic["positive_reward_events"], 3)
        self.assertEqual(diagnostic["completed_episode_returns"], [1, 1, 1])
        self.assertEqual(values[2].sum().item(), diagnostic["observed_reward_sum"])

    def test_legacy_mismatch_is_reproduced_and_not_silently_redefined(self):
        _, diagnostic, flags = self.fixture("legacy_next_step", policy=True)
        self.assertEqual(flags, [(0, False), (1, False), (2, True), (10, False), (11, False), (12, True)])
        self.assertEqual(diagnostic["actual_environment_actions"], 4)
        self.assertEqual(diagnostic["ignored_reset_actions"], 2)

    def test_diagnostics_do_not_change_replay_tensors_or_rng(self):
        for mode in ("legacy_next_step", "same_step"):
            a, _, _ = self.fixture(mode, with_diagnostics=False)
            state_a = self.np.random.get_state()
            torch_state_a = self.torch.get_rng_state()
            b, _, _ = self.fixture(mode, with_diagnostics=True)
            state_b = self.np.random.get_state()
            torch_state_b = self.torch.get_rng_state()
            for x, y in zip(a, b):
                self.assertTrue(self.torch.equal(x, y))
            self.assertTrue(self.np.array_equal(state_a[1], state_b[1]))
            self.assertEqual(state_a[2:], state_b[2:])
            self.assertTrue(self.torch.equal(torch_state_a, torch_state_b))

    def test_sparse_reward_diagnostics_are_finite_and_hand_computed(self):
        from wm import reward_learning_diagnostics, symlog
        torch = self.torch
        pred = torch.tensor([0.25, 0.75, 0.0], requires_grad=True)
        target = torch.tensor([0., 1., 0.])
        rng = torch.get_rng_state()
        metrics = reward_learning_diagnostics(symlog(pred), target)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(not v.requires_grad for v in metrics.values()))
        self.assertAlmostEqual(metrics["LearningAudit/replay_positive_reward_fraction"].item(), 1/3, places=6)
        self.assertAlmostEqual(metrics["LearningAudit/positive_reward_absolute_error"].item(), .25, places=6)
        self.assertAlmostEqual(metrics["LearningAudit/zero_reward_absolute_error"].item(), .125, places=6)
        zero = reward_learning_diagnostics(symlog(pred), torch.zeros_like(target))
        self.assertTrue(all(torch.isfinite(v) for v in zero.values()))
        self.assertEqual(zero["LearningAudit/positive_reward_prediction_mean"].item(), 0)

    def test_epoch_summary_uses_positive_target_weighting_and_null_for_missing_data(self):
        from train import _reward_learning_epoch_summary
        summary = _reward_learning_epoch_summary([10, 2, 8, 1.0, 1.8, 0.8, 0.4])
        self.assertEqual(summary["sampled_positive_reward_targets"], 2)
        self.assertAlmostEqual(summary["positive_reward_prediction_mean"], .5)
        self.assertAlmostEqual(summary["positive_reward_absolute_error"], .4)
        self.assertAlmostEqual(summary["zero_reward_absolute_error"], .05)
        empty = _reward_learning_epoch_summary([10, 0, 10, 0, 0, 0, 0])
        self.assertIsNone(empty["positive_reward_absolute_error"])
        self.assertIsNone(empty["positive_reward_prediction_mean"])

    def test_evaluation_receives_the_same_explicit_autoreset_mode(self):
        import train
        config = SimpleNamespace(
            n_sync=1, env_repeat=1, action_space=7, uses_task_experts=False,
            deterministic_evaluation=True, collection_autoreset_mode="same_step",
        )
        with patch.object(train, "evaluate", return_value=(.5, .1)) as evaluate:
            train._evaluate_policy_tasks(
                config, wm=object(), aco=None, eval_funcs=[[lambda: None]], task_seeds=[17],
            )
        self.assertEqual(evaluate.call_args.kwargs["autoreset_mode"], "same_step")
        self.assertNotIn("task_id", evaluate.call_args.kwargs)

    def test_schema_preserves_legacy_default_and_rejects_unknown_mode(self):
        from config import Config
        data = json.loads((ROOT / "configs/minigrid/arrow_50_formal_v1.json").read_text())
        self.assertEqual(Config.from_dict(data).collection_autoreset_mode, "legacy_next_step")
        self.assertFalse(Config.from_dict(data).learning_diagnostics)
        data.update(collection_autoreset_mode="same_step", learning_diagnostics=True)
        self.assertEqual(Config.from_dict(data).collection_autoreset_mode, "same_step")
        data["collection_autoreset_mode"] = "typo"
        with self.assertRaisesRegex(ValueError, "collection_autoreset_mode"):
            Config.from_dict(data)
        data.update(collection_autoreset_mode="same_step", learning_diagnostics="true")
        with self.assertRaisesRegex(ValueError, "learning_diagnostics"):
            Config.from_dict(data)


if __name__ == "__main__":
    unittest.main()
