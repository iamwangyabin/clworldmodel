"""Pure/fixture contracts; no real environment actions or optimizer steps."""

from dataclasses import replace
from contextlib import redirect_stdout
import io
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from clworldmodel.continual_dreamer import ContinualDreamerConfig, should_update
from clworldmodel.environments.episode_stream import EpisodeStream
from clworldmodel.exploration import DisagreementEnsemble
from clworldmodel.replay.episode_slots import EpisodeSlotsReplay

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from continual_dreamer_v3_support import isolated_evaluation_rng, resolve_native_config, verify_source, SourceRecipeAgent
from run_continual_dreamer_v3 import build_manifest


def episode(identifier=0, length=5):
    """Small complete episode fixture with incoming one-hot actions."""
    result = dict(
        image=np.full((length + 1, 2, 2, 3), identifier, np.uint8),
        action=np.zeros((length + 1, 3), np.float32),
        reward=np.zeros(length + 1, np.float32),
        is_first=np.zeros(length + 1, bool), is_last=np.zeros(length + 1, bool),
        is_terminal=np.zeros(length + 1, bool),
    )
    result["action"][1:, 0] = 1
    result["is_first"][0] = True
    result["is_last"][-1] = True
    return result


def replay(method="rs", capacity=4, seed=0):
    return EpisodeSlotsReplay(capacity, method=method, min_decisions=3,
        max_decisions=6, sequence_length=3, seed=seed)


class RecipeTests(unittest.TestCase):
    def test_sources_and_backbone_recipe_resolution(self):
        self.assertFalse(verify_source()["local_modifications"])
        cfg = ContinualDreamerConfig()
        native = resolve_native_config(cfg)
        self.assertEqual((native.batch_size, native.batch_length), (16, 50))
        self.assertEqual((native.dyn_stoch, native.dyn_discrete), (32, 32))
        self.assertEqual(native.actor["dist"], "onehot")
        self.assertEqual(native.actor["lr"], 3e-5)
        self.assertEqual(native.actor["entropy"], 3e-3)
        self.assertFalse(native.eval_state_mean)
        self.assertEqual(native.grad_heads, ["decoder", "reward", "cont"])

    def test_no_implicit_cli_override_unknown_and_frozen_keys(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cfg.json"
            path.write_text('{"method":"rs", "seed":4}')
            cfg = ContinualDreamerConfig.from_json(path, seed=None, method=None)
            self.assertEqual((cfg.method, cfg.seed), ("rs", 4))
            self.assertEqual(ContinualDreamerConfig.from_json(path, seed=2).seed, 2)
            path.write_text('{"stealth_task_label":true}')
            with self.assertRaises(ValueError):
                ContinualDreamerConfig.from_json(path)
        for kwargs in ({"batch_size": 8}, {"seed": True}, {"decisions_per_task": 10000}, {"task_count": 2}, {"evidence_level": "official"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ContinualDreamerConfig(**kwargs)

    def test_action_cadence_and_total_budget(self):
        cfg = ContinualDreamerConfig()
        self.assertFalse(should_update(10000, cfg))
        self.assertTrue(should_update(10001, cfg))
        self.assertFalse(should_update(10010, cfg))
        self.assertTrue(should_update(10011, cfg))
        self.assertEqual(cfg.expected_updates, 74002)
        self.assertEqual(replace(cfg, task_count=3).expected_updates, 224002)

    def test_matched_arms_and_honest_capacity(self):
        cfg = ContinualDreamerConfig()
        first = build_manifest(cfg, resolve_native_config(cfg), Path("/tmp/unused"))
        rs = replace(cfg, method="rs")
        second = build_manifest(rs, resolve_native_config(rs), Path("/tmp/unused"))
        for key in ("resolved_native_config", "budgets", "environment", "evaluation"):
            self.assertEqual(first[key], second[key])
        self.assertEqual(first["replay"]["capacity_action_upper_bound"], 2000000)
        self.assertEqual(first["replay"]["fifo_minibatch_probability"], 0.5)
        self.assertFalse(first["checkpoint"]["resumable"])

    def test_dry_run_creates_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "never_created"
            result = subprocess.run([sys.executable, str(ROOT / "scripts/run_continual_dreamer_v3.py"),
                "--dry-run", "--output-dir", str(output), "--method", "rs"],
                check=True, text=True, capture_output=True, cwd="/tmp")
            value = json.loads(result.stdout)
            self.assertEqual(value["resolved_config"]["method"], "rs")
            self.assertFalse(output.exists())

    def test_replay_leaf_import_outside_checkout_needs_no_r2_vendor(self):
        subprocess.run([sys.executable, "-c",
            "import sys; from clworldmodel.replay.episode_slots import EpisodeSlotsReplay; "
            "assert 'clworldmodel.r2dreamer.agent' not in sys.modules"],
            check=True, cwd="/tmp", capture_output=True)


class ReplayTests(unittest.TestCase):
    def test_partial_capacity_wrap_and_short_episodes(self):
        store = replay("arrow50")
        self.assertFalse(store.add(episode(length=2)))
        with self.assertRaises(ValueError):
            store.sample(1)
        for n in range(6):
            store.add(episode(n))
        self.assertEqual([int(e["image"][0, 0, 0, 0]) for e in store.fifo], [4, 5])
        self.assertEqual(len(store.reservoir), 2)
        self.assertEqual(store.eligible_seen, 6)
        self.assertEqual(store.rejected_short, 1)
        self.assertEqual(store.accounting()["retained_action_occupancy"], 20)

    def test_seeded_retention_independent_of_sampling(self):
        a, b = replay(seed=9), replay(seed=9)
        for n in range(30):
            a.add(episode(n)); b.add(episode(n))
            a.sample(2)
        self.assertEqual([int(e["image"][0, 0, 0, 0]) for e in a.reservoir],
                         [int(e["image"][0, 0, 0, 0]) for e in b.reservoir])

    def test_algorithm_r_uniform_retention(self):
        counts = np.zeros(8)
        for seed in range(800):
            store = replay(capacity=2, seed=seed)
            for n in range(8):
                store.add(episode(n))
            for ep in store.reservoir:
                counts[int(ep["image"][0, 0, 0, 0])] += 1
        self.assertTrue(np.all(abs(counts - 200) < 45), counts)

    def test_source_end_prioritization_and_whole_batch_selection(self):
        store = replay("arrow50")
        for n in range(4):
            ep = episode(n)
            ep["reward"][:] = np.arange(6)
            store.add(ep)
        for _ in range(1000):
            sample = store.sample(3)
            self.assertEqual(sample["image"].shape, (3, 3, 2, 2, 3))
            self.assertEqual(sample["image"].dtype, np.uint8)
            self.assertTrue(sample["is_first"][:, 0].all())
            self.assertFalse(sample["is_first"][:, 1:].any())
            self.assertTrue((np.diff(sample["reward"], axis=1) == 1).all())
        self.assertLess(abs(store.fifo_selections / 1000 - 0.5), 0.06)
        # Exact source draw range 0..(last_start + minlen), clamped.
        class LastDraw:
            def integers(self, upper): return upper - 1
        store.sampling_rng = LastDraw()
        self.assertTrue((store.sample(4)["reward"][:, 0] == 3).all())

    def test_shapes_terminal_reset_and_ownership(self):
        store = replay()
        original = episode(7)
        store.add(original)
        original["image"][:] = 99
        self.assertTrue((store.sample(1)["image"] == 7).all())
        store.sample(1)["image"][:] = 88
        self.assertTrue((store.sample(1)["image"] == 7).all())
        for invalid in ("task_id", "early_terminal", "float_image"):
            ep = episode()
            if invalid == "task_id": ep["task_id"] = np.zeros(6)
            if invalid == "early_terminal": ep["is_terminal"][1] = True
            if invalid == "float_image": ep["image"] = ep["image"].astype(np.float32)
            with self.assertRaises(ValueError): store.add(ep)
        stats = store.accounting()
        self.assertEqual(stats["payload_bytes"], sum(a.nbytes for a in store.reservoir[0].values()))
        self.assertGreater(stats["accounted_bytes"], stats["payload_bytes"])


class P2ETests(unittest.TestCase):
    def test_population_std_and_gaussian_nll_reduction(self):
        ensemble = DisagreementEnsemble(2, 1, 2, models=2, hidden_layers=1, hidden_features=3)
        with torch.no_grad():
            for i, model in enumerate(ensemble.models):
                for p in model.parameters(): p.zero_()
                model[-1].bias.fill_(2 * i)
        states, actions = torch.ones(3, 4, 2), torch.ones(3, 4, 1)
        target = torch.zeros(3, 4, 2)
        self.assertTrue(torch.equal(ensemble.disagreement(states, actions), torch.ones(3, 4, 1)))
        self.assertEqual(ensemble.prediction_loss(states, actions, target).item(), 4.0)

    def test_ensemble_does_not_backprop_to_world_model(self):
        ensemble = DisagreementEnsemble(2, 1, 2, models=2, hidden_layers=1, hidden_features=3)
        states = torch.ones(3, 4, 2, requires_grad=True)
        actions = torch.ones(3, 4, 1, requires_grad=True)
        target = torch.zeros(3, 4, 2, requires_grad=True)
        ensemble.prediction_loss(states, actions, target).backward()
        self.assertIsNone(states.grad); self.assertIsNone(actions.grad); self.assertIsNone(target.grad)
        self.assertTrue(any(p.grad is not None for p in ensemble.parameters()))

    def test_native_outgoing_actions_are_shifted_to_source_incoming(self):
        class Recorder:
            def disagreement(self, feat, action):
                self.actions = action
                return torch.ones(*feat.shape[:-1], 1)
        obj = SourceRecipeAgent.__new__(SourceRecipeAgent)
        obj.recipe = ContinualDreamerConfig()
        obj.ensemble = Recorder()
        obj.task_reward = lambda feat, state, action: torch.ones(*feat.shape[:-1], 1) * 2
        feat, action = torch.zeros(4, 2, 5), torch.arange(24).reshape(4, 2, 3).float()
        result = obj.exploration_reward(feat, {}, action)
        self.assertTrue(torch.equal(obj.ensemble.actions[1:], action[:-1]))
        self.assertTrue(torch.allclose(result, torch.full((4, 2, 1), 2.7)))

    def test_evaluation_restores_python_numpy_and_torch_rng(self):
        random.seed(8); np.random.seed(8); torch.manual_seed(8)
        py, numpy, th = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        with isolated_evaluation_rng(12):
            random.random(); np.random.random(); torch.rand(3)
        self.assertEqual(random.getstate(), py)
        self.assertTrue(np.array_equal(np.random.get_state()[1], numpy[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), th))


class CollectionTests(unittest.TestCase):
    def test_three_task_runner_counters_and_evaluation_isolation_fixture(self):
        import gymnasium as gym
        import run_continual_dreamer_v3 as runner

        class Env(gym.Env):
            observation_space = gym.spaces.Box(0, 255, (2, 2, 3), np.uint8)
            def __init__(self):
                self.action_space = gym.spaces.Discrete(7)
                self.position = 0
            def reset(self, seed=None):
                self.position = 0
                return np.zeros((2, 2, 3), np.uint8), {}
            def step(self, action):
                self.position += 1
                last = self.position == 4
                return np.full((2, 2, 3), self.position, np.uint8), float(last), last, False, {}

        class Agent:
            instances = []
            def __init__(self, *args):
                self.updates = 0
                self.instances.append(self)
            def act(self, observation, state=None, *, evaluation=False):
                if set(observation) != {"image", "is_first", "is_terminal"}:
                    raise AssertionError("Privileged input at agent boundary")
                return 0, None
            def update(self, data):
                if set(data) != {"image", "reward", "action", "is_first", "is_last", "is_terminal"}:
                    raise AssertionError("Privileged data in replay")
                self.updates += 1
                return {"fixture_updates": self.updates}
            def save_inference_snapshot(self, *args): pass

        values = ContinualDreamerConfig().as_dict()
        values.update(task_count=3, decisions_per_task=12, total_decisions=36,
            prefill_decisions=4, initial_updates=2, train_every_decisions=2,
            batch_size=2, batch_length=3, episode_limit=4, replay_episode_slots=4,
            eval_every_decisions=4, eval_decisions_per_task=8, log_every_decisions=4,
            expected_updates=18, device="cpu")
        cfg = SimpleNamespace(**values)
        with tempfile.TemporaryDirectory() as d, patch.object(runner, "SourceRecipeAgent", Agent), patch(
            "clworldmodel.environments.minigrid.make_minigrid_environment", side_effect=lambda *a, **k: Env()
        ), redirect_stdout(io.StringIO()):
            path = Path(d)
            manifest = {"resolved_config": values}
            runner.run(cfg, None, path, manifest)
            records = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual(len(Agent.instances), 1)  # No model reset at task boundaries.
        self.assertEqual(Agent.instances[0].updates, 18)
        counts = manifest["final_counters"]
        self.assertEqual(counts["environment_decisions"], 36)
        self.assertEqual(counts["collected_transitions"], 36)
        self.assertEqual(counts["world_model_updates"], 18)
        self.assertEqual(counts["evaluation_decisions"], 3 * (1 + 2 + 3) * 8)
        self.assertEqual(manifest["replay_final"]["eligible_episodes_seen"], 9)
        self.assertEqual([r["task_index"] for r in records if r["kind"] == "task_boundary"], [0, 1, 2])
        evaluations = [r for r in records if r["kind"] == "evaluation"]
        self.assertEqual(len(evaluations), 18)
        self.assertTrue(all(r["raw_returns"] == [1.0, 1.0] for r in evaluations))

    def test_terminal_observation_action_reward_and_timeout_bootstrap(self):
        class Env:
            action_space = SimpleNamespace(n=3, seed=lambda seed: None)
            def reset(self, seed=None): return np.zeros((2, 2, 3), np.uint8), {}
            def step(self, action):
                return np.full((2, 2, 3), 17, np.uint8), 0.7, self.terminal, not self.terminal, {}
        for terminal in (True, False):
            env = Env(); env.terminal = terminal
            stream = EpisodeStream(env, 0)
            stream.reset()
            obs, ep, reward = stream.step(2)
            self.assertEqual(reward, 0.7)
            self.assertEqual(ep["reward"][0], 0)
            self.assertEqual(ep["action"][-1].tolist(), [0, 0, 1])
            self.assertTrue((ep["image"][-1] == 17).all())
            self.assertTrue(ep["is_last"][-1])
            self.assertEqual(ep["is_terminal"][-1], terminal)
            self.assertNotIn("task_id", obs)
            with self.assertRaises(RuntimeError): stream.step(1)

    def test_pil_nearest_not_legacy_area(self):
        import gymnasium as gym
        from PIL import Image
        from clworldmodel.environments.minigrid import _ResizeRgbObservation
        env = gym.Env()
        env.observation_space = gym.spaces.Box(0, 255, (56, 56, 3), np.uint8)
        image = np.random.default_rng(1).integers(0, 256, (56, 56, 3), dtype=np.uint8)
        wrapper = _ResizeRgbObservation(env, "nearest")
        self.assertTrue(np.array_equal(wrapper.observation(image), np.asarray(Image.fromarray(image).resize((64, 64), Image.Resampling.NEAREST))))


if __name__ == "__main__":
    unittest.main()
