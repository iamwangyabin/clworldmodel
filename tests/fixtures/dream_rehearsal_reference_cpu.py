"""Generated, deterministic CPU fixtures; no ROMs, environment run or optimizer step.

Provenance: synthetic arrays defined below. The tested reference functions are
hash-checked against the commits in third_party/*/provenance.json. Optimizer
spies inspect losses/gradients without mutating parameters. Model-size changes
exist only in this fixture, never in a launchable baseline preset.
"""

from collections import OrderedDict
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import gym
import gymnasium
import numpy as np
import torch

from dream_rehearsal_reference_support import (
    load_reference_runtime, resolve_model_config, verify_reference_sources,
)
from clworldmodel.reference.atari import AtariNM512, FrameCounter
from clworldmodel.reference.dream_rehearsal import MemoryPairConfig, OfficialDreamRehearsalConfig
from clworldmodel.reference.episode_history import EpisodeHistory
from dream_rehearsal_collection import RetainedHistoryCollector
from train_dream_rehearsal_official_atari import (
    EpisodeArchive, isolated_evaluation_rng, replay_accounting,
)

D, tools, reference, parallel, wrappers = load_reference_runtime()
torch.set_num_threads(1)


def episode(value, length=5):
    return {
        "image": [np.full((8, 8, 3), value, np.uint8) for _ in range(length)],
        "reward": [np.float32(0)] + [np.float32(value)] * (length - 1),
        "discount": [np.float32(1)] * length,
        "is_first": [True] + [False] * (length - 1),
        "is_last": [False] * (length - 1) + [True],
        "is_terminal": [False] * (length - 1) + [True],
        "action": [np.zeros(18, np.float32)] + [np.eye(18, dtype=np.float32)[value % 18]] * (length - 1),
        "logprob": [np.float32(0)] * length,
    }


class ReferenceSourceTests(unittest.TestCase):
    def test_source_rehearsal_generator_tracks_eviction_not_a_frozen_backup(self):
        class FixedChoices:
            def __init__(self):
                self.choices = iter((0, 1))

            def randrange(self, n):
                return next(self.choices)

        cfg = SimpleNamespace(batch_length=4, batch_size=2)
        with TemporaryDirectory() as td:
            h = EpisodeHistory(Path(td) / "h", total_decisions=16, capacity_transitions=8,
                               block_length=4, rng=FixedChoices(), dataset_factory=lambda eps: D.make_dataset(eps, cfg))

            def feed(key, value, phase):
                e = episode(value)
                initial = {k: np.asarray(v[0]) for k, v in e.items()}
                h.begin_episode(key, initial, phase=phase, online=True)
                for i in range(1, 5):
                    h.add_transition(key, {k: np.asarray(v[i]) for k, v in e.items()})

            feed("old0", 1, 0)
            feed("old1", 2, 0)
            h.finish_phase(0)
            ds = h.rehearsal_datasets[0]
            next(ds)  # Suspend the original generator before a slot eviction.
            feed("new0", 3, 1)
            self.assertEqual(set(h.phase_episodes[0]), {"old1"})
            for _ in range(5):
                batch = next(ds)
                self.assertTrue(np.all(batch["image"] == 2))
            feed("new1", 4, 1)
            self.assertFalse(h.phase_episodes[0])
            with self.assertRaisesRegex(RuntimeError, "No retained data"):
                h.require_rehearsal_phase(0)
            self.assertEqual(h.accounting()["collected_transitions_retained"], 8)

    def test_memory_pair_resolves_identical_original_model_and_optimizer_config(self):
        full = resolve_model_config(MemoryPairConfig(device="cpu"), Path("/unused"), tools_module=tools)
        bounded = resolve_model_config(MemoryPairConfig(device="cpu", history_capacity_transitions=524288),
                                       Path("/unused"), tools_module=tools)
        self.assertEqual(vars(full), vars(bounded))

    def test_streaming_collector_and_source_sampler_match_upstream_before_any_eviction(self):
        # Fixed API-response fixture, not an environment campaign. No optimizer
        # exists. Exercise prefill, partial episodes, reset, chunk continuation,
        # original episode reconstruction and sampling before every decision.
        class TraceEnv:
            def __init__(self):
                self.resets, self.t = 0, 0
                self.id = "not-reset"

            def obs(self, first=False):
                return {"image": np.full((4, 4, 3), self.resets * 10 + self.t, np.uint8),
                        "is_first": first, "is_last": self.t == 3, "is_terminal": self.t == 3}

            def reset(self):
                self.resets += 1
                self.t, self.id = 0, f"episode-{self.resets}"
                return self.obs(True)

            def step(self, action):
                self.t += 1
                reward = np.float32(np.argmax(action["action"]) + self.t)
                return self.obs(), reward, self.t == 3, {"discount": np.float32(self.t != 3)}

        class QuietLogger:
            step = 0

            def scalar(self, *args):
                pass

            def write(self, **kwargs):
                pass

        cfg = SimpleNamespace(num_actions=3, dataset_size=0, batch_length=5, batch_size=2)
        with TemporaryDirectory() as td:
            traces, episodes = [], []
            for mode, capacity in (("source", None), ("full", None), ("bounded", 64)):
                env = parallel.Damy(TraceEnv())
                logger = QuietLogger()
                output = Path(td) / mode
                if mode == "source":
                    cache, simulator, prefill_tools = OrderedDict(), tools.simulate, tools
                else:
                    history = EpisodeHistory(output / "history", total_decisions=15,
                                             capacity_transitions=capacity, block_length=4,
                                             rng=random.Random(4), dataset_factory=lambda eps: D.make_dataset(eps, cfg))
                    cache = history.episodes
                    collector = RetainedHistoryCollector(env, history, tools, phase=0, num_actions=3)
                    simulator = collector.simulate
                    prefill_tools = SimpleNamespace(OneHotDist=tools.OneHotDist, simulate=simulator)
                with torch.random.fork_rng():
                    torch.manual_seed(101)
                    reference.random_prefill(prefill_tools, cfg, [env], cache, output / "train_eps", logger, 4)
                if mode != "source":
                    collector.online = True
                dataset, trace = D.make_dataset(cache, cfg), []

                def policy(obs, reset, state):
                    batch = next(dataset)
                    trace.append((copy.deepcopy(obs), reset.copy(), state, batch))
                    state = 0 if state is None else state
                    return {"action": torch.eye(3)[state % 3][None], "logprob": torch.zeros(1)}, state + 1

                state = None
                for steps in (2, 5, 4):
                    state = simulator(policy, [env], cache, output / "train_eps", logger,
                                      limit=0, steps=steps, state=state)
                traces.append(trace)
                episodes.append([{k: np.asarray(v) if mode == "source" else v[:]
                                  for k, v in ep.items()} for ep in cache.values()])
                if mode != "source":
                    self.assertEqual(history.accounting()["collected_transitions_retained"], 15)
                    self.assertFalse((output / "train_eps").exists())
                    history.finish_phase(0)
                    with self.assertRaisesRegex(ValueError, "Evaluation"):
                        simulator(policy, [env], cache, output, logger, is_eval=True, episodes=1)
            for trace in traces[1:]:
                self.assertEqual(len(trace), len(traces[0]))
                for actual, expected in zip(trace, traces[0]):
                    for j in (0, 3):
                        self.assertEqual(set(actual[j]), set(expected[j]))
                        for key in actual[j]:
                            np.testing.assert_array_equal(actual[j][key], expected[j][key])
                            self.assertEqual(actual[j][key].dtype, expected[j][key].dtype)
                    np.testing.assert_array_equal(actual[1], expected[1])
                    self.assertEqual(actual[2], expected[2])
            for arm in episodes[1:]:
                self.assertEqual(len(arm), len(episodes[0]))
                for actual, expected in zip(arm, episodes[0]):
                    for key in expected:
                        np.testing.assert_array_equal(actual[key], expected[key])

    def test_ordinary_learning_forwards_same_shared_batch_to_world_model_and_behavior(self):
        batch, posterior, calls = {"old_and_current": object()}, {"stoch": object()}, []

        def wm_train(data):
            calls.append(("wm", data))
            return posterior, {}, {"fixture_loss": 0.}

        def behavior_train(start, reward):
            calls.append(("behavior", start))
            return (None, None, None, None, {"fixture_behavior_loss": 0.})

        fake = SimpleNamespace(
            _wm=SimpleNamespace(_train=wm_train),
            _task_behavior=SimpleNamespace(_train=behavior_train),
            _config=SimpleNamespace(expl_behavior="greedy"), _metrics={},
        )
        D.Dreamer._train(fake, batch)
        self.assertIs(calls[0][1], batch)
        self.assertIs(calls[1][1], posterior)
        self.assertEqual([x[0] for x in calls], ["wm", "behavior"])

    def test_source_evaluation_never_invokes_training_or_advances_training_counter(self):
        calls = []
        fake = SimpleNamespace(_step=19, _policy=lambda o, s, t: calls.append(t) or ({}, None))
        D.Dreamer.__call__(fake, {}, [False], training=False)
        self.assertEqual(calls, [False])
        self.assertEqual(fake._step, 19)

    def test_sources_and_config_are_normal_not_smoke(self):
        sources = verify_reference_sources()
        self.assertTrue(all(not s["modified"] for s in sources.values()))
        c = resolve_model_config(OfficialDreamRehearsalConfig(device="cpu"), Path("/unused"), tools_module=tools)
        self.assertEqual((c.batch_size, c.batch_length, c.train_ratio), (16, 64, 512))
        self.assertEqual((c.imag_horizon, c.pretrain, c.dataset_size), (15, 100, 0))
        self.assertEqual(c.actor["lr"], 3e-5)
        self.assertEqual(c.actor["unimix_ratio"], 0.01)
        # Preserve the author's published unused value key, not a silent fix.
        self.assertEqual(c.value["layers"], 5)
        self.assertEqual(c.critic["layers"], 2)

    def test_exact_source_update_cadence(self):
        cfg = OfficialDreamRehearsalConfig.smoke()
        once, every = tools.Once(), tools.Every(cfg.batch_size * cfg.batch_length / cfg.train_ratio)
        online = cfg.projected_budgets()["online_agent_decisions"]
        updates = sum(cfg.pretrain_updates if once() else every(i) for i in range(online))
        self.assertEqual(updates, cfg.projected_budgets()["world_model_updates"])

    def test_archive_preserves_source_sampler_batches_and_never_evicts(self):
        eps = OrderedDict(old=episode(1, 3), older=episode(2, 7), active=episode(3, 4))
        control = copy.deepcopy(eps)
        own_phase_view = OrderedDict(old=eps["old"])
        with TemporaryDirectory() as td:
            archive = EpisodeArchive(Path(td) / "pixels", 40, (8, 8, 3))
            archive.archive(eps, {"active"})
            self.assertIs(own_phase_view["old"], eps["old"])
            self.assertIsInstance(eps["old"]["image"], np.memmap)
            self.assertIsInstance(eps["active"]["image"], list)
            original = tools.from_generator(tools.sample_episodes(control, 9), 4)
            mapped = tools.from_generator(tools.sample_episodes(eps, 9), 4)
            for _ in range(10):
                a, b = next(original), next(mapped)
                self.assertEqual(set(a), set(b))
                for key in a:
                    np.testing.assert_array_equal(a[key], b[key])
                    self.assertEqual(a[key].dtype, b[key].dtype)
            before = set(eps)
            self.assertEqual(tools.erase_over_episodes(eps, 0), 11)
            self.assertEqual(set(eps), before)
            self.assertEqual(replay_accounting(eps)["collected_transitions_retained"], 11)
            self.assertFalse(eps["old"]["image"].flags.writeable)
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                EpisodeArchive(Path(td) / "pixels", 40, (8, 8, 3))

    def test_evaluation_rng_restores_all_training_random_streams(self):
        random.seed(7)
        np.random.seed(8)
        torch.manual_seed(9)
        ps, ns, ts = random.getstate(), np.random.get_state(), torch.get_rng_state()
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with isolated_evaluation_rng(torch, np, 123):
                random.random(), np.random.rand(), torch.rand(3)
                raise RuntimeError("fixture")
        self.assertEqual(ps, random.getstate())
        for a, b in zip(ns, np.random.get_state()):
            np.testing.assert_array_equal(a, b)
        self.assertTrue(torch.equal(ts, torch.get_rng_state()))


class Mode:
    def __init__(self, value):
        self.value = value

    def mode(self):
        return self.value


class FixedActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0.4, -0.3]))
        self.selected_features = None

    def forward(self, features):
        self.selected_features = features.detach().clone()
        return tools.OneHotDist(logits=self.logits.expand(*features.shape[:-1], 2), unimix_ratio=0.01)


class DreamGradingTests(unittest.TestCase):
    def test_realized_first_reach_survival_topk_bootstrap_and_actor_only_gradient(self):
        # Dream 1 has huge post-terminal reward, dream 2 makes only a promise,
        # dream 0 exactly equals .3 (strict >), dream 3 really succeeds late.
        feats = torch.arange(16, dtype=torch.float32).reshape(2, 8, 1)
        rewards = torch.tensor([[.3, 0, .2, 0, .31, .32, 0, 0],
                                [0, 100, 0, .5, 0, 0, 0, 0]])
        cont = torch.ones(2, 8)
        cont[0, 1] = 0
        values = torch.tensor([0., 0, 8., 0, 0, 0, 0, 0])
        actions = torch.nn.functional.one_hot(torch.arange(16).reshape(2, 8) % 2, 2).float()
        actor = FixedActor().requires_grad_(False)
        captured = {}

        def value_head(f):
            captured["bootstrap_features"] = f.clone()
            return Mode(values[:, None])

        def optimizer_spy(loss, parameters):
            captured["loss"] = float(loss.detach())
            params = tuple(parameters)
            captured["grads"] = torch.autograd.grad(loss, params)
            return {}

        wm = SimpleNamespace(
            preprocess=lambda batch: batch, encoder=lambda batch: batch["image"],
            dynamics=SimpleNamespace(observe=lambda *a: ({"stoch": torch.zeros(1, 8, 2)}, None)),
            heads={"reward": lambda f: Mode(rewards[..., None]),
                   "cont": lambda f: SimpleNamespace(mean=cont[..., None])},
        )
        behavior = SimpleNamespace(
            _config=SimpleNamespace(imag_horizon=2, discount=1.),
            _imagine=lambda *a: (feats, {}, actions), value=value_head,
            actor=actor, _actor_opt=optimizer_spy,
        )
        initial = actor.logits.detach().clone()
        loss = reference.tunnel_update(
            SimpleNamespace(_wm=wm, _task_behavior=behavior),
            {"image": torch.zeros(1), "action": torch.zeros(1), "is_first": torch.zeros(1)},
            topk_frac=.25, cont_grading=True,
        )
        # Hand-computed scores: [.3, 0, 8.2, 10.5, 10.31, 10.32, 0, 0].
        torch.testing.assert_close(actor.selected_features, feats[:, [3, 5]])
        torch.testing.assert_close(captured["bootstrap_features"], feats[-1])
        expected = -tools.OneHotDist(logits=initial, unimix_ratio=.01).log_prob(actions[:, [3, 5]]).mean()
        self.assertAlmostEqual(loss, expected.item(), places=6)
        self.assertTrue(torch.isfinite(captured["grads"][0]).all())
        self.assertGreater(captured["grads"][0].abs().sum().item(), 0.)
        self.assertFalse(actor.logits.requires_grad)
        torch.testing.assert_close(actor.logits, initial)  # Spy never updates parameters.

    def test_real_nm512_model_observe_imagine_and_rehearsal_boundary(self):
        cfg = resolve_model_config(OfficialDreamRehearsalConfig(device="cpu"), Path("/unused"), tools_module=tools)
        # Tiny deterministic tensor fixture only; no environment or training run.
        cfg.dyn_stoch, cfg.dyn_discrete, cfg.dyn_deter, cfg.dyn_hidden = 2, 4, 16, 16
        cfg.units, cfg.imag_horizon = 16, 3
        for head in (cfg.actor, cfg.critic, cfg.reward_head, cfg.cont_head):
            head["layers"] = 1
        cfg.encoder["cnn_depth"] = cfg.decoder["cnn_depth"] = 2
        space = gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, (8, 8, 3), dtype=np.uint8),
            "is_first": gym.spaces.Box(0, 1, (), dtype=np.uint8),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=np.uint8),
        })
        actions = gym.spaces.Box(0, 1, (18,), dtype=np.float32)
        actions.discrete = True
        torch.manual_seed(123)
        with redirect_stdout(io.StringIO()):
            agent = D.Dreamer(space, actions, cfg, SimpleNamespace(step=0), iter(())).requires_grad_(False)
        batch = next(tools.from_generator(tools.sample_episodes(OrderedDict(a=episode(1), b=episode(2)), 4), 2))
        captured = {}

        def gradient_spy(loss, parameters):
            captured["grads"] = torch.autograd.grad(loss, tuple(parameters))
            return {}

        before = {k: v.clone() for k, v in agent.state_dict().items()}
        with patch.object(agent._task_behavior, "_actor_opt", gradient_spy):
            loss = reference.tunnel_update(agent, batch, topk_frac=.25, cont_grading=True)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(all(torch.isfinite(g).all() for g in captured["grads"]))
        self.assertGreater(sum(g.abs().sum().item() for g in captured["grads"]), 0.)
        for k, v in agent.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)
        self.assertFalse(any(p.requires_grad for p in agent.parameters()))
        self.assertTrue(all(p.grad is None for p in agent._wm.parameters()))
        self.assertTrue(all(p.grad is None for p in agent._task_behavior.value.parameters()))


class FakeAtari(gymnasium.Env):
    """Fixed return values for adapter unit calls, not an environment campaign."""
    def __init__(self, terminal):
        self.action_space = gymnasium.spaces.Discrete(18)
        self.observation_space = gymnasium.spaces.Box(0, 255, (8, 8, 3), dtype=np.uint8)
        self.terminal, self.seeds, self.actions = terminal, [], []

    def reset(self, *, seed=None, options=None):
        self.seeds.append(seed)
        return np.zeros((8, 8, 3), np.uint8), {}

    def step(self, action):
        self.actions.append(action)
        return np.ones((8, 8, 3), np.uint8), 123.5, self.terminal, not self.terminal, {}


class AdapterTests(unittest.TestCase):
    def test_raw_reward_full_action_set_reset_and_terminal_vs_time_limit(self):
        for terminal in (True, False):
            fake = FakeAtari(terminal)
            counter = FrameCounter(fake)
            adapter = AtariNM512(counter, counter, seed=42)
            wrapped = wrappers.OneHotAction(adapter)
            self.assertTrue(wrapped.reset()["is_first"])
            obs, reward, done, info = wrapped.step(np.eye(18, dtype=np.float32)[17])
            self.assertEqual(fake.actions, [17])
            self.assertEqual(reward, 123.5)
            self.assertTrue(done and obs["is_last"])
            self.assertEqual(obs["is_terminal"], terminal)
            self.assertEqual(info["discount"], 0. if terminal else 1.)
            self.assertEqual((adapter.agent_decisions, adapter.raw_frames), (1, 1))
            self.assertEqual(adapter.episode_returns, [123.5])
            self.assertNotIn("task_id", obs)
            wrapped.reset()
            self.assertEqual(fake.seeds, [42, None])


if __name__ == "__main__":
    unittest.main(verbosity=2)
