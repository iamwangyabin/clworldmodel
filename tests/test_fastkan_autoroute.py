"""New D-derived protocol: task-aware training, task-ID-free inference.

These tests use fixed tensors or mocked orchestration, not Atari interaction.
"""

from __future__ import annotations

import json
import copy
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_evolving_atomic_rssm import (
    FASTKAN_AUTOROUTE_METHOD,
    FASTKAN_AUTOROUTE_PROTOCOL,
    SHARED_FASTKAN_AUTOROUTE_BEHAVIOR,
    _budget_manifest,
    _parameter_manifest,
    _protocol_for_task_order,
    _resolved_config,
)


def source_config() -> dict:
    folder = ROOT / "third_party/arrow/Configs/Atari configs/CL-task configs/Original Order"
    return json.loads(next(folder.glob("*Enduro-s0-arrow.json")).read_text())


def new_config() -> dict:
    return _resolved_config(
        source_config(), task_order="arrow-original-six",
        prediction_head_profile="shared_distilled", adaptive_qfp_compression=True,
        behavior_profile=SHARED_FASTKAN_AUTOROUTE_BEHAVIOR,
    )


class FastkanAutorouteLauncherTests(unittest.TestCase):
    def test_separate_protocol_keeps_d_world_model_and_update_budgets(self):
        data = new_config()
        d = _resolved_config(
            source_config(), task_order="arrow-original-six",
            prediction_head_profile="shared_distilled", adaptive_qfp_compression=True,
        )
        self.assertEqual(data["continual_method"], FASTKAN_AUTOROUTE_METHOD)
        for key in ("task_mechanism_parameterization", "task_mechanism_recurrent_width",
                    "task_shared_prediction_heads", "shared_prediction_distill_scale",
                    "evolving_task0_profile", "first_task_shared_core_lr",
                    "current_batch_n", "memory_batch_n", "replay_buffers"):
            self.assertEqual(data[key], d[key], key)
        self.assertEqual(data["task_route_inference"], "first_frame_reconstruction")
        self.assertFalse(data["task_private_actor_critic"])
        self.assertFalse(data["adaptive_behavior_residuals"])
        self.assertEqual(data["actor_network"], "fast_kan_ac_stable")
        budget = _budget_manifest(data)
        self.assertEqual(budget["total_world_model_optimizer_steps"], 552_000)
        self.assertEqual(budget["actor_critic_updates"], 432_000)
        self.assertEqual(budget["adaptive_compression_validation_rollouts"], 1680)
        self.assertEqual(_budget_manifest(d)["adaptive_compression_validation_rollouts"], 480)
        parameters = _parameter_manifest(data)
        self.assertEqual(parameters["world_model_parameters"], 42_601_625)
        self.assertEqual(parameters["behavior_parameters"], 1_700_670)
        self.assertEqual(parameters["per_later_task_behavior_growth"], 0)
        self.assertEqual(parameters["online_parameters"], 44_302_295)
        self.assertEqual(_protocol_for_task_order(
            "arrow-original-six", prediction_head_profile="shared_distilled",
            adaptive_qfp_compression=True,
            behavior_profile=SHARED_FASTKAN_AUTOROUTE_BEHAVIOR,
        ), FASTKAN_AUTOROUTE_PROTOCOL)

    def test_requires_d_not_shared_down_or_uncompressed_world_model(self):
        with self.assertRaises(ValueError):
            _resolved_config(source_config(), task_order="arrow-original-six",
                             behavior_profile=SHARED_FASTKAN_AUTOROUTE_BEHAVIOR)
        with self.assertRaises(ValueError):
            _resolved_config(source_config(), task_order="mspacman-boxing-crazyclimber",
                             prediction_head_profile="shared_distilled",
                             adaptive_qfp_compression=True,
                             behavior_profile=SHARED_FASTKAN_AUTOROUTE_BEHAVIOR)


try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    sys.path.insert(0, str(ROOT / "src"))
    from clworldmodel.routing import EpisodeReconstructionRouter

vendor_available = False
if torch is not None:
    try:
        sys.path.insert(0, str(ROOT / "third_party/arrow/Code/ARROW_and_DV3/Atari"))
        from config import Config
        import generate_trajectory as trajectory
        import train
        import numpy as np
        vendor_available = True
    except ModuleNotFoundError:
        pass


@unittest.skipIf(torch is None, "requires PyTorch")
class ReconstructionRouterTests(unittest.TestCase):
    def test_per_worker_first_frame_lock_and_reset(self):
        router = EpisodeReconstructionRouter((0, 1))
        calls = []

        def reconstruction(task_id, x):
            calls.append(task_id)
            return torch.full_like(x, task_id)

        x = torch.tensor([0., 1.]).reshape(2, 1, 1, 1)
        before = torch.random.get_rng_state().clone()
        ids = router.route(x, torch.tensor([True, True]), reconstruction)
        self.assertEqual(ids.tolist(), [0, 1])
        self.assertEqual(calls, [0, 1])
        ids = router.route(1 - x, torch.tensor([False, False]), reconstruction)
        self.assertEqual(ids.tolist(), [0, 1])
        self.assertEqual(calls, [0, 1])
        ids = router.route(1 - x, torch.tensor([False, True]), reconstruction)
        self.assertEqual(ids.tolist(), [0, 0])
        self.assertEqual([event["worker_index"] for event in router.events], [0, 1, 1])
        torch.testing.assert_close(torch.random.get_rng_state(), before)

    def test_tie_breaking_and_single_route_are_finite(self):
        for candidates in [(0,), (0, 2)]:
            router = EpisodeReconstructionRouter(candidates)
            x = torch.zeros(1, 1, 1, 1)
            self.assertEqual(router.route(x, torch.ones(1, dtype=torch.bool),
                                          lambda _, obs: obs).tolist(), [0])
            self.assertEqual(router.events[0]["margin"], 0.)

    def test_invalid_candidates_shapes_and_nonfinite_scores_fail_closed(self):
        for candidates in [(), (1, 0), (0, 0), (-1,), (True,)]:
            with self.assertRaises(ValueError):
                EpisodeReconstructionRouter(candidates)
        router = EpisodeReconstructionRouter((0, 1))
        x = torch.zeros(1, 1, 1, 1)
        with self.assertRaises(ValueError):
            router.route(x, torch.zeros(2, dtype=torch.bool), lambda _, obs: obs)
        with self.assertRaises(FloatingPointError):
            router.route(x, torch.ones(1, dtype=torch.bool),
                         lambda _, obs: obs * float("nan"))


def fixed_models():
    """Deterministic tensor fixtures: decoder i reconstructs uniform pixels i."""
    class Rssm(torch.nn.Module):
        def initial_state(self, n):
            return torch.zeros(n, 1, 1), torch.zeros(n, 2)

        def forward(self, z, action, h, obs, reset, *, task_id, stochastic):
            # This fixture records the adapter contract without a learned model.
            self.last_action = action.clone()
            state = torch.stack((torch.full((len(obs),), float(task_id)),
                                 h[:, 1] * (1 - reset[:, 0]) + 1), -1)
            return z, z, state

    class World(torch.nn.Module):
        a_dim = 18
        compute_dtype = "float32"

        def __init__(self):
            super().__init__()
            self.rssm = Rssm()

        def decoder_for(self, route_id):
            return lambda state: state[:, -2].reshape(-1, 1, 1, 1).expand(-1, 3, 2, 2)

    class Behavior(torch.nn.Module):
        def actor(self, state):
            return torch.nn.functional.one_hot(state[:, -2].long(), 18).float()

        def set_task_route(self, _):
            raise AssertionError("A shared actor must not receive the true task ID")

    return World(), Behavior()


@unittest.skipUnless(vendor_available, "requires pinned Atari imports (no ROM interaction)")
class FastkanAutorouteIntegrationTests(unittest.TestCase):
    def test_shared_checkpoint_restores_registry_teacher_targets_and_rng_without_updates(self):
        from test_evolving_atomic_rssm import EvolvingAtomicRssmTests
        from ac import build_actor_critic_opt
        config = Config.from_dict(new_config())
        wm = EvolvingAtomicRssmTests._world_model("adaptive_dense_width", shared_prediction_heads=True)
        teacher = copy.deepcopy(wm)
        aco = build_actor_critic_opt(wm, lr=config.ac_lr, **train._actor_critic_constructor_kwargs(config))
        aco.return_scale_ema, aco.return_mean_ema = torch.tensor(2.), torch.tensor(-1.)
        optimizer = torch.optim.Adam([next(wm.parameters())], lr=config.shared_core_lr)
        replay = mock.Mock()
        replay.state_dict.return_value = {"fixture": "no collected transitions"}
        schedule = SimpleNamespace(_step=179)
        generators = [np.random.default_rng(i) for i in range(5)]
        expected = copy.deepcopy(train._actor_critic_opt_resumable_state_dict(aco))
        common = dict(
            config=config, wm=wm, boundary_teacher=teacher, shared_optimizer=optimizer,
            private_optimizers={}, route_optimizers={}, actor_critic_bank=None, aco=aco,
            shared_behavior_update_rng=generators[4], replay_buffer=replay,
            environment_schedule=schedule, task_update_rng=generators[0],
            collection_environment_seed_rng=generators[1],
            validation_environment_seed_rng=generators[2], final_environment_seed_rng=generators[3],
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "boundary.pt"
            train._save_evolving_resumable_checkpoint(
                path, **common, epoch=179, current_task_id=1, world_model_updates=0,
                actor_critic_updates=0, total_env_steps=0,
                shared_behavior_replay_updates={0: 0, 1: 0},
            )
            payload = torch.load(path, weights_only=False)
            self.assertEqual(payload["inference_routing"]["eligible_route_ids"], [0, 1])
            expected_draws = [g.integers(0, 100000) for g in generators]
            aco.return_scale_ema = None
            with torch.no_grad():
                next(aco.ac.actor.parameters()).add_(1)
                next(aco.slow_critic.parameters()).add_(1)
            restored = train._restore_evolving_resumable_checkpoint(path, **common, actor_critic_factory=None)
            self.assertEqual(restored["shared_actor_teacher_seen_tasks"], 2)
            self.assertEqual([g.integers(0, 100000) for g in generators], expected_draws)
            self.assertEqual(schedule._step, 180)
            replay.load_state_dict.assert_called_once_with({"fixture": "no collected transitions"})
            torch.testing.assert_close(aco.return_scale_ema, torch.tensor(2.))
            for key, value in aco.ac.state_dict().items():
                torch.testing.assert_close(value, expected["actor_critic"][key])
            for key, value in aco.slow_critic.state_dict().items():
                torch.testing.assert_close(value, expected["slow_critic"][key])
            self.assertTrue(all(not p.requires_grad for p in restored["shared_actor_teacher"].parameters()))
            payload["inference_routing"]["eligible_route_ids"] = [0, 1, 2]
            # Check the eligibility guard independently of the checksum guard.
            with mock.patch.object(train.torch, "load", return_value=payload):
                with self.assertRaisesRegex(ValueError, "eligibility"):
                    train._restore_evolving_resumable_checkpoint(path, **common, actor_critic_factory=None)

    def test_collection_locks_each_worker_and_uses_same_step_reset_frames(self):
        wm, ac = fixed_models()
        fixture = mock.Mock()
        pixels = lambda values: np.array(values, np.uint8)[:, None, None, None] * np.ones((2, 2, 2, 3), np.uint8)
        fixture.reset.return_value = (pixels([0, 255]), {})
        fixture.step.side_effect = [
            (pixels([255, 0]), np.ones(2), np.array([False, False]), np.zeros(2, bool), {}),
            (pixels([255, 0]), np.ones(2), np.array([True, False]), np.zeros(2, bool), {}),
            (pixels([255, 0]), np.ones(2), np.array([False, False]), np.zeros(2, bool), {}),
        ]
        diagnostic = {}
        with mock.patch.object(trajectory, "AsyncVectorEnv", return_value=fixture) as constructor:
            actions, _, _, _, resets = trajectory.generate_trajectories(
                8, 2, wm=wm, ac=ac, env_fns=[mock.sentinel.env] * 2,
                eligible_route_ids=(0, 1), seed=4, deterministic_policy=True,
                routing_diagnostics=diagnostic,
            )
        self.assertEqual(constructor.call_args.kwargs["autoreset_mode"], "SameStep")
        self.assertEqual([call.args[0].tolist() for call in fixture.step.call_args_list], [[0, 1], [0, 1], [1, 1]])
        self.assertEqual([e["selected_route_id"] for e in diagnostic["routing_events"]], [0, 1, 1])
        self.assertEqual(resets[:4, 0].tolist(), [1., 0., 1., 0.])
        self.assertEqual(actions[:4].argmax(-1).tolist(), [0, 0, 0, 1])
        fixture.close.assert_called_once()

    def test_typed_config_and_roundtrip_reject_protocol_leaks(self):
        data = new_config()
        config = Config.from_dict(data)
        self.assertEqual(Config.from_dict(config.to_dict()).to_dict(), config.to_dict())
        self.assertTrue(config.uses_shared_actor)
        self.assertTrue(config.uses_replay_rehearsed_shared_behavior)
        self.assertTrue(config.uses_adaptive_qfp_compression)
        self.assertTrue(config.uses_shared_prediction_heads)
        self.assertFalse(config.uses_adaptive_behavior_compression)
        for key, value in {
            "task_route_inference": "oracle", "evaluation_episode_count_mode": "legacy",
            "task_mechanism_parameterization": "shared_frozen_down_film",
            "task_private_actor_critic": True, "actor_network": "mlp",
            "evolving_task0_profile": "fixed_v2", "epochs": 270,
        }.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                Config.from_dict({**data, key: value})
        with self.assertRaises(TypeError):
            Config.from_dict({**data, "unknown_router_option": True})

    def test_grouped_posterior_uses_inferred_routes_and_resets_noop(self):
        wm, ac = fixed_models()
        router = EpisodeReconstructionRouter((0, 1))
        z, h = wm.rssm.initial_state(2)
        x = torch.tensor([0., 1.])[:, None, None, None].expand(2, 3, 2, 2)
        previous = torch.nn.functional.one_hot(torch.tensor([7, 8]), 18)
        rng = torch.random.get_rng_state().clone()
        z, h, action = trajectory._routed_policy_step(
            wm, ac, router, x, z, h, previous, torch.ones(2, 1), stochastic=False,
        )
        self.assertEqual(action.tolist(), [0, 1])
        self.assertEqual(wm.rssm.last_action.argmax(-1).tolist(), [0])
        z, h, action = trajectory._routed_policy_step(
            wm, ac, router, 1 - x, z, h, previous, torch.zeros(2, 1), stochastic=False,
        )
        self.assertEqual(action.tolist(), [0, 1])
        self.assertEqual(len(router.events), 2)
        z, h, action = trajectory._routed_policy_step(
            wm, ac, router, 1 - x, z, h, previous, torch.tensor([[0.], [1.]]), stochastic=False,
        )
        self.assertEqual(action.tolist(), [0, 0])
        torch.testing.assert_close(torch.random.get_rng_state(), rng)

    def test_exact_evaluation_consumes_fixed_mock_episodes_and_closes(self):
        # Canned step/reset responses only: no Gym constructor or simulator runs.
        wm, ac = fixed_models()
        wm.train()
        wm.rssm.eval()  # Mixed module modes must also survive evaluation.
        fixtures = []
        def factory(*_):
            env = mock.Mock()
            state = {}
            def reset(*, seed):
                state.update(length=0, target=1 + seed % 3, seed=seed)
                return np.full((2, 2, 3), 255 * (seed % 2), np.uint8), {}
            def step(action):
                state["length"] += 1
                done = state["length"] == state["target"]
                return np.zeros((2, 2, 3), np.uint8), 3., done, False, {}
            env.reset.side_effect = reset
            env.step.side_effect = step
            fixtures.append(env)
            return env
        before = torch.random.get_rng_state().clone()
        with mock.patch.object(trajectory, "_make_atari_env", side_effect=factory):
            for _ in range(2):
                diagnostic = {}
                mean, std = trajectory.evaluate(
                    2, wm, ac, [mock.sentinel.env_factory] * 2, n_rollouts=5,
                    seed=123, deterministic_policy=True, eligible_route_ids=(0, 1),
                    diagnostics=diagnostic,
                )
                self.assertEqual(diagnostic["completed_episodes"], 5)
                self.assertEqual(len(diagnostic["routing_events"]), 5)
                self.assertEqual([r["episode_index"] for r in diagnostic["episodes"]], list(range(5)))
                self.assertEqual(sum(e.step.call_count for e in fixtures[-2:]), diagnostic["agent_decisions"])
                self.assertEqual(mean, np.mean([r["scaled_return"] for r in diagnostic["episodes"]]))
                if _ == 0:
                    reference = copy.deepcopy(diagnostic)
                else:
                    self.assertEqual(diagnostic, reference)
        self.assertTrue(all(e.close.call_count == 1 for e in fixtures))
        self.assertTrue(wm.training)
        self.assertFalse(wm.rssm.training)
        torch.testing.assert_close(torch.random.get_rng_state(), before)

    def test_evaluation_cap_fails_instead_of_reporting_partial_return(self):
        wm, ac = fixed_models()
        fixture = mock.Mock()
        fixture.reset.return_value = (np.zeros((2, 2, 3), np.uint8), {})
        fixture.step.return_value = (np.zeros((2, 2, 3), np.uint8), 1., False, False, {})
        with mock.patch.object(trajectory, "_make_atari_env", return_value=fixture):
            with self.assertRaisesRegex(RuntimeError, "partial return"):
                trajectory.evaluate(1, wm, ac, [mock.sentinel.env], n_rollouts=1,
                                    seed=1, deterministic_policy=True, eligible_route_ids=(0,),
                                    max_agent_decisions_per_episode=2)
        fixture.close.assert_called_once()
        fixture.step.return_value = (np.zeros((2, 2, 3), np.uint8), float("nan"), False, False, {})
        with mock.patch.object(trajectory, "_make_atari_env", return_value=fixture):
            with self.assertRaisesRegex(FloatingPointError, "non-finite reward"):
                trajectory.evaluate(1, wm, ac, [mock.sentinel.env], n_rollouts=1,
                                    seed=1, deterministic_policy=True, eligible_route_ids=(0,))
        self.assertEqual(fixture.close.call_count, 2)

    def test_eval_task_labels_only_reach_posthoc_audit_and_future_routes_excluded(self):
        config = Config.from_dict(new_config())
        def evaluate(*args, **kwargs):
            self.assertNotIn("task_id", kwargs)
            self.assertEqual(kwargs["eligible_route_ids"], (0, 1))
            kwargs["diagnostics"]["routing_events"] = [{"selected_route_id": 0}]
            return 10., 0.
        diagnostics = []
        with mock.patch.object(train, "evaluate", side_effect=evaluate) as evaluator:
            train._evaluate_policy_tasks(
                config, mock.sentinel.wm, SimpleNamespace(ac=mock.sentinel.shared_ac),
                [[mock.sentinel.env]] * 6, tuple(range(6)), eligible_task_count=2,
                routing_diagnostics=diagnostics,
            )
            self.assertEqual(evaluator.call_count, 6)
        self.assertEqual([d["audit"]["true_task_id_for_audit_only"] for d in diagnostics], list(range(6)))
        self.assertEqual([d["true_task_is_eligible"] for d in diagnostics], [True, True, False, False, False, False])
        with self.assertRaisesRegex(ValueError, "acquired route count"):
            train._evaluate_policy_tasks(config, None, SimpleNamespace(ac=None), [[]], (1,))
        with self.assertRaisesRegex(ValueError, "forbids oracle"):
            trajectory.evaluate(1, task_id=0, eligible_route_ids=(0,), deterministic_policy=True)
        with self.assertRaisesRegex(ValueError, "forbids oracle"):
            trajectory.generate_trajectories(1, 1, task_id=0, eligible_route_ids=(0,))

    def test_old_task_return_drop_rejects_even_if_current_task_improves(self):
        before = {"raw_mean": 100., "seen_task_validation": [
            {"task_id": 0, "raw_mean": -10.}, {"task_id": 1, "raw_mean": 100.}]}
        after = {"raw_mean": 120., "seen_task_validation": [
            {"task_id": 0, "raw_mean": -11.}, {"task_id": 1, "raw_mean": 120.}]}
        passed, drops = train._adaptive_qfp_validation_gate(before, after, .05)
        self.assertFalse(passed)
        self.assertEqual(drops, [.1, -.2])
        after["seen_task_validation"][0]["raw_mean"] = -10.5
        self.assertTrue(train._adaptive_qfp_validation_gate(before, after, .05)[0])

    def test_real_fastkan_controller_has_one_pair_and_constant_size(self):
        from ac import build_actor_critic_opt
        config = Config.from_dict(new_config())
        wm = torch.nn.Linear(1, 1)
        wm.ls, wm.h_dim, wm.a_dim = (32, 32), 512, 18
        aco = build_actor_critic_opt(wm, lr=config.ac_lr, **train._actor_critic_constructor_kwargs(config))
        count = lambda module: sum(p.numel() for p in module.parameters())
        self.assertEqual(count(aco.ac.actor), 793_692)
        self.assertEqual(count(aco.ac.critic), 906_978)
        self.assertEqual(count(aco.slow_critic), 906_978)
        original_ids = [id(p) for p in aco.ac.parameters()]
        for task in range(6):
            aco.ac.set_task_route(task)
            self.assertEqual([id(p) for p in aco.ac.parameters()], original_ids)
        # Exercise serialization without any optimizer step or gradient update.
        saved = copy.deepcopy(train._actor_critic_opt_resumable_state_dict(aco))
        aco.return_scale_ema = torch.tensor(123.)
        train._load_actor_critic_opt_resumable_state_dict(aco, saved)
        self.assertIsNone(aco.return_scale_ema)
        self.assertEqual([id(p) for p in aco.ac.parameters()], original_ids)

    def test_compact_shared_decoder_world_model_routes_after_strict_reload(self):
        from test_evolving_atomic_rssm import EvolvingAtomicRssmTests
        from ac import build_actor_critic_opt
        wm = EvolvingAtomicRssmTests._world_model(
            "adaptive_dense_width", shared_prediction_heads=True,
        ).eval()
        teacher = copy.deepcopy(wm)
        ac = build_actor_critic_opt(wm, lr=4e-5, actor_network="fast_kan_ac_stable",
                                   fastkan_hidden_features=53).ac.eval()
        train._structured_adaptive_qfp_candidate(wm=wm, dense_teacher=teacher, task_id=1, fraction=.5)
        restored = copy.deepcopy(teacher)
        restored.load_state_dict(wm.state_dict(), strict=True)
        self.assertEqual(restored.rssm.adaptive_compression_layout(), wm.rssm.adaptive_compression_layout())
        self.assertIs(wm.decoder_for(0), wm.decoder_for(1))
        outputs = []
        for model in (wm, restored):
            z, h = model.rssm.initial_state(2)
            router = EpisodeReconstructionRouter((0, 1))
            result = trajectory._routed_policy_step(
                model, ac, router, torch.zeros(2, 3, 64, 64), z, h,
                torch.nn.functional.one_hot(torch.zeros(2, dtype=torch.long), 4),
                torch.ones(2, 1), stochastic=False,
            )
            outputs.append((result, router.events))
        for actual, expected in zip(outputs[0][0], outputs[1][0]):
            torch.testing.assert_close(actual, expected)
        self.assertEqual(outputs[0][1], outputs[1][1])


if __name__ == "__main__":
    unittest.main()
