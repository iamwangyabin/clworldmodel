"""D-AutoRoute: private behavior ownership and label-free inference.

Fixed tensors and canned environment responses only; no training or simulator
interaction is launched by these tests.
"""

from __future__ import annotations

import copy
import importlib
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from test_fastkan_autoroute import fixed_models, source_config, torch, vendor_available
if vendor_available:
    from test_fastkan_autoroute import Config, np, train, trajectory
from run_evolving_atomic_rssm import (
    _budget_manifest, _parameter_manifest, _protocol_for_task_order, _resolved_config,
)


BEHAVIOR = "private_mlp_autoroute"
METHOD = "evolving_atomic_rssm_adaptive_compression_shared_heads_autoroute_arrow"


def new_config():
    return _resolved_config(
        source_config(), task_order="arrow-original-six",
        prediction_head_profile="shared_distilled", adaptive_qfp_compression=True,
        behavior_profile=BEHAVIOR,
    )


class DAutorouteLauncherTests(unittest.TestCase):
    def test_only_inference_changes_from_d_and_budgets_are_explicit(self):
        data = new_config()
        old = _resolved_config(
            source_config(), task_order="arrow-original-six",
            prediction_head_profile="shared_distilled", adaptive_qfp_compression=True,
        )
        changed = {k for k in data.keys() | old.keys() if data.get(k) != old.get(k)}
        self.assertEqual(changed, {
            "continual_method", "task_route_inference", "evaluation_episode_count_mode",
            "evaluation_max_agent_decisions_per_episode",
        })
        self.assertEqual(data["continual_method"], METHOD)
        self.assertTrue(data["task_private_actor_critic"])
        self.assertEqual(data["actor_network"], "mlp")
        self.assertFalse(data["adaptive_behavior_residuals"])
        budget = _budget_manifest(data)
        self.assertEqual(budget["total_world_model_optimizer_steps"], 552_000)
        self.assertEqual(budget["actor_critic_updates"], 432_000)
        self.assertEqual(budget["adaptive_compression_validation_rollouts"], 1680)
        self.assertEqual(budget["adaptive_behavior_compression_updates"], 0)
        parameters = _parameter_manifest(data)
        self.assertEqual(parameters["online_parameters"], 52_897_535)
        self.assertEqual(parameters["behavior_parameters"], 10_295_910)
        self.assertEqual(parameters["per_later_task_behavior_growth"], 1_715_985)
        self.assertIn("auto-routed", parameters["adaptive_compression"]["selection_metric"])

    def test_independent_entrypoint_dry_run_and_fixed_architecture(self):
        entry = importlib.import_module("run_evolving_atomic_rssm_d_autoroute")
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(entry.main(["--dry-run", "--seed", "0"]), 0)
        launch, _ = json.JSONDecoder().raw_decode(output.getvalue())
        self.assertEqual(launch["protocol"], entry.PROTOCOL)
        self.assertEqual(launch["behavior_profile"], BEHAVIOR)
        self.assertFalse(launch["task_identity_exposed_during_action_selection"])
        self.assertTrue(launch["task_identity_exposed_during_training"])
        self.assertEqual(launch["inference_routing"]["mode"], "first_frame_reconstruction")
        self.assertEqual(launch["parameter_budget"]["behavior_parameters"], 10_295_910)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            entry._parser().parse_args(["--behavior-profile", "shared_fastkan_autoroute"])

    def test_invalid_compositions_are_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            _resolved_config(source_config(), task_order="arrow-original-six", behavior_profile=BEHAVIOR)
        with self.assertRaises(ValueError):
            _protocol_for_task_order("mspacman-boxing-crazyclimber", behavior_profile=BEHAVIOR,
                                     adaptive_qfp_compression=True, prediction_head_profile="shared_distilled")

    def test_relative_paths_are_repository_rooted_and_launch_is_delegated(self):
        entry = importlib.import_module("run_evolving_atomic_rssm_d_autoroute")
        with mock.patch.object(entry, "_launch", return_value=0) as launch:
            entry.main(["--output-dir", "runs/d-route-fixture", "--cpu-threads", "3", "--dry-run"])
        command = launch.call_args.args[0]
        self.assertEqual(command[command.index("--output-dir") + 1], str(entry.ROOT / "runs/d-route-fixture"))
        self.assertEqual(command[command.index("--cpu-threads") + 1], "3")
        self.assertIn("--dry-run", command)

    def test_standalone_entrypoint_cannot_bypass_clean_pushed_git_guard(self):
        entry = importlib.import_module("run_evolving_atomic_rssm_d_autoroute")
        launcher = importlib.import_module("run_evolving_atomic_rssm")
        with mock.patch.object(launcher, "require_synced_training_git_state", side_effect=RuntimeError("unsynced")):
            with mock.patch.object(launcher, "_run_and_tee") as run:
                with self.assertRaisesRegex(RuntimeError, "unsynced"):
                    entry.main([])
        run.assert_not_called()


@unittest.skipUnless(vendor_available, "requires pinned Atari imports, no ROMs")
class DAutorouteIntegrationTests(unittest.TestCase):
    def actors(self):
        from clworldmodel.routing import RoutedActorBank
        actors = {}
        for route, action in ((0, 7), (1, 11)):
            actor = torch.nn.Linear(3, 18)
            with torch.no_grad():
                actor.weight.zero_()
                actor.bias.zero_()
                actor.bias[action] = 10.
            actors[route] = actor
        return RoutedActorBank(actors), actors

    def test_typed_config_keeps_d_ownership_and_rejects_leaks(self):
        data = new_config()
        config = Config.from_dict(data)
        self.assertEqual(Config.from_dict(config.to_dict()).to_dict(), config.to_dict())
        self.assertTrue(config.uses_reconstruction_task_inference)
        self.assertTrue(config.uses_task_labelled_replay)
        self.assertTrue(config.uses_evolving_atomic_rssm)
        self.assertTrue(config.uses_adaptive_qfp_compression)
        self.assertTrue(config.uses_shared_prediction_heads)
        self.assertFalse(config.uses_shared_actor)
        self.assertFalse(config.uses_replay_rehearsed_shared_behavior)
        self.assertFalse(config.uses_adaptive_behavior_compression)
        for key, value in {
            "task_private_actor_critic": False, "actor_network": "fast_kan_ac_stable",
            "task_route_inference": "oracle", "evaluation_episode_count_mode": "legacy",
            "task_mechanism_parameterization": "shared_frozen_down_film",
            "evolving_shared_behavior_current_task_fraction": .75,
        }.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                Config.from_dict({**data, key: value})
        with self.assertRaises(TypeError):
            Config.from_dict({**data, "unknown_router_option": True})

    def test_gpu_smoke_profile_resolves_without_running_updates(self):
        smoke = importlib.import_module("smoke_evolving_atomic_rssm")
        config = smoke._config(method_profile=smoke.D_AUTOROUTE_PROFILE)
        self.assertEqual(config.continual_method, METHOD)
        self.assertTrue(config.task_private_actor_critic)

    def test_per_worker_inferred_id_selects_private_actor_not_current_actor(self):
        from clworldmodel.routing import EpisodeReconstructionRouter
        wm, _ = fixed_models()
        view, actors = self.actors()
        before = {id(p) for actor in actors.values() for p in actor.parameters()}
        self.assertEqual({id(p) for p in view.parameters()}, before)
        router = EpisodeReconstructionRouter((0, 1))
        z, h = wm.rssm.initial_state(2)
        x = torch.tensor([0., 1.])[:, None, None, None].expand(2, 3, 2, 2)
        previous = torch.nn.functional.one_hot(torch.tensor([5, 6]), 18)
        for frames, reset, expected in (
            (x, torch.ones(2, 1), [7, 11]),
            (1 - x, torch.zeros(2, 1), [7, 11]),
            (1 - x, torch.tensor([[0.], [1.]]), [7, 7]),
        ):
            z, h, action = trajectory._routed_policy_step(
                wm, view, router, frames, z, h, previous, reset, stochastic=False,
            )
            self.assertEqual(action.tolist(), expected)
        with self.assertRaisesRegex(ValueError, "eligib"):
            trajectory._routed_policy_step(wm, view, EpisodeReconstructionRouter((0,)),
                                          x, z, h, previous, torch.ones(2, 1), stochastic=False)

    def test_private_actor_view_validates_shapes_eligibility_and_finiteness(self):
        from clworldmodel.routing import RoutedActorBank
        view, actors = self.actors()
        with self.assertRaises(ValueError):
            RoutedActorBank({})
        for routes in (torch.tensor([0., 1.]), torch.tensor([0]), torch.tensor([0, 2])):
            with self.subTest(routes=routes), self.assertRaises(ValueError):
                view(torch.zeros(2, 3), routes)
        with torch.no_grad():
            actors[0].bias.fill_(float("nan"))
        with self.assertRaises(FloatingPointError):
            view(torch.zeros(2, 3), torch.tensor([0, 1]))

    def test_all_evaluation_tasks_receive_same_acquired_actor_view_not_oracle(self):
        from clworldmodel.routing import RoutedActorBank
        config = Config.from_dict(new_config())
        _, actors = self.actors()
        bank = mock.Mock()
        bank.get.side_effect = lambda task: SimpleNamespace(ac=SimpleNamespace(actor=actors[task]))
        diagnostics = []
        def evaluate(*args, **kwargs):
            self.assertNotIn("task_id", kwargs)
            self.assertIsInstance(kwargs["ac"], RoutedActorBank)
            self.assertEqual(kwargs["ac"].route_ids, (0, 1))
            self.assertEqual(kwargs["eligible_route_ids"], (0, 1))
            kwargs["diagnostics"]["routing_events"] = [{"selected_route_id": 1}]
            return 10., 0.
        with mock.patch.object(train, "evaluate", side_effect=evaluate) as evaluator:
            train._evaluate_policy_tasks(config, mock.sentinel.wm, mock.sentinel.current_aco,
                                         [[mock.sentinel.env]] * 6, tuple(range(6)),
                                         actor_critic_bank=bank, eligible_task_count=2,
                                         routing_diagnostics=diagnostics)
        self.assertEqual(evaluator.call_count, 6)
        self.assertEqual([c.args[0] for c in bank.get.call_args_list], [0, 1])
        self.assertEqual([d["true_task_is_eligible"] for d in diagnostics], [True, True, False, False, False, False])

    def test_compression_validation_receives_all_eligible_private_actors(self):
        config = Config.from_dict(new_config())
        _, actors = self.actors()
        bank = mock.Mock()
        bank.get.side_effect = lambda task: SimpleNamespace(ac=SimpleNamespace(actor=actors[task]))
        with mock.patch.object(train, "evaluate", return_value=(10., 0.)) as evaluator:
            train._evaluate_adaptive_compression_task(
                config=config, wm=mock.sentinel.wm, actor_critic_bank=bank,
                task_id=0, eval_env_fns=[mock.sentinel.env], validation_seed=123,
                eligible_task_count=2,
            )
        kwargs = evaluator.call_args.kwargs
        self.assertNotIn("task_id", kwargs)
        self.assertEqual(kwargs["ac"].route_ids, (0, 1))

    def test_private_collection_uses_episode_locks_and_resets(self):
        wm, _ = fixed_models()
        view, _ = self.actors()
        fixture = mock.Mock()
        def pixels(values):
            return np.array(values, np.uint8)[:, None, None, None] * np.ones((2, 2, 2, 3), np.uint8)
        fixture.reset.return_value = (pixels([0, 255]), {})
        fixture.step.side_effect = [
            (pixels([255, 0]), np.ones(2), np.array([False, False]), np.zeros(2, bool), {}),
            (pixels([255, 0]), np.ones(2), np.array([True, False]), np.zeros(2, bool), {}),
            (pixels([255, 0]), np.ones(2), np.array([False, False]), np.zeros(2, bool), {}),
        ]
        diagnostic = {}
        with mock.patch.object(trajectory, "AsyncVectorEnv", return_value=fixture) as constructor:
            trajectory.generate_trajectories(
                8, 2, wm=wm, ac=view, env_fns=[mock.sentinel.env] * 2,
                eligible_route_ids=(0, 1), seed=4, deterministic_policy=True,
                routing_diagnostics=diagnostic,
            )
        self.assertEqual(constructor.call_args.kwargs["autoreset_mode"], "SameStep")
        self.assertEqual([c.args[0].tolist() for c in fixture.step.call_args_list], [[7, 11], [7, 11], [11, 11]])
        self.assertEqual([e["selected_route_id"] for e in diagnostic["routing_events"]], [0, 1, 1])
        fixture.close.assert_called_once()

    def test_exact_private_evaluation_restores_modes_weights_rng_and_no_extra_episodes(self):
        wm, _ = fixed_models()
        view, actors = self.actors()
        actors[0].eval().requires_grad_(False)
        actors[1].train().requires_grad_(True)
        before = copy.deepcopy(view.state_dict())
        rng = torch.random.get_rng_state().clone()
        fixtures = []
        def factory(*_):
            fixture = mock.Mock()
            state = {}
            def reset(*, seed):
                state["route"] = seed % 2
                return np.full((2, 2, 3), 255 * state["route"], np.uint8), {}
            def step(action):
                self.assertEqual(action, (7, 11)[state["route"]])
                return np.zeros((2, 2, 3), np.uint8), 3., True, False, {}
            fixture.reset.side_effect, fixture.step.side_effect = reset, step
            fixtures.append(fixture)
            return fixture
        diagnostic = {}
        with mock.patch.object(trajectory, "_make_atari_env", side_effect=factory):
            result = trajectory.evaluate(
                2, wm, view, [mock.sentinel.env] * 2, n_rollouts=5, seed=123,
                deterministic_policy=True, eligible_route_ids=(0, 1), diagnostics=diagnostic,
            )
        self.assertEqual(result, (3., 0.))
        self.assertEqual(sum(e.step.call_count for e in fixtures), 5)
        self.assertEqual(diagnostic["completed_episodes"], 5)
        self.assertTrue(all(e.close.call_count == 1 for e in fixtures))
        self.assertFalse(actors[0].training)
        self.assertTrue(actors[1].training)
        self.assertFalse(next(actors[0].parameters()).requires_grad)
        self.assertTrue(next(actors[1].parameters()).requires_grad)
        torch.testing.assert_close(view.state_dict(), before)
        torch.testing.assert_close(torch.random.get_rng_state(), rng)

    def test_real_independent_mlp_counts_and_old_parameters_remain_frozen(self):
        from ac import build_actor_critic_opt
        from clworldmodel.continual import ActorCriticBank
        config = Config.from_dict(new_config())
        wm = torch.nn.Linear(1, 1)
        wm.ls, wm.h_dim, wm.a_dim = (32, 32), 512, 18
        bank = ActorCriticBank()
        for task in range(2):
            bank.ensure(task, lambda _: build_actor_critic_opt(
                wm, lr=config.ac_lr, **train._actor_critic_constructor_kwargs(config),
            ))
        bank.activate(1)
        self.assertEqual(sum(p.numel() for p in bank.get(0).ac.parameters()), 1_715_985)
        self.assertFalse({id(p) for p in bank.get(0).ac.parameters()} & {id(p) for p in bank.get(1).ac.parameters()})
        view = train._autorouted_behavior(config, bank.get(1), bank, 2)
        self.assertIs(view.actors["0"], bank.get(0).ac.actor)
        self.assertTrue(all(not p.requires_grad for p in bank.get(0).ac.parameters()))
        self.assertTrue(all(p.requires_grad for p in bank.get(1).ac.parameters()))

    def test_compact_checkpoint_restores_private_bank_eligibility_and_rng(self):
        from ac import build_actor_critic_opt
        from clworldmodel.continual import ActorCriticBank
        from clworldmodel.routing import EpisodeReconstructionRouter
        from test_evolving_atomic_rssm import EvolvingAtomicRssmTests
        config = Config.from_dict(new_config())
        wm = EvolvingAtomicRssmTests._world_model("adaptive_dense_width", shared_prediction_heads=True)
        dense = copy.deepcopy(wm)
        train._structured_adaptive_qfp_candidate(wm=wm, dense_teacher=dense, task_id=1, fraction=.5)
        teacher = copy.deepcopy(wm)
        factory = lambda _: build_actor_critic_opt(wm, lr=config.ac_lr, **train._actor_critic_constructor_kwargs(config))
        bank = ActorCriticBank()
        for task in range(2):
            bank.ensure(task, factory)
        bank.activate(1)
        expected = copy.deepcopy(bank.resumable_state_dict())
        optimizer = torch.optim.Adam([next(wm.parameters())], lr=config.shared_core_lr)
        generators = [np.random.default_rng(i) for i in range(4)]
        replay = mock.Mock()
        replay.state_dict.return_value = {"fixture": "no transitions"}
        schedule = SimpleNamespace(_step=179)
        common = dict(
            config=config, wm=wm, boundary_teacher=teacher, shared_optimizer=optimizer,
            private_optimizers={}, route_optimizers={}, actor_critic_bank=bank,
            replay_buffer=replay, environment_schedule=schedule, task_update_rng=generators[0],
            collection_environment_seed_rng=generators[1], validation_environment_seed_rng=generators[2],
            final_environment_seed_rng=generators[3],
        )
        def inference():
            z, h = wm.rssm.initial_state(2)
            return trajectory._routed_policy_step(
                wm, train._autorouted_behavior(config, None, bank, 2), EpisodeReconstructionRouter((0, 1)),
                torch.zeros(2, 3, 64, 64), z, h,
                torch.nn.functional.one_hot(torch.zeros(2, dtype=torch.long), wm.a_dim),
                torch.ones(2, 1), stochastic=False,
            )
        expected_inference = inference()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "boundary.pt"
            train._save_evolving_resumable_checkpoint(
                path, **common, epoch=179, current_task_id=1, world_model_updates=0,
                actor_critic_updates=0, total_env_steps=0,
            )
            payload = torch.load(path, weights_only=False)
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["inference_routing"]["eligible_route_ids"], [0, 1])
            draws = [g.integers(0, 100000) for g in generators]
            wm.load_state_dict(dense.state_dict(), strict=True)
            with torch.no_grad():
                next(bank.get(0).ac.actor.parameters()).add_(1.)
            train._restore_evolving_resumable_checkpoint(path, **common, actor_critic_factory=factory)
            self.assertEqual(wm.rssm.adaptive_compression_layout(), teacher.rssm.adaptive_compression_layout())
            restored_bank = bank.resumable_state_dict()
            self.assertEqual(restored_bank["artifact_kind"], expected["artifact_kind"])
            torch.testing.assert_close(restored_bank["tasks"], expected["tasks"])
            torch.testing.assert_close(inference(), expected_inference)
            self.assertEqual([g.integers(0, 100000) for g in generators], draws)
            self.assertEqual(schedule._step, 180)
            replay.load_state_dict.assert_called_once_with({"fixture": "no transitions"})
            payload["inference_routing"]["eligible_route_ids"] = [0, 1, 2]
            with mock.patch.object(train.torch, "load", return_value=payload):
                with self.assertRaisesRegex(ValueError, "eligibility"):
                    train._restore_evolving_resumable_checkpoint(path, **common, actor_critic_factory=factory)


if __name__ == "__main__":
    unittest.main()
