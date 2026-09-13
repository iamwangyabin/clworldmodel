"""D-AutoRoute: private behavior ownership and label-free inference.

Fixed tensors and canned environment responses only; no training or simulator
interaction is launched by these tests.
"""

from __future__ import annotations

import copy
import importlib
import io
import json
import runpy
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from test_reconstruction_router import fixed_models, source_config, torch, vendor_available
if vendor_available:
    from test_reconstruction_router import Config, np, train, trajectory
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
    def test_host_control_preserves_full_config_and_only_caps_execution(self):
        launcher = importlib.import_module("run_evolving_atomic_rssm")
        entry = importlib.import_module("run_evolving_atomic_rssm_d_autoroute")
        with mock.patch.object(launcher, "_resolved_config", wraps=launcher._resolved_config) as resolved:
            output = io.StringIO()
            with redirect_stdout(output):
                entry.main(["--seed", "1", "--cpu-threads", "8", "--stop-after-first-task", "--dry-run"])
        manifest, _ = json.JSONDecoder().raw_decode(output.getvalue())
        self.assertEqual(manifest["seed"], 1337)
        self.assertTrue(manifest["protocol"].endswith("-FirstTask90-HostControl-v1"))
        self.assertEqual(manifest["parent_protocol"], entry.PROTOCOL)
        self.assertEqual(manifest["configured_full_curriculum_budgets"]["total_world_model_optimizer_steps"], 552000)
        self.assertIn("--stop-after-first-task", manifest["command"])
        self.assertNotIn("--evaluate-final", manifest["command"])
        self.assertEqual(resolved.call_args.kwargs["task_order"], "arrow-original-six")
        budget = manifest["budgets"]
        self.assertEqual(budget["total_world_model_optimizer_steps"], 92000)
        self.assertEqual(budget["actor_critic_updates"], 72000)
        self.assertEqual(budget["raw_environment_frames"], 5898240)
        self.assertEqual(budget["adaptive_compression_validation_rollouts"], 80)
        self.assertEqual(budget["online_memory_sequences"], 0)
        self.assertEqual(budget["boundary_validation_rollouts"], 16)
        self.assertEqual(budget["replay"], launcher._budget_manifest(new_config())["replay"])
        with self.assertRaisesRegex(ValueError, "fresh"):
            entry.main(["--stop-after-first-task", "--resume-from", "/unused"])

    def test_awm_names_preserve_method_identifiers_and_version_routing_protocol(self):
        launcher = importlib.import_module("run_evolving_atomic_rssm")
        for profile, name, method, protocol in (
            (
                "private_mlp", "AWM (Accumulative World Modeling)",
                "evolving_atomic_rssm_adaptive_compression_shared_heads_arrow",
                launcher.ADAPTIVE_QFP_COMPRESSION_PROTOCOL,
            ),
            (
                BEHAVIOR,
                "AWM-AutoRoute",
                METHOD, launcher.D_AUTOROUTE_PROTOCOL,
            ),
        ):
            with self.subTest(profile=profile):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(launcher.main([
                        "--dry-run", "--behavior-profile", profile,
                    ]), 0)
                manifest, _ = json.JSONDecoder().raw_decode(output.getvalue())
                self.assertEqual(manifest["method"], name)
                self.assertEqual(manifest["protocol"], protocol)
                self.assertEqual(
                    manifest["adaptive_compression_protocol"]["validation_scope"],
                    "completed task with oracle routing",
                )
                config = launcher._resolved_config(source_config(), behavior_profile=profile)
                self.assertEqual(config["continual_method"], method)

    def test_only_inference_changes_from_d_and_budgets_are_explicit(self):
        data = new_config()
        old = _resolved_config(
            source_config(), task_order="arrow-original-six",
            prediction_head_profile="shared_distilled", adaptive_qfp_compression=True,
        )
        changed = {k for k in data.keys() | old.keys() if data.get(k) != old.get(k)}
        self.assertEqual(changed, {
            "continual_method", "task_route_inference", "task_route_inference_version",
        })
        self.assertEqual(data["continual_method"], METHOD)
        self.assertTrue(data["task_private_actor_critic"])
        self.assertEqual(data["actor_network"], "mlp")
        self.assertNotIn("adaptive_behavior_residuals", data)
        budget = _budget_manifest(data)
        self.assertEqual(budget["total_world_model_optimizer_steps"], 552_000)
        self.assertEqual(budget["actor_critic_updates"], 432_000)
        self.assertEqual(budget["adaptive_compression_validation_rollouts"], 480)
        self.assertEqual(budget["adaptive_compression_validation_scope"], "current_task_oracle")
        self.assertEqual(budget["evaluation_episode_count_mode"], "legacy")
        self.assertEqual(budget["adaptive_behavior_compression_updates"], 0)
        parameters = _parameter_manifest(data)
        self.assertEqual(parameters["online_parameters"], 52_897_535)
        self.assertEqual(parameters["behavior_parameters"], 10_295_910)
        self.assertEqual(parameters["per_later_task_behavior_growth"], 1_715_985)
        self.assertEqual(
            parameters["adaptive_compression"]["selection_metric"],
            "current-task oracle raw episodic return",
        )

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
        self.assertEqual(launch["inference_routing"]["mode"], "two_frame_probability_reconstruction")
        self.assertEqual(launch["inference_routing"]["maximum_scored_observations_per_episode"], 2)
        self.assertIn("-v4-", launch["protocol"])
        self.assertEqual(launch["inference_routing"]["protocol_version"], 4)
        self.assertEqual(launch["parameter_budget"]["behavior_parameters"], 10_295_910)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            entry._parser().parse_args(["--behavior-profile", "shared_fastkan_autoroute"])

    def test_invalid_compositions_are_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            _resolved_config(source_config(), task_order="arrow-original-six", behavior_profile=BEHAVIOR, adaptive_qfp_compression=False)
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
    def test_host_control_boundary_predicate_does_not_touch_rng_or_stop_early(self):
        boundary = {"boundary_index": 1, "task_index": 0}
        rng = torch.random.get_rng_state().clone()
        for epoch in range(90):
            self.assertFalse(train._first_task_control_boundary(False, boundary, epoch, 0, 0))
            self.assertFalse(train._first_task_control_boundary(True, None, epoch, 0, 0))
        self.assertTrue(train._first_task_control_boundary(True, boundary, 90, 92000, 72000))
        with self.assertRaises(RuntimeError):
            train._first_task_control_boundary(True, boundary, 90, 90000, 72000)
        torch.testing.assert_close(rng, torch.random.get_rng_state(), rtol=0, atol=0)

    def test_private_mlp_actor_updates_without_retired_consolidation_interface(self):
        from ac import ActorCriticTrainingStep, build_actor_critic_opt, train_ac_from_wm
        from replay import FifoReplay
        from retained_method_support import retained_world_model

        wm = retained_world_model()
        wm.activate_task_expert(0)
        data = FifoReplay(4, 4, wm.a_dim, "cpu", store_task_ids=True,
                          observation_dtype="uint8")
        actions = torch.nn.functional.one_hot(torch.zeros(4, 4, dtype=torch.long), wm.a_dim).float()
        data.add(actions, torch.zeros(4, 4, 3, 64, 64), torch.zeros(4, 4, 1),
                 torch.ones(4, 4, 1), torch.zeros(4, 4, 1), task_id=0)
        aco = build_actor_critic_opt(wm, lr=1e-4)
        returned, _, metrics = train_ac_from_wm(
            wm, data, 1, n_sync=2, dream_steps=2, aco=aco, task_id=0,
        )
        self.assertIs(returned, aco)
        self.assertEqual(metrics["kan_consolidation_loss"], 0.)
        self.assertEqual({int(s["step"].item()) for s in aco.opt.state.values()}, {1})
        step = ActorCriticTrainingStep(aco.ac, entropy_scale=.0003,
                                      replay_critic_loss_scale=0., slow_critic_regularizer=0.)
        values = step(torch.zeros(2, 2, wm.zh_transform.out_features), actions[:2, :2],
                      torch.zeros(2, 2, 1), torch.tensor(1.), None, None, None, None, None)
        self.assertTrue(all(bool(torch.isfinite(v).all()) for v in values))
        self.assertEqual(values[-1].item(), 0.)

    def test_parameter_accounting_covers_dense_and_compact_retained_models(self):
        from retained_method_support import retained_world_model

        wm = retained_world_model()
        for compact in (False, True):
            if compact:
                train._structured_adaptive_qfp_candidate(
                    wm=wm, dense_teacher=copy.deepcopy(wm), task_id=1, fraction=.5,
                )
            report = train._world_model_parameter_accounting(wm)
            json.dumps(report)
            self.assertEqual(report["world_model"]["parameters"],
                             sum(p.numel() for p in wm.parameters()))
            self.assertEqual(report["prediction_adapter_parameters_per_task"],
                             {str(k): 0 for k in range(3)})
            self.assertEqual(
                sum(report["rssm_task_mechanism_parameters_per_later_task"].values()),
                sum(p.numel() for bank in wm.rssm.mechanism_banks().values()
                    for p in bank.parameters()),
            )

    def test_real_trainer_cli_preserves_config_before_cuda_initialization(self):
        from clworldmodel.distributed import DistributedContext

        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(new_config()))
            for extra in ([], ["--actor-network", "mlp"]):
                with self.subTest(extra=extra), mock.patch.object(
                    sys, "argv", [train.__file__, "--config", str(path), *extra]
                ), mock.patch.object(
                    DistributedContext, "initialize",
                    side_effect=RuntimeError("stop before CUDA initialization"),
                ), self.assertRaisesRegex(RuntimeError, "stop before CUDA initialization"):
                    runpy.run_path(train.__file__, run_name="__main__")

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
        # There is one maintained router; historical configs must fail closed.
        with self.assertRaises(ValueError):
            Config.from_dict({**data, "task_route_inference": "first_frame_reconstruction"})
        historical = dict(data)
        historical.pop("task_route_inference_version")
        with self.assertRaisesRegex(ValueError, "inference version"):
            Config.from_dict(historical)
        for version in (3, 4.0, True):
            with self.assertRaisesRegex(ValueError, "inference version"):
                Config.from_dict({**data, "task_route_inference_version": version})
        self.assertNotIn("adaptive_behavior_residuals", config.to_dict())
        for key, value in {
            "task_private_actor_critic": False, "actor_network": "fast_kan_ac_stable",
            "task_route_inference": "oracle", "evaluation_episode_count_mode": "exact",
            "task_mechanism_parameterization": "shared_frozen_down_film",
            "evolving_shared_behavior_current_task_fraction": .75,
        }.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                Config.from_dict({**data, key: value})
        with self.assertRaises(TypeError):
            Config.from_dict({**data, "unknown_router_option": True})
        with self.assertRaises(ValueError):
            Config.from_dict({**data, "task_route_inference": "continuous_reconstruction"})

    def test_gpu_smoke_profile_resolves_without_running_updates(self):
        smoke = importlib.import_module("smoke_evolving_atomic_rssm")
        config = smoke._config(method_profile=smoke.D_AUTOROUTE_PROFILE)
        self.assertEqual(config.continual_method, METHOD)
        self.assertTrue(config.task_private_actor_critic)

    def test_per_worker_inferred_id_selects_private_actor_not_current_actor(self):
        from clworldmodel.routing import TwoFrameReconstructionRouter
        wm, _ = fixed_models()
        view, actors = self.actors()
        before = {id(p) for actor in actors.values() for p in actor.parameters()}
        self.assertEqual({id(p) for p in view.parameters()}, before)
        router = TwoFrameReconstructionRouter((0, 1))
        z, h = wm.rssm.initial_state(2)
        x = torch.tensor([0., 1.])[:, None, None, None].expand(2, 3, 2, 2)
        previous = torch.nn.functional.one_hot(torch.tensor([5, 6]), 18)
        for frames, reset, expected in (
            (x, torch.ones(2, 1), [7, 11]),
            (1 - x, torch.zeros(2, 1), [7, 7]),
            (1 - x, torch.tensor([[0.], [1.]]), [7, 7]),
        ):
            z, h, action = trajectory._routed_policy_step(
                wm, view, router, frames, z, h, previous, reset, stochastic=False,
            )
            self.assertEqual(action.tolist(), expected)
        with self.assertRaisesRegex(ValueError, "eligib"):
            trajectory._routed_policy_step(wm, view, TwoFrameReconstructionRouter((0,)),
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
            self.assertEqual(kwargs["task_route_inference"], "two_frame_probability_reconstruction")
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

    def test_training_time_validation_keeps_d_oracle_routes(self):
        config = Config.from_dict(new_config())
        _, actors = self.actors()
        bank = mock.Mock()
        bank.get_optional.side_effect = lambda task: SimpleNamespace(
            ac=SimpleNamespace(actor=actors[task])
        )

        def evaluate(*args, **kwargs):
            self.assertNotIn("eligible_route_ids", kwargs)
            self.assertEqual(kwargs["task_id"], evaluate.calls)
            evaluate.calls += 1
            return 10., 0.

        evaluate.calls = 0
        with mock.patch.object(train, "evaluate", side_effect=evaluate) as evaluator:
            train._evaluate_policy_tasks(
                config, mock.sentinel.wm, mock.sentinel.current_aco,
                [[mock.sentinel.env]] * 2, (10, 11), actor_critic_bank=bank,
                oracle_routes=True,
            )
        self.assertEqual(evaluator.call_count, 2)

    def test_compression_validation_matches_d_oracle_current_task(self):
        config = Config.from_dict(new_config())
        _, actors = self.actors()
        bank = mock.Mock()
        bank.get.side_effect = lambda task: SimpleNamespace(ac=SimpleNamespace(actor=actors[task]))
        with mock.patch.object(train, "evaluate", return_value=(10., 0.)) as evaluator:
            train._evaluate_adaptive_compression_task(
                config=config, wm=mock.sentinel.wm, actor_critic_bank=bank,
                task_id=0, eval_env_fns=[mock.sentinel.env], validation_seed=123,
            )
        kwargs = evaluator.call_args.kwargs
        self.assertEqual(kwargs["task_id"], 0)
        self.assertIs(kwargs["ac"].actor, actors[0])
        self.assertEqual(bank.get.call_args.args, (0,))

    def test_private_collection_uses_arrow_next_step_and_episode_locks(self):
        wm, _ = fixed_models()
        view, _ = self.actors()
        fixture = mock.Mock()
        def pixels(values):
            return np.array(values, np.uint8)[:, None, None, None] * np.ones((2, 2, 2, 3), np.uint8)
        fixture.reset.return_value = (pixels([0, 255]), {})
        fixture.step.side_effect = [
            (pixels([255, 0]), np.ones(2), np.array([False, False]), np.zeros(2, bool), {}),
            (pixels([255, 0]), np.ones(2), np.array([True, False]), np.zeros(2, bool), {}),
            (pixels([255, 0]), np.zeros(2), np.array([False, False]), np.zeros(2, bool), {}),
            (pixels([255, 0]), np.ones(2), np.array([False, False]), np.zeros(2, bool), {}),
        ]
        diagnostic = {}
        with mock.patch.object(trajectory, "AsyncVectorEnv", return_value=fixture) as constructor:
            trajectory.generate_trajectories(
                10, 2, wm=wm, ac=view, env_fns=[mock.sentinel.env] * 2,
                eligible_route_ids=(0, 1), seed=4, deterministic_policy=True,
                routing_diagnostics=diagnostic,
            )
        self.assertNotIn("autoreset_mode", constructor.call_args.kwargs)
        self.assertEqual([c.args[0].tolist() for c in fixture.step.call_args_list], [[7, 11], [7, 7], [7, 7], [11, 7]])
        self.assertEqual([e["selected_route_id"] for e in diagnostic["routing_events"]], [0, 1, 0, 0, 1])
        fixture.close.assert_called_once()

    def test_constant_policy_collection_preserves_d_storage_contract(self):
        from clworldmodel.routing import RoutedActorBank

        wm, _ = fixed_models()
        actor = torch.nn.Linear(3, 18)
        with torch.no_grad():
            actor.weight.zero_()
            actor.bias.zero_()
            actor.bias[7] = 10.
        routed = RoutedActorBank({0: actor})
        oracle = SimpleNamespace(actor=actor, set_task_route=lambda _: None)

        def pixels(values):
            return np.array(values, np.uint8)[:, None, None, None] * np.ones(
                (2, 2, 2, 3), np.uint8
            )

        responses = [
            (pixels([32, 64]), np.ones(2), np.array([False, False]), np.zeros(2, bool), {}),
            (pixels([96, 128]), np.ones(2), np.array([True, False]), np.zeros(2, bool), {}),
            (pixels([160, 192]), np.zeros(2), np.array([False, False]), np.zeros(2, bool), {}),
            (pixels([224, 16]), np.ones(2), np.array([False, False]), np.zeros(2, bool), {}),
        ]

        def environment():
            fixture = mock.Mock()
            fixture.reset.return_value = (pixels([0, 0]), {})
            fixture.step.side_effect = copy.deepcopy(responses)
            return fixture

        oracle_env, routed_env = environment(), environment()
        with mock.patch.object(trajectory, "AsyncVectorEnv", return_value=oracle_env):
            expected = trajectory.generate_trajectories(
                10, 2, wm=wm, ac=oracle, env_fns=[mock.sentinel.env] * 2,
                task_id=0, seed=4, deterministic_policy=True,
            )
        with mock.patch.object(trajectory, "AsyncVectorEnv", return_value=routed_env) as constructor:
            actual = trajectory.generate_trajectories(
                10, 2, wm=wm, ac=routed, env_fns=[mock.sentinel.env] * 2,
                eligible_route_ids=(0,), seed=4, deterministic_policy=True,
            )
        self.assertNotIn("autoreset_mode", constructor.call_args.kwargs)
        for expected_tensor, actual_tensor in zip(expected, actual):
            torch.testing.assert_close(actual_tensor, expected_tensor)
        self.assertEqual(
            [call.args[0].tolist() for call in routed_env.step.call_args_list],
            [call.args[0].tolist() for call in oracle_env.step.call_args_list],
        )

    def test_autoroute_evaluation_uses_d_legacy_trajectory_budget(self):
        wm, _ = fixed_models()
        view, _ = self.actors()
        actions = torch.nn.functional.one_hot(torch.zeros(6, dtype=torch.long), 18).float()
        rewards = torch.tensor([[0.], [2.], [0.], [0.], [4.], [0.]])
        continues = torch.tensor([[1.], [0.], [1.], [1.], [0.], [1.]])
        resets = torch.tensor([[1.], [0.], [1.], [1.], [0.], [1.]])
        diagnostic = {}
        with mock.patch.object(
            trajectory, "generate_trajectories",
            return_value=(actions, None, rewards, continues, resets),
        ) as generate:
            result = trajectory.evaluate(
                2, wm, view, [mock.sentinel.env] * 2, n_rollouts=5, seed=123,
                deterministic_policy=True, eligible_route_ids=(0, 1), diagnostics=diagnostic,
            )
        self.assertEqual(result, (3., 1.))
        self.assertEqual(generate.call_args.args[0], 5 * 2**13 // 2)
        self.assertEqual(generate.call_args.args[6], 5)
        self.assertEqual(generate.call_args.kwargs["eligible_route_ids"], (0, 1))
        self.assertEqual(diagnostic["episode_count_mode"], "legacy")

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
        from clworldmodel.routing import TwoFrameReconstructionRouter
        from retained_method_support import retained_world_model
        config = Config.from_dict(new_config())
        wm = retained_world_model("adaptive_dense_width", shared_prediction_heads=True)
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
            router = TwoFrameReconstructionRouter((0, 1))
            for t in range(3):
                z, h, action = trajectory._routed_policy_step(
                    wm, train._autorouted_behavior(config, None, bank, 2), router,
                    torch.full((2, 3, 64, 64), .1 * t), z, h,
                    torch.nn.functional.one_hot(torch.full((2,), t), wm.a_dim),
                    torch.full((2, 1), float(t == 0)), stochastic=False,
                )
            return z, h, action, router.events
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
            self.assertEqual(payload["inference_routing"]["mode"], "two_frame_probability_reconstruction")
            self.assertEqual(payload["inference_routing"]["protocol_version"], 4)
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
            old_payload = copy.deepcopy(payload)
            old_payload["config"].pop("task_route_inference_version")
            with mock.patch.object(train.torch, "load", return_value=old_payload):
                with self.assertRaisesRegex(ValueError, "Resolved config changed"):
                    train._restore_evolving_resumable_checkpoint(path, **common, actor_critic_factory=factory)
            old_payload = copy.deepcopy(payload)
            old_payload["config"]["task_route_inference"] = "first_frame_reconstruction"
            with mock.patch.object(train.torch, "load", return_value=old_payload):
                with self.assertRaisesRegex(ValueError, "Resolved config changed"):
                    train._restore_evolving_resumable_checkpoint(path, **common, actor_critic_factory=factory)
            payload["inference_routing"]["eligible_route_ids"] = [0, 1, 2]
            with mock.patch.object(train.torch, "load", return_value=payload):
                with self.assertRaisesRegex(ValueError, "eligibility"):
                    train._restore_evolving_resumable_checkpoint(path, **common, actor_critic_factory=factory)


if __name__ == "__main__":
    unittest.main()
