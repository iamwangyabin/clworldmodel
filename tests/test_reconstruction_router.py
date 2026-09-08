"""AWM-AutoRoute: task-aware training, task-ID-free inference.

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


def source_config() -> dict:
    folder = ROOT / "third_party/arrow/Configs/Atari configs/CL-task configs/Original Order"
    return json.loads(next(folder.glob("*Enduro-s0-arrow.json")).read_text())


try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    sys.path.insert(0, str(ROOT / "src"))
    from clworldmodel.routing import TwoFrameReconstructionRouter

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
class ReconstructionRouterIntegrationTests(unittest.TestCase):


    def test_collection_locks_each_worker_with_arrow_next_step_autoreset(self):
        wm, ac = fixed_models()
        fixture = mock.Mock()
        pixels = lambda values: np.array(values, np.uint8)[:, None, None, None] * np.ones((2, 2, 2, 3), np.uint8)
        fixture.reset.return_value = (pixels([0, 255]), {})
        fixture.step.side_effect = [
            (pixels([255, 0]), np.ones(2), np.array([False, False]), np.zeros(2, bool), {}),
            (pixels([255, 0]), np.ones(2), np.array([True, False]), np.zeros(2, bool), {}),
            (pixels([255, 0]), np.zeros(2), np.array([False, False]), np.zeros(2, bool), {}),
            (pixels([255, 0]), np.ones(2), np.array([False, False]), np.zeros(2, bool), {}),
        ]
        diagnostic = {}
        with mock.patch.object(trajectory, "AsyncVectorEnv", return_value=fixture) as constructor:
            actions, _, _, _, resets = trajectory.generate_trajectories(
                10, 2, wm=wm, ac=ac, env_fns=[mock.sentinel.env] * 2,
                eligible_route_ids=(0, 1), seed=4, deterministic_policy=True,
                routing_diagnostics=diagnostic,
            )
        self.assertNotIn("autoreset_mode", constructor.call_args.kwargs)
        # Opposite second frames tie the cumulative scores: lowest ID wins.
        self.assertEqual([call.args[0].tolist() for call in fixture.step.call_args_list], [[0, 1], [0, 0], [0, 0], [1, 0]])
        self.assertEqual([e["selected_route_id"] for e in diagnostic["routing_events"]], [0, 1, 0, 0, 1])
        self.assertEqual(resets[:4, 0].tolist(), [1., 0., 1., 0.])
        # Stored worker trajectory includes the initial dummy before all four
        # env.step actions. The reset episode's effective action is at index 4.
        self.assertEqual(actions[:5].argmax(-1).tolist(), [0, 0, 0, 0, 1])
        fixture.close.assert_called_once()


    def test_probe_dummy_does_not_replace_policy_previous_action(self):
        wm, ac = fixed_models()
        router = TwoFrameReconstructionRouter((0, 1))
        z, h = wm.rssm.initial_state(2)
        x = torch.tensor([0., 1.])[:, None, None, None].expand(2, 3, 2, 2)
        previous = torch.nn.functional.one_hot(torch.tensor([7, 8]), 18)
        rng = torch.random.get_rng_state().clone()
        with mock.patch.object(wm.rssm, "forward", wraps=wm.rssm.forward) as calls:
            z, h, action = trajectory._routed_policy_step(
                wm, ac, router, x, z, h, previous, torch.ones(2, 1), stochastic=False,
            )
            for probe in calls.call_args_list[:2]:
                self.assertEqual(probe.args[1].argmax(-1).tolist(), [0, 0])
        self.assertEqual(action.tolist(), [0, 1])
        self.assertEqual(wm.rssm.last_action.argmax(-1).tolist(), [8])
        z, h, action = trajectory._routed_policy_step(
            wm, ac, router, x, z, h, previous, torch.zeros(2, 1), stochastic=False,
        )
        self.assertEqual(action.tolist(), [0, 1])
        self.assertEqual(wm.rssm.last_action.argmax(-1).tolist(), [8])
        self.assertEqual(len(router.events), 4)
        z, h, action = trajectory._routed_policy_step(
            wm, ac, router, 1 - x, z, h, previous, torch.tensor([[0.], [1.]]), stochastic=False,
        )
        self.assertEqual(action.tolist(), [0, 0])
        torch.testing.assert_close(torch.random.get_rng_state(), rng)

    def test_autoroute_evaluation_keeps_arrow_trajectory_budget(self):
        wm, ac = fixed_models()
        actions = torch.nn.functional.one_hot(torch.zeros(6, dtype=torch.long), 18).float()
        rewards = torch.tensor([[0.], [2.], [0.], [0.], [4.], [0.]])
        continues = torch.tensor([[1.], [0.], [1.], [1.], [0.], [1.]])
        resets = torch.tensor([[1.], [0.], [1.], [1.], [0.], [1.]])
        diagnostics = {}
        with mock.patch.object(
            trajectory, "generate_trajectories",
            return_value=(actions, None, rewards, continues, resets),
        ) as generate:
            self.assertEqual(
                trajectory.evaluate(
                    2, wm, ac, [mock.sentinel.env_factory] * 2,
                    n_rollouts=5, seed=123, deterministic_policy=True,
                    eligible_route_ids=(0, 1), diagnostics=diagnostics,
                ),
                (3., 1.),
            )
        self.assertEqual(generate.call_args.args[0], 5 * 2**13 // 2)
        self.assertEqual(generate.call_args.args[6], 5)
        self.assertEqual(diagnostics["episode_count_mode"], "legacy")

    def test_eval_task_labels_only_reach_posthoc_audit_and_future_routes_excluded(self):
        config = Config.from_dict(_route_config())
        def evaluate(*args, **kwargs):
            self.assertNotIn("task_id", kwargs)
            self.assertEqual(kwargs["eligible_route_ids"], (0, 1))
            kwargs["diagnostics"]["routing_events"] = [{"selected_route_id": 0}]
            return 10., 0.
        diagnostics = []
        with mock.patch.object(train, "evaluate", side_effect=evaluate) as evaluator:
            train._evaluate_policy_tasks(
                config, mock.sentinel.wm, SimpleNamespace(ac=mock.sentinel.shared_ac),
                [[mock.sentinel.env]] * 6, tuple(range(6)), eligible_task_count=2, actor_critic_bank=SimpleNamespace(get=lambda task: SimpleNamespace(ac=SimpleNamespace(actor=torch.nn.Identity()))),
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

    def test_compact_shared_decoder_world_model_routes_after_strict_reload(self):
        from retained_method_support import retained_world_model
        from ac import build_actor_critic_opt
        wm = retained_world_model(
            "adaptive_dense_width", shared_prediction_heads=True,
        ).eval()
        teacher = copy.deepcopy(wm)
        from clworldmodel.routing import RoutedActorBank
        ac = RoutedActorBank({route: build_actor_critic_opt(wm, lr=1e-4).ac.actor.eval() for route in (0, 1)})
        train._structured_adaptive_qfp_candidate(wm=wm, dense_teacher=teacher, task_id=1, fraction=.5)
        restored = copy.deepcopy(teacher)
        restored.load_state_dict(wm.state_dict(), strict=True)
        self.assertEqual(restored.rssm.adaptive_compression_layout(), wm.rssm.adaptive_compression_layout())
        self.assertIs(wm.decoder_for(0), wm.decoder_for(1))
        outputs = []
        for model in (wm, restored):
            z, h = model.rssm.initial_state(2)
            router = TwoFrameReconstructionRouter((0, 1))
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


def _route_config():
    from run_evolving_atomic_rssm import _resolved_config
    return _resolved_config(source_config(), task_order="arrow-original-six", behavior_profile="private_mlp_autoroute")
