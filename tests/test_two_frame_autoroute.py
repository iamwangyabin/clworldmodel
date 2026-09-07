"""AWM-AutoRoute v3: fixed tensors only, no simulator or training."""

import unittest
import copy
from unittest import mock

from test_reconstruction_router import torch, vendor_available

if torch is not None:
    from clworldmodel.routing import TwoFrameReconstructionRouter, routing_audit
if vendor_available:
    from test_reconstruction_router import trajectory


@unittest.skipIf(torch is None, "requires PyTorch")
class TwoFrameRouterTests(unittest.TestCase):
    def test_only_one_router_is_exposed_and_single_candidate_ties_are_finite(self):
        import clworldmodel.routing as routing
        self.assertFalse(hasattr(routing, "EpisodeReconstructionRouter"))
        for candidates in ((0,), (0, 2)):
            router = TwoFrameReconstructionRouter(candidates)
            x = torch.zeros(1, 1, 1, 1)
            for reset in (True, False):
                self.assertEqual(router.route(x, torch.tensor([reset]),
                                              lambda k, obs, rows, first: obs).item(), 0)
                self.assertEqual(router.events[-1]["margin"], 0.)

    def test_cumulative_choice_per_worker_reset_and_no_later_probes(self):
        router = TwoFrameReconstructionRouter((0, 1))
        calls = []

        def decode(k, frames, rows, first):
            calls.append((k, rows.tolist(), first.tolist()))
            return torch.full_like(frames, .2 if k == 0 else .8)

        def step(values, resets):
            return router.route(torch.tensor(values).reshape(-1, 1, 1, 1),
                                torch.tensor(resets), decode).tolist()

        self.assertEqual(step([.45, .8], [True, True]), [0, 1])
        self.assertEqual(step([.9, .2], [False, True]), [1, 0])
        self.assertEqual(step([.0, .2], [False, False]), [1, 0])
        self.assertEqual(calls[-2:], [(0, [1], [False]), (1, [1], [False])])
        count = len(calls)
        self.assertEqual(step([.0, 1.], [False, False]), [1, 0])
        self.assertEqual(len(calls), count)
        self.assertEqual(router.counts.tolist(), [2, 2])
        audit = routing_audit(router.events, true_task_id=1, task_count=2)
        self.assertEqual(audit["episode_starts"], 3)
        self.assertEqual(audit["routing_decisions"], 5)
        self.assertEqual(audit["accuracy"], 2 / 3)  # last available decision / episode

    def test_cumulative_not_latest_and_ties_lowest(self):
        router = TwoFrameReconstructionRouter((0, 2))
        decode = lambda k, frames, rows, first: torch.full_like(frames, k / 2)
        x = torch.zeros(1, 1, 1, 1)
        self.assertEqual(router.route(x, torch.tensor([True]), decode).item(), 0)
        self.assertEqual(router.route(x + .6, torch.tensor([False]), decode).item(), 0)
        self.assertEqual(router.route(x + .5, torch.tensor([True]), decode).item(), 0)

    def test_invalid_and_nonfinite_inputs(self):
        for ids in [(), (1, 0), (0, 0), (True,)]:
            with self.assertRaises(ValueError):
                TwoFrameReconstructionRouter(ids)
        router = TwoFrameReconstructionRouter((0,))
        with self.assertRaises(ValueError):
            router.route(torch.zeros(1, 1, 1, 1), torch.zeros(2, dtype=torch.bool),
                         lambda k, x, rows, first: x)
        with self.assertRaises(FloatingPointError):
            router.route(torch.zeros(1, 1, 1, 1), torch.tensor([True]),
                         lambda k, x, rows, first: x + float("nan"))


@unittest.skipUnless(vendor_available, "requires pinned vendor imports")
class TwoFrameAdapterTests(unittest.TestCase):
    def test_policy_batch_promotes_mixed_route_output_dtypes(self):
        from test_reconstruction_router import fixed_models
        for low_precision_route in (0, 1):
            wm, ac = fixed_models()
            original = wm.rssm.forward

            def mixed_forward(*args, **kwargs):
                q, z, h = original(*args, **kwargs)
                dtype = (torch.bfloat16 if kwargs["task_id"] == low_precision_route
                         else torch.float32)
                return q, z.to(dtype), h.to(dtype)

            router = TwoFrameReconstructionRouter((0, 1))
            z, h = wm.rssm.initial_state(2)
            obs = torch.stack((torch.zeros(3, 2, 2), torch.ones(3, 2, 2)))
            with mock.patch.object(wm.rssm, "forward", side_effect=mixed_forward):
                for t in range(3):
                    z, h, action = trajectory._routed_policy_step(
                        wm, ac, router, obs, z, h,
                        torch.nn.functional.one_hot(torch.zeros(2, dtype=torch.long), 18),
                        torch.full((2, 1), float(t == 0)), stochastic=False,
                    )
                    self.assertEqual(z.dtype, torch.float32)
                    self.assertEqual(h.dtype, torch.float32)
                    self.assertEqual(action.tolist(), [0, 1])
                    torch.testing.assert_close(h[:, 1], torch.full((2,), float(t + 1)))

    def test_historical_mode_rejected_before_environment_creation(self):
        for mode in ("first_frame_reconstruction", "unknown"):
            with mock.patch.object(trajectory, "AsyncVectorEnv") as environment:
                with self.assertRaises(ValueError):
                    trajectory.generate_trajectories(
                        2, 1, eligible_route_ids=(0,), task_route_inference=mode,
                    )
                environment.assert_not_called()

    def test_candidate_cache_preserves_recurrent_dtype_across_staggered_resets(self):
        from test_reconstruction_router import fixed_models
        wm, ac = fixed_models()
        original = wm.rssm.forward

        def mixed_forward(*args, **kwargs):
            q, z, h = original(*args, **kwargs)
            return q, z, h.to(torch.bfloat16)

        router = TwoFrameReconstructionRouter((0, 1))
        z, h = wm.rssm.initial_state(2)
        with mock.patch.object(wm.rssm, "forward", side_effect=mixed_forward):
            for first in ([True, True], [False, True], [True, False]):
                z, h, _ = trajectory._routed_policy_step(
                    wm, ac, router, torch.zeros(2, 3, 2, 2), z, h,
                    torch.nn.functional.one_hot(torch.zeros(2, dtype=torch.long), 18),
                    torch.tensor(first).float()[:, None], stochastic=False,
                )
                self.assertTrue(router.candidate_states)
                self.assertTrue(all(cached_h.dtype == torch.bfloat16
                                    for _, cached_h in router.candidate_states.values()))

    def test_real_rssm_scores_match_independent_histories_and_weights_unchanged(self):
        from retained_method_support import retained_world_model
        from ac import zh_to_ac_state
        wm = retained_world_model().eval()
        before = copy.deepcopy(wm.state_dict())

        class Actor:
            def actor(self, state):
                return torch.zeros(len(state), wm.a_dim)

        frames = torch.linspace(0, 1, 2 * 2 * 3 * 64 * 64).reshape(2, 2, 3, 64, 64)
        actions = torch.nn.functional.one_hot(torch.tensor([[0, 0], [2, 3]]), wm.a_dim)
        expected = {}
        with torch.no_grad():
            for k in (0, 1):
                z, h = wm.rssm.initial_state(2)
                for t in range(2):
                    q, z, h = wm.rssm(z, actions[t], h, frames[t],
                                      torch.full((2, 1), float(t == 0)),
                                      task_id=k, stochastic=False)
                    decoded = wm.decoder_for(k)(zh_to_ac_state(q.exp(), h))
                    expected[t, k] = (decoded.float() - frames[t]).square().flatten(1).mean(1)
        rng = torch.random.get_rng_state().clone()
        all_events = []
        for stochastic in (False, True):
            router = TwoFrameReconstructionRouter((0, 1))
            z, h = wm.rssm.initial_state(2)
            for t in range(2):
                z, h, _ = trajectory._routed_policy_step(
                    wm, Actor(), router, frames[t], z, h, actions[t],
                    torch.full((2, 1), float(t == 0)), stochastic=stochastic,
                )
                for event in router.events[-2:]:
                    w = event["worker_index"]
                    torch.testing.assert_close(torch.tensor(event["reconstruction_mse"]),
                                               torch.stack([expected[t, k][w] for k in (0, 1)]))
            all_events.append(router.events)
            if not stochastic:
                torch.testing.assert_close(torch.random.get_rng_state(), rng)
        self.assertEqual(all_events[0], all_events[1])
        torch.testing.assert_close(wm.state_dict(), before)
        self.assertTrue(all(p.grad is None for p in wm.parameters()))

    def test_next_step_terminal_is_not_a_second_probe_and_reset_is_local(self):
        from test_reconstruction_router import fixed_models, np
        wm, ac = fixed_models()
        fixture = mock.Mock()
        pixels = lambda values: np.array(values, np.uint8)[:, None, None, None] * np.ones((2, 2, 2, 3), np.uint8)
        fixture.reset.return_value = (pixels([0, 255]), {})
        fixture.step.side_effect = [
            (pixels([255, 0]), np.ones(2), np.array([True, False]), np.zeros(2, bool), {}),
            (pixels([255, 0]), np.ones(2), np.zeros(2, bool), np.zeros(2, bool), {}),
            (pixels([255, 0]), np.ones(2), np.zeros(2, bool), np.zeros(2, bool), {}),
            (pixels([255, 0]), np.ones(2), np.zeros(2, bool), np.zeros(2, bool), {}),
        ]
        diagnostics = {}
        with mock.patch.object(trajectory, "AsyncVectorEnv", return_value=fixture) as constructor:
            trajectory.generate_trajectories(
                10, 2, wm=wm, ac=ac, env_fns=[mock.sentinel.env] * 2,
                eligible_route_ids=(0, 1), deterministic_policy=True, seed=4,
                task_route_inference="two_frame_probability_reconstruction",
                routing_diagnostics=diagnostics,
            )
        self.assertNotIn("autoreset_mode", constructor.call_args.kwargs)
        events = diagnostics["routing_events"]
        self.assertEqual([(e["worker_index"], e["episode_index"], e["observation_count"])
                          for e in events], [(0, 0, 1), (1, 0, 1), (1, 0, 2), (0, 1, 1), (0, 1, 2)])
        self.assertEqual(events[3]["selected_route_id"], 1)
        fixture.close.assert_called_once()

    def test_probability_decoder_hard_actor_own_history_and_bounded_cost(self):
        class Rssm(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def initial_state(self, n):
                return torch.zeros(n, 1, 2), torch.zeros(n, 1)

            def forward(self, z, action, h, obs, reset, *, task_id, stochastic):
                self.calls.append((task_id, z.clone(), h.clone(), action.clone(), stochastic))
                p = torch.tensor([.8, .2] if task_id == 0 else [.2, .8]).expand(len(obs), 1, 2)
                hard = torch.nn.functional.one_hot(p.argmax(-1), 2).float()
                next_h = h * (1 - reset) + 10 * (task_id + 1) + action.argmax(-1)[:, None]
                return p.log(), hard, next_h

        class World:
            compute_dtype = "float32"
            rssm = Rssm()

            def decoder_for(self, route):
                def decode(features):
                    # Soft probabilities, not hard samples, only at this boundary.
                    assert bool(((features[:, :2] > 0) & (features[:, :2] < 1)).all())
                    return features[:, 1].reshape(-1, 1, 1, 1)
                return decode

        class Actor:
            def actor(self, features):
                assert bool(((features[:, :2] == 0) | (features[:, :2] == 1)).all())
                return features[:, :2]

        wm, ac = World(), Actor()
        router = TwoFrameReconstructionRouter((0, 1))
        z, h = wm.rssm.initial_state(1)
        rng = torch.random.get_rng_state().clone()
        for index, value in enumerate([.45, .9, .0]):
            previous = torch.nn.functional.one_hot(torch.tensor([index]), 3)
            z, h, action = trajectory._routed_policy_step(
                wm, ac, router, torch.tensor([value]).reshape(1, 1, 1, 1),
                z, h, previous, torch.zeros(1, 1), stochastic=False,
                route_reset=torch.tensor([index == 0]),
            )
            self.assertEqual(action.item(), 0 if index == 0 else 1)
            self.assertEqual(len(wm.rssm.calls), [3, 6, 7][index])
            if index == 1:
                self.assertEqual(h.item(), 41.)  # route 1's 20 + 20 + real action 1
                self.assertEqual(wm.rssm.calls[5][2].item(), 20.)
                self.assertEqual(wm.rssm.calls[5][1].tolist(), [[[0., 1.]]])
        self.assertFalse(router.candidate_states)
        torch.testing.assert_close(torch.random.get_rng_state(), rng)


if __name__ == "__main__":
    unittest.main()
