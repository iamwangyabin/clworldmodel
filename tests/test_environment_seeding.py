"""Contracts for isolated, deterministic Atari environment seed streams."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
VENDORED_ATARI = ROOT / "third_party" / "arrow" / "Code" / "ARROW_and_DV3" / "Atari"

try:
    import numpy as np
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in the GPU environment
    np = None
    torch = None

if torch is not None:
    sys.path.insert(0, str(VENDORED_ATARI))
    import generate_trajectory
    import train


@unittest.skipIf(torch is None, "requires the pinned PyTorch experiment environment")
class EnvironmentSeedingTests(unittest.TestCase):
    def test_collection_and_evaluation_seed_streams_are_stable_and_disjoint(self) -> None:
        collect_a, evaluate_a = train._environment_seed_streams(123456789)
        collect_b, evaluate_b = train._environment_seed_streams(123456789)

        collect_values_a = [train._next_environment_seed(collect_a) for _ in range(4)]
        collect_values_b = [train._next_environment_seed(collect_b) for _ in range(4)]
        evaluate_values_a = [train._next_environment_seed(evaluate_a) for _ in range(4)]
        evaluate_values_b = [train._next_environment_seed(evaluate_b) for _ in range(4)]

        self.assertEqual(collect_values_a, collect_values_b)
        self.assertEqual(evaluate_values_a, evaluate_values_b)
        self.assertNotEqual(collect_values_a, evaluate_values_a)

    def test_worker_reset_and_action_seeds_are_stable_and_disjoint(self) -> None:
        resets_a, actions_a = generate_trajectory._environment_worker_seeds(17, 4)
        resets_b, actions_b = generate_trajectory._environment_worker_seeds(17, 4)

        self.assertEqual(resets_a, resets_b)
        self.assertEqual(actions_a, actions_b)
        self.assertEqual(len(set(resets_a + actions_a)), 8)

    def test_seed_reaches_vector_reset_and_each_action_space(self) -> None:
        class FakeVectorEnv:
            reset_seed = None

            def __init__(self, factories) -> None:
                self.factories = factories

            def reset(self, *, seed=None):
                type(self).reset_seed = seed
                return np.zeros((2, 64, 64, 3), dtype=np.uint8), {}

        expected_resets, expected_actions = (
            generate_trajectory._environment_worker_seeds(29, 2)
        )
        with mock.patch.object(generate_trajectory, "AsyncVectorEnv", FakeVectorEnv):
            generate_trajectory.generate_trajectories(
                2,
                2,
                env_fns=[lambda: None, lambda: None],
                seed=29,
            )
        self.assertEqual(FakeVectorEnv.reset_seed, expected_resets)

        class FakeActionSpace:
            seed_value = None

            def seed(self, seed) -> None:
                self.seed_value = seed

        class FakeEnv:
            action_space = FakeActionSpace()

        fake_env = FakeEnv()
        with mock.patch.object(
            generate_trajectory,
            "AtariPreprocessing",
            side_effect=lambda env, **_kwargs: env,
        ):
            result = generate_trajectory._make_atari_env(
                lambda: fake_env, env_repeat=4, action_seed=expected_actions[0]
            )
        self.assertIs(result, fake_env)
        self.assertEqual(fake_env.action_space.seed_value, expected_actions[0])

    def test_evaluation_scope_restores_parent_rng_states(self) -> None:
        def seed_all() -> None:
            random.seed(31)
            np.random.seed(31)
            torch.manual_seed(31)

        seed_all()
        with train._preserve_training_rng_state():
            random.random()
            np.random.random()
            torch.rand(3)
            if torch.cuda.is_available():
                torch.rand(3, device="cuda")

        actual = (random.random(), np.random.random(), torch.rand(3))
        actual_cuda = torch.rand(3, device="cuda") if torch.cuda.is_available() else None

        seed_all()
        expected = (random.random(), np.random.random(), torch.rand(3))
        expected_cuda = (
            torch.rand(3, device="cuda") if torch.cuda.is_available() else None
        )

        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
        if actual_cuda is not None:
            torch.testing.assert_close(actual_cuda, expected_cuda, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
