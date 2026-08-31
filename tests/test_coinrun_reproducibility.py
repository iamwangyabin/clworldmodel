"""Reproducibility and raw-evaluation coverage for vendored CoinRun."""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
VENDORED_COINRUN = (
    ROOT / "third_party" / "arrow" / "Code" / "ARROW_and_DV3" / "CoinRun"
)
PUBLISHED_CONFIG = (
    ROOT
    / "third_party"
    / "arrow"
    / "Configs"
    / "CoinRun configs"
    / "CL-task configs"
    / "Original Order"
    / (
        "CoinRun,CoinRun+NB,CoinRun+NB+RT,CoinRun+NB+RT+GA,"
        "CoinRun+NB+RT+GA+MA,CoinRun+NB+RT+GA+MA+CA-s0-arrow.json"
    )
)

try:
    import cv2  # noqa: F401
    import gym  # noqa: F401
    import sortedcontainers  # noqa: F401
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in experiment envs
    torch = None


def _load_coinrun_modules():
    module_names = (
        "ac",
        "config",
        "generate_trajectory",
        "replay",
        "rssm",
        "vae",
        "wm",
    )
    saved_modules = {name: sys.modules.pop(name, None) for name in module_names}
    sys.path.insert(0, str(VENDORED_COINRUN))
    try:
        coinrun_config = importlib.import_module("config")
        coinrun_generate = importlib.import_module("generate_trajectory")
        return coinrun_config, coinrun_generate
    finally:
        sys.path.remove(str(VENDORED_COINRUN))
        for name in module_names:
            sys.modules.pop(name, None)
            if saved_modules[name] is not None:
                sys.modules[name] = saved_modules[name]


if torch is not None:
    COINRUN_CONFIG, COINRUN_GENERATE = _load_coinrun_modules()


@unittest.skipIf(torch is None, "requires CoinRun runtime dependencies")
class CoinRunReproducibilityTests(unittest.TestCase):
    def test_published_config_keeps_deterministic_runtime_seeding_opt_in(self) -> None:
        config = COINRUN_CONFIG.Config.from_file(PUBLISHED_CONFIG)
        self.assertFalse(config.deterministic_runtime_seeding)
        self.assertFalse(config.to_dict()["deterministic_runtime_seeding"])

    def test_seeded_environment_factory_uses_distinct_repeatable_seeds(self) -> None:
        config = COINRUN_CONFIG.EnvConfig(name="CoinRun+NB+RT")
        with mock.patch.object(
            COINRUN_CONFIG.gym,
            "make",
            side_effect=lambda name, **kwargs: (name, kwargs),
        ):
            factory = config.get_function(seed=123)
            first = factory()
            second = factory()
            published = config.get_function()()

        self.assertEqual(first[1]["rand_seed"], 123)
        self.assertEqual(second[1]["rand_seed"], 124)
        self.assertNotIn("rand_seed", published[1])
        self.assertFalse(first[1]["use_backgrounds"])
        self.assertTrue(first[1]["restrict_themes"])

    def test_config_builds_disjoint_training_and_evaluation_seed_streams(self) -> None:
        config = COINRUN_CONFIG.Config.from_file(PUBLISHED_CONFIG)
        config.deterministic_runtime_seeding = True
        observed_seeds = []

        def fake_get_function(_env_config, seed=None):
            observed_seeds.append(seed)
            return lambda: seed

        with mock.patch.object(
            COINRUN_CONFIG.EnvConfig,
            "get_function",
            autospec=True,
            side_effect=fake_get_function,
        ):
            schedule = config.get_env_schedule()

        expected_training = [
            COINRUN_CONFIG._procgen_factory_seed(
                config.seed, task_index, evaluation=False
            )
            for task_index in range(6)
        ]
        expected_evaluation = [
            COINRUN_CONFIG._procgen_factory_seed(
                config.seed, task_index, evaluation=True
            )
            for task_index in range(6)
        ]
        self.assertEqual(observed_seeds, expected_training + expected_evaluation)
        self.assertTrue(set(expected_training).isdisjoint(expected_evaluation))
        self.assertIsNot(schedule.templates, schedule.eval_templates)

    def test_complete_raw_episode_returns_are_preserved(self) -> None:
        rewards = torch.tensor([0, 1, 2, 0, 3, 4], dtype=torch.float32).reshape(
            6, 1
        )
        continues = torch.tensor([1, 1, 0, 1, 1, 0], dtype=torch.float32).reshape(
            6, 1
        )
        resets = torch.tensor([1, 0, 0, 1, 0, 0], dtype=torch.float32).reshape(
            6, 1
        )

        returns = COINRUN_GENERATE._completed_episode_returns(
            rewards, continues, resets
        )

        self.assertEqual(returns, [3.0, 7.0])

    def test_evaluation_task_set_must_match_training_task_set(self) -> None:
        schedule = COINRUN_GENERATE.SequentialEnvironments(
            2, [lambda: "train"], swap_sched=1
        )
        with self.assertRaisesRegex(ValueError, "equal length"):
            schedule.set_eval_templates([])


if __name__ == "__main__":
    unittest.main()
