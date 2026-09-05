"""Contracts for the ARROW Continual-Dreamer MiniGrid smoke route."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
HAS_MINIGRID = importlib.util.find_spec("minigrid") is not None
HAS_GYMNASIUM = importlib.util.find_spec("gymnasium") is not None


class ArrowMiniGridLauncherTests(unittest.TestCase):
    def test_dry_run_declares_task_agnostic_arrow_smoke(self) -> None:
        with TemporaryDirectory() as temporary:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_arrow_minigrid_smoke.py",
                    "--seed",
                    "7",
                    "--output-dir",
                    str(Path(temporary) / "run"),
                    "--dry-run",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        launch = json.loads(result.stdout.split("\ncommand:", maxsplit=1)[0])
        self.assertEqual(launch["method"], "ARROW-50")
        self.assertEqual(
            launch["protocol"], "ARROW-50-MiniGrid-3Task-Smoke-v1"
        )
        self.assertEqual(launch["claim_scope"], "execution-correctness-smoke-only")
        self.assertFalse(launch["task_schedule"]["task_identity_exposed_to_agent"])
        self.assertEqual(launch["task_schedule"]["epochs_per_task"], [1, 1, 1])
        self.assertEqual(launch["action"], {"type": "discrete", "count": 7, "repeat": 1})
        self.assertEqual(launch["replay"]["fifo_trajectory_slots"], 8)
        self.assertEqual(launch["replay"]["ltdm_trajectory_slots"], 8)
        self.assertEqual(launch["replay"]["observation_dtype"], "float32")
        self.assertEqual(
            launch["replay"]["buffer_selection_probability"],
            {"fifo": 0.5, "reservoir": 0.5},
        )
        self.assertEqual(launch["budgets"]["environment_decisions"], 384)


@unittest.skipUnless(
    HAS_MINIGRID and HAS_GYMNASIUM,
    "requires the MiniGrid optional dependency",
)
class MiniGridAdapterTests(unittest.TestCase):
    def test_vendored_trajectory_reinterpretation_preserves_seven_actions(self) -> None:
        import torch

        vendored = (
            ROOT / "third_party" / "arrow" / "Code" / "ARROW_and_DV3" / "Atari"
        )
        sys.path.insert(0, str(vendored))
        try:
            import generate_trajectory

            samples = 6
            actions = torch.nn.functional.one_hot(
                torch.tensor([0, 1, 2, 3, 4, 6]), 7
            ).float()
            observations = torch.zeros(samples, 3, 64, 64)
            scalar = torch.zeros(samples, 1)
            reshaped = generate_trajectory.reinterpret_nt_to_t_n(
                actions, observations, scalar, scalar, scalar, 3, 2
            )
            self.assertEqual(reshaped[0].shape, (3, 2, 7))
            self.assertTrue(torch.equal(reshaped[0][2, 1], actions[-1]))
        finally:
            sys.path.remove(str(vendored))

    def test_all_paper_tasks_have_repeatable_visual_boundaries(self) -> None:
        import numpy as np

        from clworldmodel.environments.minigrid import (
            MINIGRID_PAPER_TASKS,
            make_minigrid_environment,
        )

        for name in MINIGRID_PAPER_TASKS:
            with self.subTest(name=name):
                env = make_minigrid_environment(name)
                try:
                    first, _ = env.reset(seed=11)
                    second, _ = env.reset(seed=11)
                    self.assertEqual(first.shape, (64, 64, 3))
                    self.assertEqual(first.dtype, np.uint8)
                    self.assertTrue(np.array_equal(first, second))
                    self.assertEqual(env.action_space.n, 7)
                    if name == "MiniGrid-DoorKey-9x9-v0":
                        self.assertEqual(env.unwrapped.width, 9)
                        self.assertEqual(env.unwrapped.height, 9)
                    env.action_space.seed(13)
                    actions_a = [env.action_space.sample() for _ in range(8)]
                    env.action_space.seed(13)
                    actions_b = [env.action_space.sample() for _ in range(8)]
                    self.assertEqual(actions_a, actions_b)
                    transition = env.step(actions_a[0])
                    self.assertEqual(len(transition), 5)
                finally:
                    env.close()

    def test_released_source_doorkey_geometry_is_explicitly_8x8(self) -> None:
        from clworldmodel.environments.minigrid import make_minigrid_environment

        env = make_minigrid_environment(
            "MiniGrid-DoorKey-9x9-v0",
            doorkey_geometry="released_source_8x8",
        )
        try:
            self.assertEqual(env.unwrapped.width, 8)
            self.assertEqual(env.unwrapped.height, 8)
        finally:
            env.close()

        with self.assertRaisesRegex(ValueError, "Unsupported DoorKey geometry"):
            make_minigrid_environment(
                "MiniGrid-DoorKey-9x9-v0",
                doorkey_geometry="implicit",
            )

    def test_vendored_config_dispatches_only_declared_minigrid_family(self) -> None:
        vendored = (
            ROOT / "third_party" / "arrow" / "Code" / "ARROW_and_DV3" / "Atari"
        )
        sys.path.insert(0, str(vendored))
        try:
            import config as arrow_config

            declared = arrow_config.EnvConfig(
                name="MiniGrid-DoorKey-9x9-v0",
                family="minigrid",
                kwargs={"max_episode_steps": 100, "tile_size": 8},
            )
            self.assertEqual(declared.to_dict()["family"], "minigrid")
            self.assertNotIn(
                "family", arrow_config.EnvConfig(name="ALE/Pong-v5").to_dict()
            )
            env = declared.get_function()()
            try:
                observation, _ = env.reset(seed=17)
                self.assertEqual(observation.shape, (64, 64, 3))
                self.assertEqual(env.action_space.n, 7)
            finally:
                env.close()
            with self.assertRaisesRegex(ValueError, "Unknown environment family"):
                arrow_config.EnvConfig(name="invalid", family="unknown")
        finally:
            sys.path.remove(str(vendored))


if __name__ == "__main__":
    unittest.main()
