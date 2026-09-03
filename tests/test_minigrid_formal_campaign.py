"""Contracts for the matched ARROW-50 versus DV3-RS MiniGrid campaign."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = ROOT / "src"
VENDORED_ATARI = ROOT / "third_party" / "arrow" / "Code" / "ARROW_and_DV3" / "Atari"

try:
    import torch
    import sortedcontainers  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - experiment environment coverage
    torch = None

if torch is not None:
    sys.path.insert(0, str(PROJECT_SRC))
    sys.path.insert(0, str(VENDORED_ATARI))
    from config import Config


class MiniGridFormalLauncherTests(unittest.TestCase):
    def _dry_run(self, script: str, seed_index: int = 0) -> dict:
        with TemporaryDirectory() as temporary:
            result = subprocess.run(
                [
                    sys.executable,
                    script,
                    "--seed-index",
                    str(seed_index),
                    "--output-dir",
                    str(Path(temporary) / "run"),
                    "--dry-run",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        return json.loads(result.stdout.split("\ncommand:", maxsplit=1)[0])

    def test_formal_methods_match_outside_replay(self) -> None:
        arrow = self._dry_run("scripts/run_arrow_minigrid_formal.py")
        dv3_rs = self._dry_run("scripts/run_dv3_rs_minigrid_formal.py")

        self.assertEqual(arrow["protocol"], "ARROW-DV3RS-MiniGrid-3Task-v1")
        self.assertEqual(arrow["evidence_level"], "official-candidate")
        self.assertEqual(arrow["task_schedule"], dv3_rs["task_schedule"])
        self.assertEqual(arrow["budgets"], dv3_rs["budgets"])
        self.assertEqual(arrow["observation"], dv3_rs["observation"])
        self.assertEqual(arrow["action"], dv3_rs["action"])
        self.assertEqual(arrow["evaluation"], dv3_rs["evaluation"])
        self.assertEqual(arrow["backend"], dv3_rs["backend"])

        budgets = arrow["budgets"]
        self.assertEqual(budgets["environment_decisions"], 2_250_000)
        self.assertEqual(
            budgets["per_task_environment_decisions"],
            [750_000, 750_000, 750_000],
        )
        self.assertEqual(budgets["initial_random_prefill_decisions"], 10_000)
        self.assertEqual(budgets["world_model_updates"], 137_250)
        self.assertEqual(budgets["actor_critic_updates"], 109_809)
        self.assertTrue(arrow["evaluation"]["future_tasks_evaluated"])
        self.assertEqual(
            arrow["evaluation"]["policy"],
            "deterministic_argmax_and_latent_mode",
        )

    def test_formal_replay_capacity_and_bytes_are_matched(self) -> None:
        arrow = self._dry_run("scripts/run_arrow_minigrid_formal.py")
        dv3_rs = self._dry_run("scripts/run_dv3_rs_minigrid_formal.py")
        arrow_replay = arrow["replay"]
        dv3_replay = dv3_rs["replay"]

        for key in (
            "transition_capacity",
            "observation_dtype",
            "sampled_observation_dtype",
            "storage_device",
            "observation_bytes",
            "action_reward_continue_reset_bytes",
            "tensor_bytes_excluding_allocator_overhead",
        ):
            self.assertEqual(arrow_replay[key], dv3_replay[key])
        self.assertEqual(arrow_replay["transition_capacity"], 2_000_000)
        self.assertEqual(arrow_replay["fifo_trajectory_slots"], 20_000)
        self.assertEqual(arrow_replay["reservoir_trajectory_slots"], 20_000)
        self.assertEqual(dv3_replay["fifo_trajectory_slots"], 0)
        self.assertEqual(dv3_replay["reservoir_trajectory_slots"], 40_000)
        self.assertEqual(
            arrow_replay["tensor_bytes_excluding_allocator_overhead"],
            24_656_000_000,
        )
        self.assertEqual(arrow_replay["storage_device"], "cpu")
        self.assertEqual(arrow_replay["observation_dtype"], "uint8")

    def test_seed_indices_map_to_predeclared_values(self) -> None:
        expected = [123456789, 1337, 31337, 42, 987654321]
        observed = [
            self._dry_run("scripts/run_arrow_minigrid_formal.py", index)["seed"][
                "value"
            ]
            for index in range(5)
        ]
        self.assertEqual(observed, expected)


@unittest.skipIf(torch is None, "requires PyTorch")
class MiniGridFormalConfigTests(unittest.TestCase):
    def test_formal_uint8_configs_are_schema_valid_without_allocating_replay(self) -> None:
        for name in ("arrow_50_formal_v1.json", "dv3_rs_formal_v1.json"):
            with self.subTest(name=name):
                data = json.loads(
                    (ROOT / "configs" / "minigrid" / name).read_text(
                        encoding="utf-8"
                    )
                )
                config = Config.from_dict(data)
                self.assertEqual(config.replay_observation_dtype, "uint8")
                self.assertTrue(config.deterministic_evaluation)
                self.assertTrue(
                    all(item.rb_device == "cpu" for item in config.replay_buffers)
                )

    def test_uint8_replay_remains_rejected_outside_opted_in_protocols(self) -> None:
        data = json.loads(
            (ROOT / "configs" / "minigrid" / "dv3_rs_formal_v1.json").read_text(
                encoding="utf-8"
            )
        )
        invalid = deepcopy(data)
        for environment in invalid["esc"]["env_configs"]:
            environment.pop("family")
        with self.assertRaisesRegex(ValueError, "uint8 observation replay is reserved"):
            Config.from_dict(invalid)


if __name__ == "__main__":
    unittest.main()
