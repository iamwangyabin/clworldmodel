"""Contracts for the DreamerV3 port of Continual-Dreamer replay."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = ROOT / "src"
VENDORED_ATARI = (
    ROOT / "third_party" / "arrow" / "Code" / "ARROW_and_DV3" / "Atari"
)

try:
    import torch
    import sortedcontainers  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - experiment environment coverage
    torch = None

if torch is not None:
    sys.path.insert(0, str(PROJECT_SRC))
    sys.path.insert(0, str(VENDORED_ATARI))
    from config import Config
    from replay import LongTermReplay


class Dv3RsMiniGridLauncherTests(unittest.TestCase):
    def _dry_run(self, script: str, output_dir: Path) -> dict:
        result = subprocess.run(
            [
                sys.executable,
                script,
                "--seed",
                "7",
                "--output-dir",
                str(output_dir),
                "--dry-run",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout.split("\ncommand:", maxsplit=1)[0])

    def test_dry_run_declares_matched_reservoir_mechanism_port(self) -> None:
        with TemporaryDirectory() as temporary:
            launch = self._dry_run(
                "scripts/run_dv3_rs_minigrid_smoke.py",
                Path(temporary) / "run",
            )
        self.assertEqual(launch["method"], "DV3-RS")
        self.assertEqual(
            launch["protocol"], "DV3-RS-MiniGrid-3Task-Smoke-v1"
        )
        self.assertTrue(launch["method_semantics"]["not_a_paper_reproduction"])
        self.assertEqual(
            launch["method_semantics"]["continual_dreamer_source"]["commit"],
            "77f05bcebc56ad2f9bc22f82f6d4d02e62da87f6",
        )
        self.assertFalse(launch["task_schedule"]["task_identity_exposed_to_agent"])
        self.assertEqual(launch["replay"]["total_trajectory_slots"], 16)
        self.assertEqual(launch["replay"]["fifo_trajectory_slots"], 0)
        self.assertEqual(launch["replay"]["reservoir_trajectory_slots"], 16)
        self.assertEqual(
            launch["replay"]["buffer_selection_probability"],
            {"fifo": 0.0, "reservoir": 1.0},
        )
        self.assertEqual(launch["budgets"]["environment_decisions"], 384)
        self.assertNotIn("--arrow-replay-ratio", launch["command"])

    def test_dv3_rs_and_arrow_have_identical_non_replay_budgets(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dv3_rs = self._dry_run(
                "scripts/run_dv3_rs_minigrid_smoke.py", root / "dv3_rs"
            )
            arrow = self._dry_run(
                "scripts/run_arrow_minigrid_smoke.py", root / "arrow"
            )

        self.assertEqual(dv3_rs["task_schedule"], arrow["task_schedule"])
        self.assertEqual(dv3_rs["budgets"], arrow["budgets"])
        self.assertEqual(dv3_rs["observation"], arrow["observation"])
        self.assertEqual(dv3_rs["action"], arrow["action"])
        for key in (
            "total_trajectory_slots",
            "sequence_length",
            "observation_dtype",
            "storage_device",
            "tensor_bytes_excluding_allocator_overhead",
        ):
            self.assertEqual(dv3_rs["replay"][key], arrow["replay"][key])


@unittest.skipIf(torch is None, "requires PyTorch")
class Dv3ReservoirRetentionTests(unittest.TestCase):
    def test_resolved_config_constructs_one_full_capacity_reservoir(self) -> None:
        config_data = json.loads(
            (ROOT / "configs" / "minigrid" / "dv3_rs_smoke.json").read_text(
                encoding="utf-8"
            )
        )
        config = Config.from_dict(config_data)
        replay = config.get_replay_buffer()

        self.assertIsInstance(replay, LongTermReplay)
        self.assertEqual(replay.n, 16)
        self.assertEqual(replay.t, 16)
        self.assertEqual(replay.acts.shape[-1], 7)

    def test_long_term_replay_retains_the_top_iid_random_keys(self) -> None:
        replay = LongTermReplay(2, 2, 7, "cpu")
        candidate_ids = torch.tensor([1.0, 2.0, 3.0, 4.0])
        actions = torch.zeros(2, 4, 7)
        observations = torch.zeros(2, 4, 3, 64, 64)
        rewards = candidate_ids.reshape(1, 4, 1).repeat(2, 1, 1)
        continues = torch.ones(2, 4, 1)
        resets = torch.zeros(2, 4, 1)

        with mock.patch.object(
            sys.modules[LongTermReplay.__module__].np.random,
            "randn",
            side_effect=[0.1, 0.4, -0.2, 1.0],
        ):
            replay.add(actions, observations, rewards, continues, resets)

        self.assertEqual(replay.n_valid, 2)
        retained_ids = sorted(replay.rews[0, :, 0].tolist())
        self.assertEqual(retained_ids, [2.0, 4.0])
        self.assertEqual(
            sorted(priority for priority, _ in replay.collection), [0.4, 1.0]
        )


if __name__ == "__main__":
    unittest.main()
