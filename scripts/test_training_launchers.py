"""CPU-only contracts for reproducible continual-training launchers."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]


class TrainingLauncherTests(unittest.TestCase):
    def _arrow_dry_run(self, *extra_args: str) -> dict:
        with TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "arrow_run"
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_arrow_ar50_atari.py",
                    "--seed",
                    "0",
                    "--output-dir",
                    str(output_dir),
                    "--dry-run",
                    *extra_args,
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            launch = json.loads(result.stdout.split("\ncommand:", maxsplit=1)[0])
            return launch

    def test_arrow_dry_run_records_complete_analysis_snapshot_contract(self) -> None:
        launch = self._arrow_dry_run()
        output_dir = Path(launch["output_dir"])
        self.assertEqual(launch["method"], "ARROW-50")
        self.assertEqual(launch["role"], "primary-method")
        self.assertEqual(launch["fifo_slots"], 512)
        self.assertEqual(launch["ltdm_slots"], 512)
        self.assertEqual(launch["replay_buffer_selection"], {"fifo": 0.5, "ltdm": 0.5})
        self.assertEqual(launch["observation_objective"]["name"], "reconstruction")
        self.assertTrue(launch["observation_objective"]["decoder_enabled"])
        self.assertIsNone(launch["r2_dreamer_reference"])
        self.assertEqual(
            launch["analysis_snapshot_semantics"]["task_boundary_epochs"],
            [89, 179, 269, 359, 449, 539],
        )
        self.assertEqual(launch["analysis_snapshot_semantics"]["final_epoch"], 540)
        self.assertFalse(launch["analysis_snapshot_semantics"]["resumable"])

        command = launch["command"]
        self.assertNotIn("--observation-objective", command)
        self.assertEqual(command[command.index("--arrow-replay-ratio") + 1], "50-50")
        self.assertEqual(command[command.index("--log-dir") + 1], str(output_dir.resolve()))
        self.assertEqual(
            command[command.index("--analysis-snapshot-dir") + 1],
            str((output_dir / "analysis_snapshots").resolve()),
        )

    def test_r2_dry_run_is_decoder_free_and_keeps_arrow_50_replay(self) -> None:
        launch = self._arrow_dry_run("--observation-objective", "r2")

        self.assertEqual(launch["method"], "ARROW-R2Rep-50")
        self.assertEqual(launch["role"], "representation-objective-ablation")
        self.assertEqual((launch["fifo_slots"], launch["ltdm_slots"]), (512, 512))
        self.assertEqual(launch["replay_buffer_selection"], {"fifo": 0.5, "ltdm": 0.5})
        objective = launch["observation_objective"]
        self.assertEqual(objective["name"], "r2")
        self.assertFalse(objective["decoder_enabled"])
        self.assertEqual(objective["barlow_loss_scale"], 0.05)
        self.assertEqual(objective["redundancy_scale"], 5e-4)
        self.assertEqual(objective["normalization_eps"], 1e-8)
        self.assertEqual(objective["target_gradient"], "stopped")
        self.assertEqual(
            launch["project_pythonpath_prepend"], str((ROOT / "src").resolve())
        )
        self.assertEqual(
            launch["r2_dreamer_reference"]["commit"],
            "546e4fab8146ea4b14e1d7726bbc1a8a1d50322f",
        )

        command = launch["command"]
        self.assertEqual(command[command.index("--observation-objective") + 1], "r2")
        self.assertEqual(command[command.index("--r2-barlow-loss-scale") + 1], "0.05")
        self.assertEqual(command[command.index("--r2-redundancy-scale") + 1], "0.0005")
        self.assertEqual(command[command.index("--r2-normalization-eps") + 1], "1e-08")


if __name__ == "__main__":
    unittest.main()
