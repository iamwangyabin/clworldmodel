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

    def _native_r2_dry_run(self, *extra_args: str) -> dict:
        with TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "r2dreamer_arrow_run"
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_r2dreamer_arrow_atari.py",
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
            return json.loads(result.stdout.split("\ncommand:", maxsplit=1)[0])

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

    def test_native_r2_dry_run_uses_size12m_geometry_and_arrow_replay(self) -> None:
        launch = self._native_r2_dry_run()

        self.assertEqual(launch["method"], "R2Dreamer-ARROW-50")
        self.assertEqual(launch["role"], "native-r2dreamer-with-arrow-replay")
        self.assertEqual(launch["scope"], "single-task")
        self.assertEqual(launch["status_label"], "pilot")
        self.assertEqual(
            launch["upstream"]["r2dreamer"]["commit"],
            "546e4fab8146ea4b14e1d7726bbc1a8a1d50322f",
        )
        r2 = launch["r2dreamer"]
        self.assertFalse(r2["decoder_enabled"])
        self.assertEqual((r2["embedding_dim"], r2["rssm_feature_dim"]), (1024, 2560))
        self.assertEqual((r2["batch_size"], r2["batch_length"]), (16, 64))
        self.assertEqual(r2["flattened_barlow_samples"], 1024)
        self.assertEqual(r2["optimizer"], "LaProp")

        replay = launch["arrow_replay"]
        self.assertEqual((replay["fifo_slots"], replay["ltdm_slots"]), (512, 512))
        self.assertEqual(replay["buffer_selection"], {"fifo": 0.5, "ltdm": 0.5})
        self.assertEqual(replay["sample_context_steps"], 1)

        budget = launch["budget"]
        self.assertEqual(budget["task_count"], 1)
        self.assertEqual(budget["epochs"], 7)
        self.assertEqual(budget["nominal_world_model_updates_per_epoch"], 2_048)
        self.assertEqual(budget["native_train_ratio"], 128)
        self.assertEqual(budget["nominal_raw_frames_per_epoch"], 65_536)
        self.assertEqual(budget["single_task_target_raw_frames"], 410_000)
        self.assertEqual(budget["source_samples_per_epoch"], 512_000)
        self.assertEqual(budget["r2_samples_per_epoch"], 2_097_152)

        command = launch["command"]
        self.assertIn("--launcher-created-log-dir", command)
        self.assertEqual(command[command.index("--task-count") + 1], "1")
        self.assertEqual(command[command.index("--epochs") + 1], "7")
        self.assertEqual(command[command.index("--native-train-ratio") + 1], "128")

    def test_native_r2_smoke_allows_native_amp_scale_calibration(self) -> None:
        launch = self._native_r2_dry_run("--smoke")

        self.assertEqual(launch["status_label"], "smoke")
        self.assertEqual(launch["budget"]["epochs"], 1)
        self.assertEqual(launch["budget"]["nominal_world_model_updates_per_epoch"], 12)
        self.assertEqual(launch["r2dreamer"]["amp_initial_scale"], 65_536.0)
        command = launch["command"]
        self.assertEqual(
            command[command.index("--world-model-updates-per-epoch") + 1], "12"
        )
        self.assertIn("--require-optimizer-step", command)
        self.assertEqual(
            launch["smoke_checks"]["required_successful_optimizer_steps"], 1
        )

    def test_native_r2_full_single_task_matches_arrow_task_duration(self) -> None:
        launch = self._native_r2_dry_run("--scope", "single-task-full")

        self.assertEqual(launch["scope"], "single-task-full")
        self.assertEqual(launch["status_label"], "full-single-task-pilot")
        budget = launch["budget"]
        self.assertEqual(budget["task_count"], 1)
        self.assertEqual(budget["epochs"], 90)
        self.assertEqual(budget["source_task_switch_epochs"], 90)
        self.assertEqual(budget["single_task_target_raw_frames"], 5_898_240)
        self.assertEqual(budget["total_nominal_raw_frames"], 5_898_240)
        self.assertEqual(
            budget["total_nominal_r2_model_sample_transitions"], 188_743_680
        )

        command = launch["command"]
        self.assertEqual(command[command.index("--task-count") + 1], "1")
        self.assertEqual(command[command.index("--epochs") + 1], "90")


if __name__ == "__main__":
    unittest.main()
