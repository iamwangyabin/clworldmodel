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
        self.assertEqual(
            launch["project_pythonpath_prepend"], str((ROOT / "src").resolve())
        )
        self.assertEqual(launch["fifo_slots"], 512)
        self.assertEqual(launch["ltdm_slots"], 512)
        self.assertEqual(launch["replay_buffer_selection"], {"fifo": 0.5, "ltdm": 0.5})
        determinism = launch["determinism"]
        self.assertEqual(determinism["python_random_seed"], 123456789)
        self.assertEqual(determinism["replay_buffer_selection_rng"], "python_random")
        self.assertTrue(determinism["environment_reset_seeded"])
        self.assertTrue(determinism["action_space_seeded"])
        self.assertTrue(determinism["evaluation_rng_state_restored"])
        self.assertNotEqual(
            determinism["environment_seed_streams"]["collection"], "global_numpy"
        )
        self.assertTrue(determinism["known_nondeterminism"])
        replay_storage = launch["replay_storage_budget"]
        self.assertEqual(replay_storage["dtype"], "float32")
        self.assertEqual(replay_storage["observation_bytes"], 25_769_803_776)
        self.assertEqual(replay_storage["allocated_tensor_bytes"], 25_813_843_968)
        self.assertEqual(replay_storage["actor_comparison_difference_bytes"], 0)
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
        self.assertNotIn("--evaluate-final", command)
        self.assertFalse(launch["final_evaluation"]["enabled"])
        self.assertEqual(command[command.index("--arrow-replay-ratio") + 1], "50-50")
        self.assertEqual(command[command.index("--log-dir") + 1], str(output_dir.resolve()))
        self.assertEqual(
            command[command.index("--analysis-snapshot-dir") + 1],
            str((output_dir / "analysis_snapshots").resolve()),
        )

    def test_arrow_cpu_replay_profile_keeps_float32_capacity_and_sampling(self) -> None:
        launch = self._arrow_dry_run("--replay-device", "cpu")

        self.assertEqual(
            launch["runtime"], "vendored-optimized-cpu-float32-replay"
        )
        self.assertIn("cpu-resident-float32-replay", launch["optimizations"])
        self.assertEqual(
            launch["config_overrides"],
            {
                "replay_buffers": [
                    {"rb_type": "FifoReplay", "rb_device": "cpu"},
                    {"rb_type": "LongTermReplay", "rb_device": "cpu"},
                ],
                "replay_observation_dtype": "float32",
            },
        )
        self.assertTrue(
            launch["resolved_training_config"].endswith(
                "resolved_training_config.json"
            )
        )
        replay_profile = launch["replay_execution_profile"]
        self.assertEqual(replay_profile["storage_device"], "cpu")
        self.assertEqual(replay_profile["observation_dtype"], "float32")
        self.assertTrue(replay_profile["capacity_unchanged"])
        self.assertTrue(replay_profile["fifo_ltdm_retention_unchanged"])
        self.assertTrue(replay_profile["buffer_selection_probability_unchanged"])
        self.assertTrue(replay_profile["sampled_tensor_values_and_dtype_unchanged"])
        self.assertTrue(replay_profile["minibatches_transferred_to_cuda_after_sampling"])

        replay_storage = launch["replay_storage_budget"]
        self.assertEqual(replay_storage["dtype"], "float32")
        self.assertEqual(replay_storage["observation_bytes"], 25_769_803_776)
        self.assertEqual(replay_storage["allocated_tensor_bytes"], 25_813_843_968)
        self.assertEqual(replay_storage["buffers"]["fifo"]["device"], "cpu")
        self.assertEqual(replay_storage["buffers"]["ltdm"]["device"], "cpu")

        command = launch["command"]
        self.assertTrue(
            command[command.index("--config") + 1].endswith(
                "resolved_training_config.json"
            )
        )

    def test_arrow_single_task_cpu_replay_uses_published_task_config(self) -> None:
        launch = self._arrow_dry_run(
            "--single-task-index",
            "2",
            "--replay-device",
            "cpu",
        )

        self.assertEqual(launch["method"], "ARROW-50-SingleTask")
        self.assertEqual(
            launch["role"], "single-task-normalization-reproduction"
        )
        self.assertEqual(launch["curriculum"], "single-task")
        self.assertTrue(
            launch["source_config"].endswith(
                "ALE_CrazyClimber-e2-s0-arrow.json"
            )
        )
        training_scope = launch["training_scope"]
        self.assertEqual(training_scope["single_task_index"], 2)
        self.assertFalse(training_scope["full_curriculum"])
        self.assertEqual(training_scope["epochs"], 91)
        self.assertEqual(training_scope["task_duration_epochs"], 90)
        self.assertEqual(training_scope["tasks"], ["ALE/CrazyClimber-v5"])
        self.assertEqual(
            launch["analysis_snapshot_semantics"]["task_boundary_epochs"],
            [90],
        )
        self.assertEqual(
            launch["analysis_snapshot_semantics"]["final_epoch"], 90
        )
        self.assertTrue(
            launch["analysis_snapshot_semantics"][
                "final_coincides_with_task_boundary"
            ]
        )
        self.assertEqual(
            launch["replay_execution_profile"]["storage_device"], "cpu"
        )
        self.assertEqual(
            launch["replay_storage_budget"]["allocated_tensor_bytes"],
            25_813_843_968,
        )
        self.assertNotIn("--evaluate-final", launch["command"])

    def test_arrow_single_task_rejects_method_ablation_overrides(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_arrow_ar50_atari.py",
                "--single-task-index",
                "0",
                "--actor-network",
                "relu_kan",
                "--dry-run",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(result.returncode, 0)
        # Retirement removed non-MLP actor choices, so argparse rejects the
        # retired override before the single-task config is even selected.
        self.assertIn("invalid choice", result.stderr)
        self.assertIn("relu_kan", result.stderr)

    def test_swanlab_mirroring_records_names_but_no_credential(self) -> None:
        launch = self._arrow_dry_run(
            "--swanlab-project",
            "clworldmodel",
            "--swanlab-experiment-name",
            "arrow-test",
        )
        logging = launch["metric_logging"]
        self.assertTrue(logging["swanlab_enabled"])
        self.assertEqual(logging["swanlab_project"], "clworldmodel")
        self.assertEqual(logging["swanlab_experiment_name"], "arrow-test")
        self.assertNotIn("api_key", json.dumps(launch).lower())

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
