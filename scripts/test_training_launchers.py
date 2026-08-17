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

    def test_kan_actor_dry_run_changes_only_actor_and_keeps_arrow_50_replay(self) -> None:
        launch = self._arrow_dry_run("--actor-network", "relu_kan")

        self.assertEqual(launch["method"], "ARROW-KANActor-50")
        self.assertEqual(launch["role"], "actor-architecture-ablation")
        self.assertEqual((launch["fifo_slots"], launch["ltdm_slots"]), (512, 512))
        self.assertEqual(launch["replay_buffer_selection"], {"fifo": 0.5, "ltdm": 0.5})
        self.assertEqual(launch["observation_objective"]["name"], "reconstruction")

        actor = launch["actor"]
        self.assertEqual(actor["network"], "relu_kan")
        self.assertEqual(actor["critic_network"], "mlp")
        self.assertEqual(actor["input_features"], 1536)
        self.assertEqual(actor["recurrent_features"], 512)
        self.assertEqual(actor["kan_hidden_features"], 64)
        self.assertEqual(actor["kan_basis_count"], 8)
        self.assertEqual(actor["kan_input_range"], [0.0, 1.0])
        self.assertFalse(actor["kan_grid_trainable"])
        self.assertEqual(actor["trainable_parameters"], 795730)
        self.assertEqual(
            launch["project_pythonpath_prepend"], str((ROOT / "src").resolve())
        )

        command = launch["command"]
        self.assertEqual(command[command.index("--actor-network") + 1], "relu_kan")
        self.assertEqual(command[command.index("--arrow-replay-ratio") + 1], "50-50")

    def test_kan_actor_cannot_be_mixed_with_r2_in_one_run(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_arrow_ar50_atari.py",
                "--actor-network",
                "relu_kan",
                "--observation-objective",
                "r2",
                "--dry-run",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be tested independently", result.stderr)

    def test_two_task_kan_pilot_stops_at_second_boundary(self) -> None:
        launch = self._arrow_dry_run(
            "--actor-network", "relu_kan", "--task-prefix-length", "2"
        )

        self.assertEqual(launch["method"], "ARROW-KANActor-50-T2Pilot")
        self.assertEqual(launch["role"], "actor-architecture-pilot")
        self.assertEqual(launch["training_scope"]["task_prefix_length"], 2)
        self.assertEqual(launch["training_scope"]["epochs"], 180)
        self.assertEqual(
            launch["training_scope"]["tasks"],
            ["ALE/MsPacman-v5", "ALE/Boxing-v5"],
        )
        self.assertEqual(
            launch["analysis_snapshot_semantics"]["task_boundary_epochs"], [89, 179]
        )
        self.assertEqual(launch["analysis_snapshot_semantics"]["final_epoch"], 179)
        self.assertTrue(
            launch["analysis_snapshot_semantics"][
                "final_coincides_with_task_boundary"
            ]
        )
        command = launch["command"]
        self.assertEqual(command[command.index("--epochs") + 1], "180")
        self.assertIn("--evaluate-final", command)
        self.assertEqual(command[command.index("--arrow-replay-ratio") + 1], "50-50")
        self.assertTrue(launch["final_evaluation"]["enabled"])
        self.assertEqual(launch["final_evaluation"]["rollouts_per_task"], 16)
        self.assertFalse(launch["final_evaluation"]["enters_replay"])
        self.assertTrue(
            launch["final_evaluation"]["reports_raw_and_scaled_returns"]
        )

    def test_bounded_kan_t1_pilot_preserves_the_single_task_budget(self) -> None:
        launch = self._arrow_dry_run(
            "--actor-network", "relu_kan_bounded", "--task-prefix-length", "1"
        )

        self.assertEqual(launch["method"], "ARROW-KANActorBounded-50-T1TrainabilityPilot")
        self.assertEqual(launch["role"], "actor-trainability-pilot")
        self.assertEqual(launch["training_scope"]["task_prefix_length"], 1)
        self.assertEqual(launch["training_scope"]["epochs"], 90)
        self.assertEqual(launch["training_scope"]["tasks"], ["ALE/MsPacman-v5"])
        self.assertEqual(launch["analysis_snapshot_semantics"]["task_boundary_epochs"], [89])
        actor = launch["actor"]
        self.assertEqual(actor["network"], "relu_kan_bounded")
        self.assertEqual(actor["kan_hidden_adapter"], "layer_norm_sigmoid")
        self.assertEqual(actor["kan_hidden_adapter_layer_norm_epsilon"], 1e-3)
        self.assertEqual(actor["trainable_parameters"], 795_858)

        command = launch["command"]
        self.assertEqual(command[command.index("--actor-network") + 1], "relu_kan_bounded")
        self.assertEqual(command[command.index("--epochs") + 1], "90")
        self.assertIn("--evaluate-final", command)

    def test_bounded_kan_t1_extension_moves_the_only_task_boundary(self) -> None:
        launch = self._arrow_dry_run(
            "--actor-network",
            "relu_kan_bounded",
            "--task-prefix-length",
            "1",
            "--task-duration-epochs",
            "180",
        )

        self.assertEqual(
            launch["method"], "ARROW-KANActorBounded-50-T1-180EpochTrainabilityPilot"
        )
        self.assertEqual(launch["role"], "actor-trainability-budget-extension")
        scope = launch["training_scope"]
        self.assertEqual(scope["task_prefix_length"], 1)
        self.assertEqual(scope["epochs"], 180)
        self.assertEqual(scope["task_duration_epochs"], 180)
        self.assertEqual(scope["baseline_task_duration_epochs"], 90)
        self.assertEqual(scope["task_duration_epoch_override"], 180)
        self.assertEqual(scope["tasks"], ["ALE/MsPacman-v5"])
        self.assertEqual(
            launch["config_overrides"],
            {"epochs": 180, "esc.kwargs.swap_sched": 180},
        )
        self.assertTrue(
            launch["resolved_training_config"].endswith("resolved_training_config.json")
        )
        self.assertEqual(
            launch["analysis_snapshot_semantics"]["task_boundary_epochs"], [179]
        )
        self.assertEqual(launch["analysis_snapshot_semantics"]["final_epoch"], 179)

        command = launch["command"]
        self.assertEqual(command[command.index("--epochs") + 1], "180")
        self.assertTrue(
            command[command.index("--config") + 1].endswith(
                "resolved_training_config.json"
            )
        )

    def test_task_duration_extension_requires_bounded_kan_t1(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_arrow_ar50_atari.py",
                "--actor-network",
                "relu_kan_bounded",
                "--task-prefix-length",
                "2",
                "--task-duration-epochs",
                "180",
                "--dry-run",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires --task-prefix-length 1", result.stderr)


if __name__ == "__main__":
    unittest.main()
