"""Launch-contract tests for never-clear Dream Rehearsal."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "scripts/run_never_clear_dream_rehearsal_atari.py"


class NeverClearDreamRehearsalLauncherTests(unittest.TestCase):
    def _dry_run(self, *extra_args: str) -> tuple[dict, Path]:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        output_dir = Path(temporary.name) / "run"
        result = subprocess.run(
            [
                sys.executable,
                SCRIPT,
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
        return launch, output_dir

    def test_full_protocol_retains_every_collected_trajectory(self) -> None:
        launch, output_dir = self._dry_run()

        self.assertEqual(
            launch["method"], "Never-Clear-Dream-Rehearsal-v1-Atari"
        )
        self.assertEqual(launch["role"], "never-clear-paper-semantics-pilot")
        replay = launch["full_history_replay"]
        self.assertEqual(replay["trajectory_slots"], 17_312)
        self.assertEqual(replay["sequence_length"], 512)
        self.assertEqual(replay["transition_capacity"], 8_863_744)
        self.assertEqual(
            replay["projected_collected_trajectory_slots"], 17_312
        )
        self.assertTrue(replay["no_overwrite_expected"])
        self.assertTrue(replay["never_clear_replay"])
        self.assertEqual(replay["ordinary_world_model_sampling"], "current_task_only")
        self.assertEqual(
            replay["ordinary_actor_critic_sampling"], "current_task_only"
        )
        self.assertEqual(replay["old_task_sampling"], "dream_rehearsal_only")
        self.assertEqual(replay["observation"]["dtype"], "uint8")
        self.assertEqual(replay["observation"]["allocated_bytes"], 108_917_686_272)
        self.assertEqual(replay["allocated_tensor_bytes"], 109_662_379_264)
        self.assertEqual(
            launch["config_overrides"]["replay_buffers"],
            [{"rb_type": "FifoReplay", "rb_device": "cpu"}],
        )
        self.assertEqual(
            launch["config_overrides"]["continual_method"],
            "never_clear_dream_rehearsal",
        )
        self.assertFalse(output_dir.exists())

    def test_paper_schedule_and_task_identity_isolation_are_preserved(self) -> None:
        launch, _ = self._dry_run()

        rehearsal = launch["rehearsal"]
        self.assertEqual(rehearsal["policy"], "single_shared_actor")
        self.assertFalse(rehearsal["task_id_exposed_to_world_model_or_actor"])
        self.assertEqual(rehearsal["interval_agent_decisions"], 2_000)
        self.assertEqual(rehearsal["updates_per_prior_task_per_interval"], 50)
        self.assertEqual(rehearsal["imagined_trajectories_per_update"], 64)
        self.assertEqual(rehearsal["selected_trajectories_per_update"], 16)
        self.assertEqual(rehearsal["extra_actor_only_updates"], 554_900)
        self.assertEqual(
            launch["comparison_contract"]["claim_eligibility"],
            "single_seed_pilot",
        )

    def test_short_run_is_explicitly_non_claimable_and_right_sized(self) -> None:
        launch, _ = self._dry_run("--smoke-epochs", "2")

        self.assertEqual(launch["role"], "gpu-smoke-execution-only")
        self.assertEqual(launch["training_scope"]["epochs"], 2)
        self.assertEqual(launch["full_history_replay"]["trajectory_slots"], 64)
        self.assertEqual(
            launch["full_history_replay"]["transition_capacity"], 32_768
        )
        self.assertEqual(
            launch["comparison_contract"]["claim_eligibility"],
            "execution_only",
        )

    def test_smoke_cannot_silently_become_the_full_protocol(self) -> None:
        result = subprocess.run(
            [sys.executable, SCRIPT, "--dry-run", "--smoke-epochs", "541"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be shorter", result.stderr)


if __name__ == "__main__":
    unittest.main()
