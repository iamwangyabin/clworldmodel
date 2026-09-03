"""CPU-only launch-contract tests for Bounded Dream Rehearsal."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "scripts/run_bounded_dream_rehearsal_atari.py"


class BoundedDreamRehearsalLauncherTests(unittest.TestCase):
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

    def test_default_is_transition_capacity_matched_and_never_unbounded(self) -> None:
        launch, output_dir = self._dry_run()

        self.assertEqual(launch["method"], "Bounded-Dream-Rehearsal-v1-Atari")
        self.assertEqual(
            launch["role"], "bounded-baseline-capacity-matched-to-arrow-50"
        )
        replay = launch["bounded_replay"]
        self.assertEqual(replay["trajectory_slots"], 1_024)
        self.assertEqual(replay["sequence_length"], 512)
        self.assertEqual(replay["transition_capacity"], 524_288)
        self.assertEqual(replay["matched_arrow_50_transition_capacity"], 524_288)
        self.assertTrue(replay["capacity_matches_arrow_50"])
        self.assertFalse(replay["never_clear_unbounded_replay"])
        self.assertEqual(
            replay["comparison_basis"],
            "trajectory_and_transition_capacity_primary",
        )
        self.assertEqual(replay["observation"]["dtype"], "uint8")
        self.assertEqual(replay["observation"]["device"], "cpu_mmap")
        self.assertEqual(replay["allocated_tensor_bytes"], 6_486_499_328)
        self.assertEqual(
            launch["config_overrides"]["replay_buffers"],
            [{"rb_type": "LongTermReplay", "rb_device": "cpu"}],
        )
        self.assertEqual(
            launch["config_overrides"]["continual_method"],
            "bounded_dream_rehearsal",
        )
        self.assertFalse(output_dir.exists())

    def test_method_keeps_one_shared_actor_and_reports_extra_compute(self) -> None:
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
            rehearsal["extra_actor_only_updates_by_replay_task"],
            {
                "0": 184_300,
                "1": 147_850,
                "2": 111_000,
                "3": 74_100,
                "4": 37_250,
                "5": 400,
            },
        )
        self.assertEqual(rehearsal["world_model_updates_from_rehearsal"], 0)
        self.assertEqual(rehearsal["critic_updates_from_rehearsal"], 0)
        self.assertTrue(
            launch["comparison_contract"][
                "extra_actor_compute_is_not_compute_matched_to_plain_dreamer"
            ]
        )
        reference = launch["dream_rehearsal_reference"]
        self.assertEqual(
            reference["commit"],
            "7680778f798be3a27a17c320cc875b573c45f0e1",
        )

    def test_capacity_override_is_named_as_an_ablation_without_more_samples(self) -> None:
        launch, _ = self._dry_run(
            "--replay-capacity-transitions", str(256 * 512)
        )

        self.assertEqual(launch["role"], "bounded-storage-capacity-ablation")
        self.assertEqual(launch["bounded_replay"]["trajectory_slots"], 256)
        self.assertEqual(
            launch["bounded_replay"]["transition_capacity"], 131_072
        )
        self.assertFalse(launch["bounded_replay"]["capacity_matches_arrow_50"])

    def test_non_integral_trajectory_capacity_is_rejected(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                SCRIPT,
                "--dry-run",
                "--replay-capacity-transitions",
                "1000",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be divisible", result.stderr)


if __name__ == "__main__":
    unittest.main()
