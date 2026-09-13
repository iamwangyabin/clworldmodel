"""Dry-run contracts for the ARROW-budget Dream Rehearsal memory pair."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_dream_rehearsal_arrow_matched_atari.py"


class DreamRehearsalArrowMatchedLauncherTests(unittest.TestCase):
    def _dry_run(self, history: str, *extra: str) -> dict:
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / history
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--history",
                    history,
                    "--output-dir",
                    str(output),
                    "--dry-run",
                    *extra,
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertFalse(output.exists())
            return json.loads(result.stdout.split("\ncommand:", maxsplit=1)[0])

    def test_full_and_bounded_match_arrow_base_budgets(self) -> None:
        full = self._dry_run("full")
        bounded = self._dry_run("bounded")

        for launch in (full, bounded):
            self.assertEqual(
                launch["method"], "Dream-Rehearsal-ArrowMatched-v1-Atari"
            )
            scope = launch["training_scope"]
            self.assertEqual(scope["agent_decisions"], 8_863_744)
            self.assertEqual(scope["raw_environment_frames"], 35_454_976)
            self.assertEqual(scope["world_model_updates"], 541_000)
            self.assertEqual(scope["base_actor_updates"], 432_800)
            self.assertEqual(scope["base_critic_updates"], 432_800)
            self.assertEqual(scope["requested_periodic_evaluation_episodes"], 5_280)
            self.assertTrue(
                launch["comparison_contract"]["agent_decisions_match_arrow"]
            )
            self.assertFalse(
                launch["comparison_contract"]["total_optimizer_compute_matches_arrow"]
            )

        self.assertEqual(full["replay"]["transition_capacity"], 8_863_744)
        self.assertEqual(full["replay"]["trajectory_slots"], 17_312)
        self.assertTrue(full["replay"]["retains_all_declared_training_history"])
        self.assertFalse(full["replay"]["eviction_possible"])
        self.assertEqual(bounded["replay"]["transition_capacity"], 524_288)
        self.assertEqual(bounded["replay"]["trajectory_slots"], 1_024)
        self.assertTrue(bounded["replay"]["capacity_matches_arrow_50"])
        self.assertTrue(bounded["replay"]["eviction_possible"])
        self.assertEqual(
            full["memory_comparison_contract"][
                "shared_algorithm_and_schedule_sha256"
            ],
            bounded["memory_comparison_contract"][
                "shared_algorithm_and_schedule_sha256"
            ],
        )

    def test_reference_rehearsal_boundary_is_preserved_and_separately_counted(self) -> None:
        launch = self._dry_run("bounded")
        rehearsal = launch["rehearsal"]

        self.assertEqual(rehearsal["imagined_trajectories_per_update"], 1_024)
        self.assertEqual(rehearsal["selected_trajectories_per_update"], 256)
        self.assertEqual(rehearsal["extra_actor_only_updates"], 554_900)
        self.assertEqual(
            rehearsal["bootstrap_feature"],
            "last_imagined_pre_transition_feature",
        )
        self.assertTrue(
            launch["config_overrides"][
                "dream_rehearsal_bootstrap_last_imagined_feature"
            ]
        )
        self.assertEqual(
            launch["replay"]["ordinary_world_model_and_actor_critic_sampling"],
            "shared_retained_history",
        )
        self.assertFalse(launch["replay"]["separate_full_history_backup"])

    def test_smoke_crosses_a_boundary_without_claiming_comparability(self) -> None:
        launch = self._dry_run("bounded", "--smoke")

        self.assertEqual(launch["classification"], "smoke")
        self.assertEqual(launch["training_scope"]["epochs"], 2)
        self.assertEqual(launch["training_scope"]["agent_decisions"], 4_096)
        self.assertEqual(launch["rehearsal"]["extra_actor_only_updates"], 1)
        self.assertFalse(
            launch["comparison_contract"]["agent_decisions_match_arrow"]
        )


if __name__ == "__main__":
    unittest.main()
