"""Launcher contracts for the Task1-seeded projector/RSSM-LoRA pilot."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_cnn_projector_lora_incremental import (  # noqa: E402
    COMPACT_LORA_PROFILE,
    DEFAULT_LORA_PROFILE,
    EXPECTED_TASKS,
    LORA_PROFILES,
    _incremental_config,
    _parser,
)


class CnnProjectorLoraLauncherTests(unittest.TestCase):
    @staticmethod
    def _source() -> dict:
        return {
            "epochs": 270,
            "continual_method": "cnn_fullbank_arrow",
            "n_sync": 4,
            "gen_seq_len": 4096,
            "env_repeat": 4,
            "steps_per_batch": 500,
            "ac_train_steps": 400,
            "esc": {
                "env_schedule_type": "SequentialEnvironments",
                "env_configs": [
                    {"name": name, "kwargs": {}, "rew_scale": 1.0}
                    for name in EXPECTED_TASKS
                ],
                "kwargs": {"swap_sched": 90},
            },
            "replay_buffers": [
                {"rb_type": "FifoReplay", "rb_device": "cpu"},
                {"rb_type": "LongTermReplay", "rb_device": "cpu"},
            ],
        }

    def test_config_preserves_source_and_starts_after_task1(self) -> None:
        source = self._source()
        original = copy.deepcopy(source)
        config = _incremental_config(source, epochs_after_task1=180)

        self.assertEqual(source, original)
        self.assertEqual(config["epochs"], 270)
        self.assertEqual(config["continual_method"], "cnn_projector_lora_arrow")
        self.assertFalse(config["task_banked_image_encoder"])
        self.assertTrue(config["task_projected_image_encoder"])
        self.assertEqual(
            (
                config["task_lora_recurrent_rank"],
                config["task_lora_representation_rank"],
                config["task_lora_transition_rank"],
            ),
            (128, 128, 32),
        )
        self.assertEqual(config["shared_core_mode"], "task1_frozen_projector_lora")
        self.assertEqual(config["random_policy"], "new")

    def test_smoke_duration_does_not_retrain_task1(self) -> None:
        config = _incremental_config(self._source(), epochs_after_task1=1)
        self.assertEqual(config["epochs"], 91)

    def test_compact_profile_changes_only_the_named_lora_ranks(self) -> None:
        source = self._source()
        capacity = _incremental_config(source, epochs_after_task1=180)
        compact = _incremental_config(
            source,
            epochs_after_task1=180,
            lora_ranks=LORA_PROFILES[COMPACT_LORA_PROFILE].ranks,
        )

        changed = {
            key
            for key in capacity
            if capacity.get(key) != compact.get(key)
        }
        self.assertEqual(
            changed,
            {
                "task_lora_recurrent_rank",
                "task_lora_representation_rank",
                "task_lora_transition_rank",
            },
        )
        self.assertEqual(
            (
                compact["task_lora_recurrent_rank"],
                compact["task_lora_representation_rank"],
                compact["task_lora_transition_rank"],
            ),
            (32, 32, 16),
        )
        profile = LORA_PROFILES[COMPACT_LORA_PROFILE]
        self.assertEqual(profile.expected_parameters_per_later_task, 643_648)
        self.assertEqual(4 * profile.expected_parameters_per_later_task, 2_574_592)

    def test_unnamed_lora_ranks_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "named profile"):
            _incremental_config(
                self._source(),
                epochs_after_task1=180,
                lora_ranks=(16, 16, 8),
            )

    def test_run_classification_is_explicit(self) -> None:
        smoke = _parser().parse_args(
            ["--task1-boundary-snapshot", "task1.pt", "--classification", "smoke"]
        )
        pilot = _parser().parse_args(["--task1-boundary-snapshot", "task1.pt"])

        self.assertEqual(smoke.classification, "smoke")
        self.assertEqual(pilot.classification, "pilot")
        self.assertEqual(pilot.lora_profile, DEFAULT_LORA_PROFILE)

        compact = _parser().parse_args(
            [
                "--task1-boundary-snapshot",
                "task1.pt",
                "--lora-profile",
                COMPACT_LORA_PROFILE,
            ]
        )
        self.assertEqual(compact.lora_profile, COMPACT_LORA_PROFILE)

    def test_curriculum_or_duration_changes_are_rejected(self) -> None:
        invalid = self._source()
        invalid["esc"]["env_configs"][2]["name"] = "ALE/Enduro-v5"
        with self.assertRaisesRegex(ValueError, "frozen three-task curriculum"):
            _incremental_config(invalid, epochs_after_task1=180)

        invalid = self._source()
        invalid["esc"]["kwargs"]["swap_sched"] = 45
        with self.assertRaisesRegex(ValueError, "90 epochs"):
            _incremental_config(invalid, epochs_after_task1=180)


if __name__ == "__main__":
    unittest.main()
