"""Launcher contracts for the Task1-seeded MB-RSSM pilot and ablation."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_cnn_mechanism_bank_incremental import (  # noqa: E402
    EXPECTED_TASKS,
    MECHANISM_PARAMETERS_PER_LATER_TASK,
    MECHANISM_RESIDUAL_SCALE,
    MECHANISM_WIDTHS,
    PROTOCOL,
    _compile_environment_override,
    _incremental_config,
    _parser,
)


class CnnMechanismBankLauncherTests(unittest.TestCase):
    def test_compile_environment_records_only_explicit_libcuda_discovery(self) -> None:
        self.assertEqual(
            _compile_environment_override(
                {
                    "TRITON_LIBCUDA_PATH": "/usr/lib/x86_64-linux-gnu",
                    "PATH": "/usr/bin",
                }
            ),
            {"TRITON_LIBCUDA_PATH": "/usr/lib/x86_64-linux-gnu"},
        )
        self.assertEqual(_compile_environment_override({}), {})

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

    def test_config_preserves_source_and_selects_only_mechanism_paths(self) -> None:
        source = self._source()
        original = copy.deepcopy(source)
        config = _incremental_config(source, epochs_after_task1=180)

        self.assertEqual(source, original)
        self.assertEqual(config["epochs"], 270)
        self.assertEqual(config["continual_method"], "cnn_mechanism_bank_arrow")
        self.assertTrue(config["task_projected_image_encoder"])
        self.assertFalse(config["task_banked_image_encoder"])
        self.assertEqual(
            (
                config["task_lora_recurrent_rank"],
                config["task_lora_representation_rank"],
                config["task_lora_transition_rank"],
                config["task_recurrent_output_adapter_features"],
            ),
            (0, 0, 0, 0),
        )
        self.assertTrue(config["task_mechanism_bank"])
        self.assertTrue(config["task_mechanism_reuse"])
        self.assertEqual(
            (
                config["task_mechanism_recurrent_width"],
                config["task_mechanism_representation_width"],
                config["task_mechanism_transition_width"],
            ),
            MECHANISM_WIDTHS,
        )
        self.assertEqual(
            config["task_mechanism_residual_scale"], MECHANISM_RESIDUAL_SCALE
        )
        self.assertEqual(
            config["shared_core_mode"], "task1_frozen_mechanism_bank"
        )
        self.assertFalse(config["shared_actor_imagination_distillation"])
        self.assertEqual(PROTOCOL.split("-v1-")[0], "CNN-MechanismBank-RSSM-ARROW")

    def test_no_reuse_changes_only_the_static_reuse_switch(self) -> None:
        reuse = _incremental_config(
            self._source(), epochs_after_task1=180, reuse_enabled=True
        )
        no_reuse = _incremental_config(
            self._source(), epochs_after_task1=180, reuse_enabled=False
        )
        changed = {
            key for key in reuse if reuse.get(key) != no_reuse.get(key)
        }
        self.assertEqual(changed, {"task_mechanism_reuse"})
        self.assertEqual(MECHANISM_PARAMETERS_PER_LATER_TASK, 3_816_192)

    def test_smoke_duration_does_not_retrain_task1(self) -> None:
        config = _incremental_config(self._source(), epochs_after_task1=1)
        self.assertEqual(config["epochs"], 91)

    def test_cli_exposes_reuse_ablation_and_explicit_classification(self) -> None:
        default = _parser().parse_args(
            ["--task1-boundary-snapshot", "task1.pt"]
        )
        no_reuse = _parser().parse_args(
            [
                "--task1-boundary-snapshot",
                "task1.pt",
                "--reuse-mode",
                "no-reuse",
                "--classification",
                "smoke",
            ]
        )
        self.assertEqual(default.reuse_mode, "reuse")
        self.assertEqual(default.classification, "pilot")
        self.assertEqual(no_reuse.reuse_mode, "no-reuse")
        self.assertEqual(no_reuse.classification, "smoke")

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
