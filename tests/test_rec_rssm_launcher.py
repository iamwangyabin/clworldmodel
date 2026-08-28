"""Launcher contracts for the Task1-seeded REC-RSSM pilot."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_cnn_mechanism_bank_incremental import (  # noqa: E402
    EXPECTED_TASKS,
    MECHANISM_PARAMETERS_PER_LATER_TASK,
    REC_CONSOLIDATION_BATCHES,
    REC_MAX_VALIDATION_DROP,
    REC_MIN_CONTRIBUTION,
    REC_NUM_ATOMS,
    REC_PROTOCOL,
    REC_REUSE_PROBE_EPOCHS,
    REC_ROUTE_LR_SCALE,
    REC_ROUTE_PARAMETERS_FOR_TASK3,
    _absolute_path_preserving_symlinks,
    _incremental_config,
    _parser,
)


class RecRssmLauncherTests(unittest.TestCase):
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

    def test_rec_profile_is_lossless_capacity_matched_and_budget_matched(self) -> None:
        source = self._source()
        original = copy.deepcopy(source)
        config = _incremental_config(
            source,
            epochs_after_task1=180,
            method_profile="rec-rssm",
        )

        self.assertEqual(source, original)
        self.assertEqual(config["continual_method"], "rec_rssm_arrow")
        self.assertEqual(config["epochs"], 270)
        self.assertEqual(config["task_mechanism_num_atoms"], REC_NUM_ATOMS)
        self.assertEqual(
            config["task_mechanism_reuse_probe_epochs"], REC_REUSE_PROBE_EPOCHS
        )
        self.assertEqual(config["task_mechanism_route_lr_scale"], REC_ROUTE_LR_SCALE)
        self.assertEqual(
            config["task_mechanism_consolidation_batches"],
            REC_CONSOLIDATION_BATCHES,
        )
        self.assertEqual(
            config["task_mechanism_min_contribution"], REC_MIN_CONTRIBUTION
        )
        self.assertEqual(
            config["task_mechanism_max_validation_drop"],
            REC_MAX_VALIDATION_DROP,
        )
        self.assertEqual(
            (
                config["task_mechanism_recurrent_width"] // REC_NUM_ATOMS,
                config["task_mechanism_representation_width"] // REC_NUM_ATOMS,
                config["task_mechanism_transition_width"] // REC_NUM_ATOMS,
            ),
            (128, 128, 64),
        )
        self.assertEqual(MECHANISM_PARAMETERS_PER_LATER_TASK, 3_816_192)
        self.assertEqual(REC_ROUTE_PARAMETERS_FOR_TASK3, 12)
        self.assertTrue(config["task_mechanism_reuse"])
        self.assertFalse(config["shared_actor_imagination_distillation"])
        self.assertEqual(config["residual_consolidation"], "none")
        self.assertEqual(
            REC_PROTOCOL,
            "REC-RSSM-ARROW-v1-Task1SnapshotSeeded-Atari-TaskAware",
        )

    def test_rec_profile_rejects_no_reuse_and_unnamed_settings(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires atom reuse"):
            _incremental_config(
                self._source(),
                epochs_after_task1=180,
                reuse_enabled=False,
                method_profile="rec-rssm",
            )
        with self.assertRaisesRegex(ValueError, "Unknown method profile"):
            _incremental_config(
                self._source(),
                epochs_after_task1=180,
                method_profile="rec-rssm-wide",
            )

    def test_dedicated_wrapper_defaults_to_rec_profile(self) -> None:
        args = _parser("rec-rssm").parse_args(
            ["--task1-boundary-snapshot", "task1.pt"]
        )
        self.assertEqual(args.method_profile, "rec-rssm")
        self.assertEqual(args.reuse_mode, "reuse")

    def test_launcher_preserves_virtualenv_interpreter_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            base_python = root / "base-python"
            base_python.touch()
            venv_python = root / ".venv" / "bin" / "python"
            venv_python.parent.mkdir(parents=True)
            venv_python.symlink_to(base_python)

            selected = _absolute_path_preserving_symlinks(venv_python)

        self.assertEqual(selected, venv_python)
        self.assertNotEqual(selected, base_python)


if __name__ == "__main__":
    unittest.main()
