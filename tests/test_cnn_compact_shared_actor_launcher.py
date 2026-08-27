"""Launcher contracts for the compact RSSM/shared-actor pilot."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_cnn_compact_shared_actor_incremental import (  # noqa: E402
    ADAPTER_SIZES,
    DISTILLATION,
    EXPECTED_TASKS,
    _budgets,
    _incremental_config,
)


class CnnCompactSharedActorLauncherTests(unittest.TestCase):
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

    def test_config_is_compact_and_keeps_one_shared_actor(self) -> None:
        source = self._source()
        original = copy.deepcopy(source)
        config = _incremental_config(source, epochs_after_task1=180)

        self.assertEqual(source, original)
        self.assertEqual(config["epochs"], 270)
        self.assertEqual(
            config["continual_method"], "cnn_compact_shared_actor_arrow"
        )
        self.assertEqual(
            (
                config["task_lora_recurrent_rank"],
                config["task_lora_representation_rank"],
                config["task_lora_transition_rank"],
                config["task_recurrent_output_adapter_features"],
            ),
            (
                ADAPTER_SIZES["recurrent_lora_rank"],
                ADAPTER_SIZES["representation_lora_rank"],
                ADAPTER_SIZES["transition_lora_rank"],
                ADAPTER_SIZES["gru_output_adapter_features"],
            ),
        )
        self.assertFalse(config["fresh_ac"])
        self.assertTrue(config["shared_actor_imagination_distillation"])
        self.assertEqual(
            config["shared_core_mode"],
            "task1_frozen_projector_compact_rssm",
        )

    def test_extra_imagination_compute_is_explicit(self) -> None:
        config = _incremental_config(self._source(), epochs_after_task1=180)
        budgets = _budgets(config, epochs_after_task1=180)
        self.assertEqual(budgets["new_actor_critic_optimizer_updates"], 72_000)
        self.assertEqual(budgets["shared_actor_distillation_batches"], 18_000)
        self.assertEqual(budgets["shared_actor_distilled_states"], 36_864_000)
        self.assertEqual(budgets["shared_actor_burnin_state_uses"], 36_864_000)
        self.assertEqual(budgets["old_real_replay_samples"], 0)
        self.assertFalse(
            budgets["additional_imagination_compute_matched_to_prior_method"]
        )
        self.assertEqual(DISTILLATION["interval"], 4)

    def test_smoke_starts_at_task2_without_retraining_task1(self) -> None:
        config = _incremental_config(self._source(), epochs_after_task1=1)
        self.assertEqual(config["epochs"], 91)


if __name__ == "__main__":
    unittest.main()
