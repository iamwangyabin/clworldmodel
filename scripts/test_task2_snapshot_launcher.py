"""Contracts for the Task-2 snapshot acquisition launcher."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_karrow_task2_from_snapshot import _task2_config  # noqa: E402


class Task2SnapshotLauncherTests(unittest.TestCase):
    def test_task_config_selects_boxing_and_resets_only_diagnostic_state(self) -> None:
        source = {
            "epochs": 540,
            "dinov3_model_path": "/old/dinov3",
            "shared_core_mode": "freeze_after_first_task",
            "residual_consolidation": "replay_functional",
            "residual_consolidation_batches": 16,
            "residual_consolidation_imagination_horizon": 8,
            "residual_consolidation_gradient_power": 2.0,
            "residual_consolidation_min_plasticity": 0.01,
            "residual_consolidation_anchor_loss_scale": 1.0,
            "esc": {
                "env_schedule_type": "SequentialEnvironments",
                "env_configs": [
                    {"name": "ALE/MsPacman-v5", "kwargs": {}, "rew_scale": 1.0},
                    {"name": "ALE/Boxing-v5", "kwargs": {}, "rew_scale": 1.0},
                ],
                "kwargs": {"swap_sched": 90},
            },
        }
        original = copy.deepcopy(source)
        result = _task2_config(
            source,
            task_index=1,
            epochs=90,
            dinov3_model_path=Path("/new/dinov3"),
        )

        self.assertEqual(result["epochs"], 90)
        self.assertEqual(result["esc"]["kwargs"]["swap_sched"], 90)
        self.assertEqual(
            [task["name"] for task in result["esc"]["env_configs"]],
            ["ALE/Boxing-v5"],
        )
        self.assertEqual(result["shared_core_mode"], "snapshot_adaptation")
        self.assertEqual(result["residual_consolidation"], "none")
        self.assertEqual(result["dinov3_model_path"], "/new/dinov3")
        self.assertEqual(source, original)


if __name__ == "__main__":
    unittest.main()
