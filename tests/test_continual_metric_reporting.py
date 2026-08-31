"""Focused tests for structured ARROW-style run reporting."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_continual_metrics import (  # noqa: E402
    DEFAULT_NORMALIZATION,
    build_run_report,
)


class ContinualMetricReportingTests(unittest.TestCase):
    def _write_run(self, root: Path, *, omit_final_boundary: bool = False) -> Path:
        run_dir = root / "run"
        run_dir.mkdir()
        config = {
            "seed": 7,
            "epochs": 2,
            "n_sync": 1,
            "gen_seq_len": 1,
            "env_repeat": 4,
            "steps_per_batch": 1,
            "ac_train_steps": 1,
            "mb_t_size": 1,
            "mb_n_size": 1,
            "data_n_max": 1,
            "data_t": 2,
            "replay_buffers": [
                {"rb_type": "FifoReplay"},
                {"rb_type": "LongTermReplay"},
            ],
            "esc": {
                "kwargs": {"swap_sched": 1},
                "env_configs": [
                    {"name": "ALE/MsPacman-v5", "rew_scale": 0.05},
                    {"name": "ALE/Boxing-v5", "rew_scale": 1.0},
                ],
            },
        }
        launch = {
            "method": "fixture-method",
            "classification": "test",
            "seed_id": 0,
            "fifo_slots": 1,
            "ltdm_slots": 1,
            "sequence_length": 2,
        }
        (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (run_dir / "launch.json").write_text(json.dumps(launch), encoding="utf-8")
        blocks = [
            """Eval for epoch:  0
Eval means: [0.62, 0.51]
Eval stds: [0.0, 0.0]
Eval raw means: [12.4, 0.51]
Eval raw stds: [0.0, 0.0]""",
            """Eval for epoch:  1
Eval means: [77.015, 0.51]
Eval stds: [1.0, 1.0]
Eval raw means: [1540.3, 0.51]
Eval raw stds: [20.0, 1.0]""",
        ]
        if not omit_final_boundary:
            blocks.append(
                """Eval for epoch:  2
Eval means: [50.0, 90.27]
Eval stds: [2.0, 2.0]
Eval raw means: [1000.0, 90.27]
Eval raw stds: [40.0, 2.0]"""
            )
        (run_dir / "train.log").write_text("\n".join(blocks), encoding="utf-8")
        return run_dir

    def test_report_preserves_raw_values_and_computes_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = build_run_report(
                self._write_run(Path(temporary)), DEFAULT_NORMALIZATION
            )

        final_q0 = (1000.0 - 12.4) / (1540.3 - 12.4)
        self.assertAlmostEqual(report["metrics"]["forgetting"], (1 - final_q0) / 2)
        self.assertAlmostEqual(report["metrics"]["acc"], (final_q0 + 1) / 2)
        self.assertAlmostEqual(report["metrics"]["min_acc"], final_q0)
        self.assertAlmostEqual(report["metrics"]["wc_acc"], (final_q0 + 1) / 2)
        self.assertIsNone(report["metrics"]["forward_transfer"])
        self.assertEqual(
            report["evaluation_checkpoints"][-1]["raw_return_mean"],
            [1000.0, 90.27],
        )
        self.assertEqual(
            report["evaluation_protocol"]["task_completion_epochs"], [1, 2]
        )

    def test_missing_task_boundary_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._write_run(
                Path(temporary), omit_final_boundary=True
            )
            with self.assertRaisesRegex(ValueError, "task completion epochs"):
                build_run_report(run_dir, DEFAULT_NORMALIZATION)


if __name__ == "__main__":
    unittest.main()
