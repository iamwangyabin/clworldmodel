"""The two requested experiments must differ only in real-history capacity."""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from clworldmodel.reference.dream_rehearsal import (
    ARROW_TRANSITION_CAPACITY, MemoryPairConfig,
)


class MemoryPairConfigTests(unittest.TestCase):
    def test_capacity_is_the_only_config_difference_and_budgets_are_identical(self):
        full = MemoryPairConfig()
        bounded = MemoryPairConfig(history_capacity_transitions=ARROW_TRANSITION_CAPACITY)
        a, b = full.as_dict(), bounded.as_dict()
        self.assertEqual({k for k in a if a[k] != b[k]}, {"history_capacity_transitions"})
        self.assertEqual(full.projected_budgets(), bounded.projected_budgets())
        self.assertEqual((full.history_arm, bounded.history_arm), ("full", "bounded"))

    def test_capacity_must_match_arrow_and_algorithm_overrides_fail(self):
        for value in (True, 0, 524287, 524288.0):
            with self.assertRaisesRegex(ValueError, "524288"):
                MemoryPairConfig(history_capacity_transitions=value)
        with self.assertRaisesRegex(ValueError, "official"):
            MemoryPairConfig(batch_size=4)
        with self.assertRaisesRegex(ValueError, "512-transition"):
            MemoryPairConfig(retention_block_transitions=64)
        self.assertEqual(MemoryPairConfig.smoke().batch_size, 16)

    def test_both_launches_use_one_trainer_and_equal_algorithm_hashes(self):
        with TemporaryDirectory() as td:
            manifests = []
            for arm in ("full", "bounded"):
                output = Path(td) / arm
                result = subprocess.run([
                    sys.executable, str(ROOT / "scripts/run_dream_rehearsal_memory_pair_atari.py"),
                    "--history", arm, "--output-dir", str(output), "--dry-run",
                ], cwd=td, check=True, text=True, capture_output=True)
                self.assertFalse(output.exists())
                manifests.append(json.loads(result.stdout))
            a, b = manifests
            self.assertEqual(a["command"][:2], b["command"][:2])
            self.assertEqual(a["projected_budgets"], b["projected_budgets"])
            self.assertEqual(a["memory_comparison_contract"]["shared_algorithm_and_schedule_sha256"],
                             b["memory_comparison_contract"]["shared_algorithm_and_schedule_sha256"])
            self.assertEqual(b["replay"]["transition_capacity"], 524288)
            self.assertTrue(b["claims"]["sample_capacity_matched_to_arrow"])
            self.assertFalse(b["replay"]["phase_libraries_can_retain_evicted_samples"])
            self.assertEqual(b["replay"]["ordinary_training"], "shared_retained_history")


if __name__ == "__main__":
    unittest.main()
