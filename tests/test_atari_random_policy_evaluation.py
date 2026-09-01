"""CPU-only contracts for the shared Atari random-policy reference."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_atari_random_policy.py"
SPEC = importlib.util.spec_from_file_location("atari_random_reference", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AtariRandomPolicyEvaluationTests(unittest.TestCase):
    def test_seed_streams_are_stable_disjoint_and_match_fixed_eval_children(self) -> None:
        table_a = MODULE._task_seed_table(123456789)
        table_b = MODULE._task_seed_table(123456789)
        self.assertEqual(table_a, table_b)

        children = np.random.SeedSequence(123456789).spawn(3)
        validation_rng = np.random.default_rng(children[1])
        final_rng = np.random.default_rng(children[2])
        expected_validation = [
            int(validation_rng.integers(0, 2**32, dtype=np.uint64))
            for _ in MODULE.TASKS
        ]
        expected_final = [
            int(final_rng.integers(0, 2**32, dtype=np.uint64))
            for _ in MODULE.TASKS
        ]
        self.assertEqual(table_a["validation"]["environment"], expected_validation)
        self.assertEqual(table_a["heldout_final"]["environment"], expected_final)
        self.assertNotEqual(
            table_a["validation"]["policy"], table_a["heldout_final"]["policy"]
        )
        self.assertTrue(
            set(table_a["validation"]["policy"]).isdisjoint(expected_validation)
        )

    def test_episode_returns_match_vendored_start_end_semantics(self) -> None:
        rewards = np.asarray([0, 1, 2, 0, 4, 5, 0], dtype=np.float32)
        continuations = np.asarray([1, 1, 0, 1, 1, 0, 1], dtype=np.float32)
        resets = np.asarray([1, 0, 0, 1, 0, 0, 1], dtype=np.float32)
        self.assertEqual(
            MODULE._episode_returns(rewards, continuations, resets), [3.0, 9.0]
        )

    def test_summary_uses_seed_median_and_iqr(self) -> None:
        records = [
            {"cohort": "validation", "task": "task", "mean": value}
            for value in (1.0, 2.0, 3.0, 4.0, 100.0)
        ]
        summary = MODULE._summaries(records)["validation"]["task"]
        self.assertEqual(summary["median_seed_mean"], 3.0)
        self.assertEqual(summary["iqr_seed_mean"], [2.0, 4.0])

    def test_dry_run_is_cpu_only_and_shared_across_methods(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--output-dir",
                "/tmp/not-created",
                "--seed-indices",
                "0",
                "--dry-run",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        manifest = json.loads(result.stdout)
        self.assertFalse(manifest["gpu_required"])
        self.assertTrue(manifest["shared_across_methods"])
        self.assertEqual(manifest["metric"], "raw_episode_return")
        self.assertEqual(manifest["rollouts_target_per_task_cohort_seed"], 16)
        self.assertEqual(manifest["environment"]["action_count"], 18)
        self.assertEqual(manifest["seed_roots"], [123456789])


if __name__ == "__main__":
    unittest.main()
