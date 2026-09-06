"""Check that the ten archived baselines can be audited without external runs."""

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from report_coinrun_baselines import (  # noqa: E402
    RECORDS,
    REPORT,
    load_campaign,
    read_run,
    render_report,
)


class CoinRunBaselineArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = load_campaign()

    def test_all_seeds_and_raw_episodes_are_preserved(self):
        self.assertEqual(len(self.runs), 10)
        self.assertEqual(sum(run["episodes"] for run in self.runs), 844879)
        for run in self.runs:
            self.assertEqual(len(run["matrix"]), 55)
            self.assertEqual(run["episodes"], run["record"]["evaluation"]["total_raw_episode_count"])
            self.assertEqual(run["record"]["evidence_level"], "diagnostic")

    def test_source_finalization_failures_are_not_hidden(self):
        for run in self.runs:
            record = run["record"]
            completion = record["completion"]
            strict = record["seed"]["id"] == 4
            self.assertEqual(completion["strict_finalization_verified"], strict)
            self.assertEqual(completion["source_launcher_state"], "completed" if strict else "running")
            self.assertEqual(completion["source_launcher_exit_code"], 0 if strict else None)
            self.assertFalse(record["runtime"]["checkpoint_semantics"]["model_weights_present"])
            self.assertFalse(completion["post_training_evaluation_available"])

    def test_report_is_reproducible(self):
        self.assertEqual(REPORT.read_text(encoding="utf-8"), render_report(self.runs))

    def test_corrupted_episode_cannot_retain_old_statistics(self):
        source = RECORDS / "coinrun-arrow-ar50-original-s0-20260903"
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / source.name
            shutil.copytree(source, target)
            path = target / "evaluation.log"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[1]["returns"][0] = 10 - rows[1]["returns"][0]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                read_run(target / "record.json")
            record_path = target / "record.json"
            record = json.loads(record_path.read_text())
            for item in record["source_artifacts"]:
                if item["role"] == "curated_raw_returns":
                    item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            record_path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "mean/std/count mismatch"):
                read_run(record_path)


if __name__ == "__main__":
    unittest.main()
