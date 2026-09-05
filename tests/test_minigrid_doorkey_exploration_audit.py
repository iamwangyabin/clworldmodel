"""Dry-run checks for the DoorKey exploration diagnostic."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DoorKeyExplorationAuditTests(unittest.TestCase):
    def test_dry_run_is_explicitly_not_agent_performance(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/audit_minigrid_doorkey_exploration.py",
                "--dry-run",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        manifest = json.loads(result.stdout)
        self.assertEqual(manifest["evidence_level"], "diagnostic")
        self.assertIn("not agent performance", manifest["claim_scope"])
        self.assertFalse(manifest["uses_training_or_evaluation_data"])
        self.assertFalse(manifest["uses_optimizer_updates"])
        self.assertEqual(manifest["source_reference"]["executable_size"], 8)
        self.assertEqual(
            manifest["geometries"],
            ["released_source_8x8", "paper_label_9x9"],
        )


if __name__ == "__main__":
    unittest.main()
