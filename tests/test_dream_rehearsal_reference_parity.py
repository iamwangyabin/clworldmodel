"""Run pinned-reference checks in a fresh process, isolated from ARROW imports."""

from importlib import metadata
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PinnedReferenceParityTests(unittest.TestCase):
    def test_reference_cpu_fixtures(self):
        for name, version in (("torch", "2.4.1"), ("gym", "0.22.0"), ("ruamel.yaml", "0.17.21")):
            try:
                installed = metadata.version(name).split("+")[0]
            except metadata.PackageNotFoundError:
                installed = None
            if installed != version:
                self.skipTest("Use the isolated requirements/dream_rehearsal_official.txt environment")
        result = subprocess.run(
            [sys.executable, str(ROOT / "tests/fixtures/dream_rehearsal_reference_cpu.py")],
            cwd=ROOT, text=True, capture_output=True, timeout=180,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
