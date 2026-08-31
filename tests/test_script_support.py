from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from artifact_io import (  # noqa: E402
    sha256_file,
    write_json_atomic,
    write_json_atomic_sorted,
    write_sha256_sidecar,
    write_text_atomic,
)
from launcher_support import run_and_tee, write_json  # noqa: E402


class ArtifactIoTests(unittest.TestCase):
    def test_atomic_writers_preserve_existing_artifact_formats(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            text_path = root / "nested" / "artifact.txt"
            write_text_atomic(text_path, "payload\n")
            self.assertEqual(text_path.read_text(encoding="utf-8"), "payload\n")

            json_path = root / "audit.json"
            payload = {"z": "保留", "a": 1}
            write_json_atomic(json_path, payload)
            self.assertEqual(
                json_path.read_text(encoding="utf-8"),
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            )

            sorted_path = root / "probe.json"
            write_json_atomic_sorted(sorted_path, payload)
            self.assertEqual(
                sorted_path.read_text(encoding="utf-8"),
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
            )

    def test_checksum_and_sidecar_match_existing_format(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "artifact.bin"
            path.write_bytes(b"continual-world-model")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()

            self.assertEqual(sha256_file(path), expected)
            self.assertEqual(write_sha256_sidecar(path), expected)
            self.assertEqual(
                path.with_suffix(".bin.sha256").read_text(encoding="ascii"),
                f"{expected}  {path.name}\n",
            )


class LauncherSupportTests(unittest.TestCase):
    def test_launcher_json_preserves_existing_format(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "launch.json"
            payload = {"command": ["python", "train.py"], "dry_run": True}
            write_json(path, payload)
            self.assertEqual(
                path.read_text(encoding="utf-8"), json.dumps(payload, indent=2) + "\n"
            )

    def test_run_and_tee_mirrors_combined_output(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            log_path = root / "train.log"
            captured = io.StringIO()
            command = [
                sys.executable,
                "-u",
                "-c",
                "import sys; print('stdout'); print('stderr', file=sys.stderr)",
            ]
            with contextlib.redirect_stdout(captured):
                return_code = run_and_tee(
                    command,
                    cwd=root,
                    env=os.environ.copy(),
                    log_path=log_path,
                )

            self.assertEqual(return_code, 0)
            self.assertEqual(captured.getvalue(), "stdout\nstderr\n")
            self.assertEqual(log_path.read_text(encoding="utf-8"), "stdout\nstderr\n")


if __name__ == "__main__":
    unittest.main()
