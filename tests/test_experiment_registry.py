"""Focused tests for the curated, text-only experiment registry."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from experiment_registry import (  # noqa: E402
    DEFAULT_RECORDS_ROOT,
    DEFAULT_REGISTRY,
    DEFAULT_RESULTS_INDEX,
    RecordValidationError,
    check_outputs,
    load_records,
    validate_record,
    write_outputs,
)


class ExperimentRegistryTests(unittest.TestCase):
    def _record(self, record_id: str = "fixture-s0") -> dict[str, object]:
        return {
            "schema_version": 1,
            "record_id": record_id,
            "run_id": "fixture_run_s0",
            "method": "Fixture Method",
            "protocol": "Fixture Protocol v1",
            "evidence_level": "smoke",
            "classification": "test_fixture",
            "status": "complete",
            "recorded_at_utc": "2026-08-31T00:00:00Z",
            "project_git": {
                "commit": "1" * 40,
                "clean": True,
                "upstream": "origin/fixture",
                "ahead": 0,
                "behind": 0,
            },
            "seed": {"id": 0, "value": 123456789},
            "task_awareness": "task_agnostic",
            "task_order": ["ALE/MsPacman-v5"],
            "completion": {
                "total_task_count": 1,
                "completed_task_count": 1,
                "completed_epochs": 1,
                "final_evaluation_performed": True,
            },
            "evaluation": {
                "metric": "raw_environment_return",
                "policy": "deterministic",
                "cohort_protocol": "fixture",
                "evaluation_transitions_enter_replay": False,
                "checkpoints": [
                    {
                        "checkpoint_id": "final",
                        "stage": "heldout_final",
                        "completed_epochs": 1,
                        "completed_task_count": 1,
                        "cohort": "heldout_final",
                        "rollouts_per_task": 2,
                        "tasks": [
                            {
                                "task_index": 0,
                                "task_name": "ALE/MsPacman-v5",
                                "raw_return_mean": 10.0,
                                "raw_return_std": 1.0,
                            }
                        ],
                    }
                ],
            },
            "headline": {"summary": "Fixture raw return: MsPacman 10.0."},
            "comparability": {
                "direct_comparison_group": None,
                "claim": "Smoke evidence only.",
                "limitations": ["Synthetic test fixture."],
            },
            "source_artifacts": [
                {
                    "role": "source_result",
                    "name": "source.json",
                    "sha256": "2" * 64,
                }
            ],
            "notes": ["No generated training artifacts are present."],
        }

    def _write_fixture(self, records_root: Path) -> Path:
        record_dir = records_root / "fixture-s0"
        record_dir.mkdir(parents=True)
        record_path = record_dir / "record.json"
        record_path.write_text(
            json.dumps(self._record(), indent=2) + "\n", encoding="utf-8"
        )
        return record_path

    def test_write_and_check_build_deterministic_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records_root = root / "records"
            self._write_fixture(records_root)
            registry_path = root / "registry.json"
            results_path = root / "RESULTS.md"

            write_outputs(records_root, registry_path, results_path)
            first_registry = registry_path.read_bytes()
            first_results = results_path.read_bytes()
            write_outputs(records_root, registry_path, results_path)

            self.assertEqual(registry_path.read_bytes(), first_registry)
            self.assertEqual(results_path.read_bytes(), first_results)
            self.assertEqual(json.loads(first_registry)["record_count"], 1)
            self.assertIn(b"records/fixture-s0/record.json", first_results)
            check_outputs(records_root, registry_path, results_path)

    def test_generated_or_heavyweight_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            records_root = Path(temporary) / "records"
            record_path = self._write_fixture(records_root)
            (record_path.parent / "model.pt").write_bytes(b"not really a weight")

            with self.assertRaisesRegex(
                RecordValidationError, "Unsupported|heavyweight|forbidden"
            ):
                load_records(records_root)

    def test_evaluation_data_leakage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            records_root = Path(temporary) / "records"
            record_path = self._write_fixture(records_root)
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record["evaluation"]["evaluation_transitions_enter_replay"] = True

            with self.assertRaisesRegex(
                RecordValidationError, "evaluation_transitions_enter_replay"
            ):
                validate_record(record, record_path)

    def test_log_excerpt_must_cite_a_source_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            records_root = Path(temporary) / "records"
            record_path = self._write_fixture(records_root)
            (record_path.parent / "evaluation.log").write_text(
                "Eval raw means: [10.0]\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(RecordValidationError, "source artifact SHA256"):
                load_records(records_root)

    def test_cross_game_raw_average_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            records_root = Path(temporary) / "records"
            record_path = self._write_fixture(records_root)
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record["headline"]["cross_game_raw_average"] = 10.0

            with self.assertRaisesRegex(RecordValidationError, "cross-game"):
                validate_record(record, record_path)

    def test_repository_indexes_are_current_and_records_are_valid(self) -> None:
        check_outputs(
            DEFAULT_RECORDS_ROOT,
            DEFAULT_REGISTRY,
            DEFAULT_RESULTS_INDEX,
        )


if __name__ == "__main__":
    unittest.main()
