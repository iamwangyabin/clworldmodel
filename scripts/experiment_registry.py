#!/usr/bin/env python3
"""Validate curated experiment records and rebuild their repository index."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDS_ROOT = ROOT / "docs" / "experiments" / "records"
DEFAULT_REGISTRY = ROOT / "docs" / "experiments" / "registry.json"
DEFAULT_RESULTS_INDEX = ROOT / "docs" / "experiments" / "RESULTS.md"

SCHEMA_VERSION = 1
RECORD_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:[a-z0-9._-]*[a-z0-9])?$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

EVIDENCE_LEVELS = {"official", "pilot", "diagnostic", "ablation", "smoke"}
RUN_STATUSES = {"complete", "partial", "stopped", "failed"}
TASK_AWARENESS_VALUES = {"task_aware", "task_agnostic", "not_applicable"}

ALLOWED_RECORD_FILENAMES = {
    "record.json",
    "evaluation.log",
    "notes.md",
}
MAX_FILE_BYTES = 256 * 1024
MAX_RECORD_BYTES = 512 * 1024
FORBIDDEN_NAME_PARTS = (
    ".ckpt",
    ".mmap",
    ".npy",
    ".npz",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
    ".tfevents",
    "checkpoint",
    "replay",
    "tensorboard",
)

REQUIRED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "record_id",
    "run_id",
    "method",
    "protocol",
    "evidence_level",
    "classification",
    "status",
    "recorded_at_utc",
    "project_git",
    "seed",
    "task_awareness",
    "task_order",
    "completion",
    "evaluation",
    "headline",
    "comparability",
    "source_artifacts",
    "notes",
}
OPTIONAL_TOP_LEVEL_FIELDS = {
    "budgets",
    "derived_metrics",
    "parameter_accounting",
    "runtime",
}


class RecordValidationError(ValueError):
    """Raised when a curated record violates the repository contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RecordValidationError(f"Cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise RecordValidationError(f"Expected a JSON object: {path}")
    return value


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecordValidationError(f"{field} must be a non-empty string")
    return value


def _require_integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RecordValidationError(f"{field} must be an integer >= {minimum}")
    return value


def _require_finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecordValidationError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RecordValidationError(f"{field} must be finite")
    return result


def _validate_project_git(project_git: Any, context: str) -> None:
    if not isinstance(project_git, dict):
        raise RecordValidationError(f"{context}.project_git must be an object")
    commit = _require_nonempty_string(
        project_git.get("commit"), f"{context}.project_git.commit"
    )
    if not COMMIT_PATTERN.fullmatch(commit):
        raise RecordValidationError(
            f"{context}.project_git.commit must be a full lowercase Git hash"
        )
    if not isinstance(project_git.get("clean"), bool):
        raise RecordValidationError(f"{context}.project_git.clean must be boolean")
    for field in ("ahead", "behind"):
        if field in project_git:
            _require_integer(project_git[field], f"{context}.project_git.{field}")


def _validate_task_result(
    task: Any,
    *,
    task_order: list[str],
    context: str,
) -> None:
    if not isinstance(task, dict):
        raise RecordValidationError(f"{context} must be an object")
    task_index = _require_integer(task.get("task_index"), f"{context}.task_index")
    if task_index >= len(task_order):
        raise RecordValidationError(f"{context}.task_index is outside task_order")
    task_name = _require_nonempty_string(task.get("task_name"), f"{context}.task_name")
    if task_name != task_order[task_index]:
        raise RecordValidationError(
            f"{context}.task_name does not match task_order[{task_index}]"
        )
    _require_finite_number(task.get("raw_return_mean"), f"{context}.raw_return_mean")
    raw_std = _require_finite_number(task.get("raw_return_std"), f"{context}.raw_return_std")
    if raw_std < 0:
        raise RecordValidationError(f"{context}.raw_return_std must be non-negative")


def _validate_evaluation(
    evaluation: Any,
    *,
    task_order: list[str],
    completed_epochs: int,
    context: str,
) -> None:
    if not isinstance(evaluation, dict):
        raise RecordValidationError(f"{context}.evaluation must be an object")
    if evaluation.get("metric") != "raw_environment_return":
        raise RecordValidationError(
            f"{context}.evaluation.metric must be raw_environment_return"
        )
    if evaluation.get("evaluation_transitions_enter_replay") is not False:
        raise RecordValidationError(
            f"{context}.evaluation.evaluation_transitions_enter_replay must be false"
        )
    _require_nonempty_string(evaluation.get("policy"), f"{context}.evaluation.policy")
    checkpoints = evaluation.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise RecordValidationError(f"{context}.evaluation.checkpoints must be non-empty")
    previous_epochs = -1
    checkpoint_ids: set[str] = set()
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        checkpoint_context = f"{context}.evaluation.checkpoints[{checkpoint_index}]"
        if not isinstance(checkpoint, dict):
            raise RecordValidationError(f"{checkpoint_context} must be an object")
        checkpoint_id = _require_nonempty_string(
            checkpoint.get("checkpoint_id"), f"{checkpoint_context}.checkpoint_id"
        )
        if checkpoint_id in checkpoint_ids:
            raise RecordValidationError(f"Duplicate checkpoint_id {checkpoint_id!r}")
        checkpoint_ids.add(checkpoint_id)
        epoch = _require_integer(
            checkpoint.get("completed_epochs"),
            f"{checkpoint_context}.completed_epochs",
        )
        if epoch < previous_epochs:
            raise RecordValidationError(f"{context}.evaluation checkpoints must be epoch ordered")
        if epoch > completed_epochs:
            raise RecordValidationError(
                f"{checkpoint_context}.completed_epochs exceeds run completion"
            )
        previous_epochs = epoch
        _require_integer(
            checkpoint.get("completed_task_count"),
            f"{checkpoint_context}.completed_task_count",
        )
        _require_nonempty_string(checkpoint.get("stage"), f"{checkpoint_context}.stage")
        _require_nonempty_string(checkpoint.get("cohort"), f"{checkpoint_context}.cohort")
        _require_integer(
            checkpoint.get("rollouts_per_task"),
            f"{checkpoint_context}.rollouts_per_task",
            minimum=1,
        )
        tasks = checkpoint.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise RecordValidationError(f"{checkpoint_context}.tasks must be non-empty")
        task_indices: list[int] = []
        for task_index, task in enumerate(tasks):
            _validate_task_result(
                task,
                task_order=task_order,
                context=f"{checkpoint_context}.tasks[{task_index}]",
            )
            task_indices.append(int(task["task_index"]))
        if task_indices != sorted(set(task_indices)):
            raise RecordValidationError(
                f"{checkpoint_context}.tasks must have unique increasing task indices"
            )


def validate_record(record: dict[str, Any], record_path: Path) -> None:
    """Validate one record without consulting any external run directory."""

    context = _portable_path(record_path)
    missing = REQUIRED_TOP_LEVEL_FIELDS - record.keys()
    if missing:
        raise RecordValidationError(f"{context} is missing fields: {sorted(missing)}")
    unknown = record.keys() - REQUIRED_TOP_LEVEL_FIELDS - OPTIONAL_TOP_LEVEL_FIELDS
    if unknown:
        raise RecordValidationError(f"{context} has unknown fields: {sorted(unknown)}")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise RecordValidationError(f"{context}.schema_version must be {SCHEMA_VERSION}")

    record_id = _require_nonempty_string(record.get("record_id"), f"{context}.record_id")
    if not RECORD_ID_PATTERN.fullmatch(record_id):
        raise RecordValidationError(f"{context}.record_id has an invalid portable form")
    if record_path.parent.name != record_id:
        raise RecordValidationError(
            f"{context}.record_id must match its parent directory {record_path.parent.name!r}"
        )
    for field in ("run_id", "method", "protocol", "classification", "recorded_at_utc"):
        _require_nonempty_string(record.get(field), f"{context}.{field}")
    if record["evidence_level"] not in EVIDENCE_LEVELS:
        raise RecordValidationError(f"{context}.evidence_level is unsupported")
    if record["status"] not in RUN_STATUSES:
        raise RecordValidationError(f"{context}.status is unsupported")
    if record["task_awareness"] not in TASK_AWARENESS_VALUES:
        raise RecordValidationError(f"{context}.task_awareness is unsupported")

    _validate_project_git(record["project_git"], context)

    seed = record["seed"]
    if not isinstance(seed, dict):
        raise RecordValidationError(f"{context}.seed must be an object")
    _require_integer(seed.get("id"), f"{context}.seed.id")
    _require_integer(seed.get("value"), f"{context}.seed.value")

    task_order_value = record["task_order"]
    if not isinstance(task_order_value, list) or not task_order_value:
        raise RecordValidationError(f"{context}.task_order must be a non-empty list")
    task_order = [
        _require_nonempty_string(task, f"{context}.task_order[{index}]")
        for index, task in enumerate(task_order_value)
    ]
    if len(task_order) != len(set(task_order)):
        raise RecordValidationError(f"{context}.task_order contains duplicates")

    completion = record["completion"]
    if not isinstance(completion, dict):
        raise RecordValidationError(f"{context}.completion must be an object")
    total_task_count = _require_integer(
        completion.get("total_task_count"), f"{context}.completion.total_task_count", minimum=1
    )
    completed_task_count = _require_integer(
        completion.get("completed_task_count"),
        f"{context}.completion.completed_task_count",
    )
    completed_epochs = _require_integer(
        completion.get("completed_epochs"), f"{context}.completion.completed_epochs"
    )
    if total_task_count > len(task_order):
        raise RecordValidationError(f"{context}.completion.total_task_count exceeds task_order")
    if completed_task_count > total_task_count:
        raise RecordValidationError(
            f"{context}.completion.completed_task_count exceeds total_task_count"
        )
    if not isinstance(completion.get("final_evaluation_performed"), bool):
        raise RecordValidationError(
            f"{context}.completion.final_evaluation_performed must be boolean"
        )

    _validate_evaluation(
        record["evaluation"],
        task_order=task_order,
        completed_epochs=completed_epochs,
        context=context,
    )

    headline = record["headline"]
    if not isinstance(headline, dict):
        raise RecordValidationError(f"{context}.headline must be an object")
    _require_nonempty_string(headline.get("summary"), f"{context}.headline.summary")
    if "cross_game_raw_average" in headline:
        raise RecordValidationError(
            f"{context}.headline must not persist a cross-game raw-return average"
        )

    comparability = record["comparability"]
    if not isinstance(comparability, dict):
        raise RecordValidationError(f"{context}.comparability must be an object")
    _require_nonempty_string(comparability.get("claim"), f"{context}.comparability.claim")
    limitations = comparability.get("limitations")
    if not isinstance(limitations, list) or not all(
        isinstance(item, str) and item.strip() for item in limitations
    ):
        raise RecordValidationError(
            f"{context}.comparability.limitations must be a list of non-empty strings"
        )

    artifacts = record["source_artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise RecordValidationError(f"{context}.source_artifacts must be non-empty")
    for artifact_index, artifact in enumerate(artifacts):
        artifact_context = f"{context}.source_artifacts[{artifact_index}]"
        if not isinstance(artifact, dict):
            raise RecordValidationError(f"{artifact_context} must be an object")
        _require_nonempty_string(artifact.get("role"), f"{artifact_context}.role")
        _require_nonempty_string(artifact.get("name"), f"{artifact_context}.name")
        checksum = _require_nonempty_string(
            artifact.get("sha256"), f"{artifact_context}.sha256"
        )
        if not SHA256_PATTERN.fullmatch(checksum):
            raise RecordValidationError(f"{artifact_context}.sha256 is invalid")
        repository_path = artifact.get("repository_path")
        if repository_path is not None:
            repository_path = _require_nonempty_string(
                repository_path, f"{artifact_context}.repository_path"
            )
            candidate = Path(repository_path)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise RecordValidationError(
                    f"{artifact_context}.repository_path must be repository relative"
                )
            source_path = (ROOT / candidate).resolve()
            try:
                source_path.relative_to(ROOT)
            except ValueError as error:
                raise RecordValidationError(
                    f"{artifact_context}.repository_path escapes the repository"
                ) from error
            if not source_path.is_file():
                raise RecordValidationError(
                    f"{artifact_context}.repository_path does not exist: {repository_path}"
                )
            if _sha256(source_path) != checksum:
                raise RecordValidationError(
                    f"{artifact_context}.repository_path checksum does not match"
                )

    notes = record["notes"]
    if not isinstance(notes, list) or not all(
        isinstance(item, str) and item.strip() for item in notes
    ):
        raise RecordValidationError(f"{context}.notes must be a list of non-empty strings")


def _validate_evidence_source_hashes(record: dict[str, Any], record_dir: Path) -> None:
    excerpt_path = record_dir / "evaluation.log"
    if not excerpt_path.is_file():
        return
    excerpt_hashes = set(
        re.findall(r"\b[0-9a-f]{64}\b", excerpt_path.read_text(encoding="utf-8"))
    )
    source_hashes = {str(artifact["sha256"]) for artifact in record["source_artifacts"]}
    if not excerpt_hashes.intersection(source_hashes):
        raise RecordValidationError(
            f"{excerpt_path} must cite at least one matching source artifact SHA256"
        )


def _validate_record_directory(record_dir: Path) -> list[dict[str, Any]]:
    if record_dir.is_symlink():
        raise RecordValidationError(f"Symlinked record directories are not allowed: {record_dir}")
    files: list[dict[str, Any]] = []
    total_bytes = 0
    for path in sorted(record_dir.iterdir(), key=lambda item: item.name):
        if path.is_dir() or path.is_symlink():
            raise RecordValidationError(f"Nested directories and symlinks are not allowed: {path}")
        lowered = path.name.lower()
        if path.name not in ALLOWED_RECORD_FILENAMES:
            raise RecordValidationError(f"Unsupported curated evidence filename: {path}")
        if any(part in lowered for part in FORBIDDEN_NAME_PARTS):
            raise RecordValidationError(f"Generated or heavyweight artifact is forbidden: {path}")
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise RecordValidationError(
                f"Curated evidence exceeds {MAX_FILE_BYTES} bytes: {path} ({size})"
            )
        total_bytes += size
        payload = path.read_bytes()
        if b"\x00" in payload:
            raise RecordValidationError(f"Binary content is forbidden in curated evidence: {path}")
        files.append(
            {
                "path": _portable_path(path),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": size,
            }
        )
    if total_bytes > MAX_RECORD_BYTES:
        raise RecordValidationError(
            f"Curated record exceeds {MAX_RECORD_BYTES} total bytes: {record_dir}"
        )
    if not any(item["path"].endswith("/record.json") for item in files):
        raise RecordValidationError(f"Missing record.json under {record_dir}")
    return files


def load_records(records_root: Path) -> list[tuple[dict[str, Any], Path, list[dict[str, Any]]]]:
    """Load and validate every immediate record directory."""

    if not records_root.is_dir():
        raise RecordValidationError(f"Records root does not exist: {records_root}")
    entries: list[tuple[dict[str, Any], Path, list[dict[str, Any]]]] = []
    seen_run_ids: set[str] = set()
    for record_dir in sorted(records_root.iterdir(), key=lambda item: item.name):
        if record_dir.name.startswith("."):
            continue
        if not record_dir.is_dir():
            raise RecordValidationError(
                f"Only record directories are allowed under {records_root}"
            )
        record_path = record_dir / "record.json"
        files = _validate_record_directory(record_dir)
        record = _json_object(record_path)
        validate_record(record, record_path)
        _validate_evidence_source_hashes(record, record_dir)
        run_id = str(record["run_id"])
        if run_id in seen_run_ids:
            raise RecordValidationError(f"Duplicate run_id in registry: {run_id}")
        seen_run_ids.add(run_id)
        entries.append((record, record_path, files))
    if not entries:
        raise RecordValidationError(f"No experiment records found under {records_root}")
    return entries


def build_registry(
    entries: list[tuple[dict[str, Any], Path, list[dict[str, Any]]]],
) -> dict[str, Any]:
    """Build the deterministic compact machine index."""

    records: list[dict[str, Any]] = []
    for record, record_path, files in entries:
        completion = record["completion"]
        latest_checkpoint = record["evaluation"]["checkpoints"][-1]
        records.append(
            {
                "record_id": record["record_id"],
                "run_id": record["run_id"],
                "method": record["method"],
                "protocol": record["protocol"],
                "evidence_level": record["evidence_level"],
                "classification": record["classification"],
                "status": record["status"],
                "task_awareness": record["task_awareness"],
                "seed": record["seed"],
                "completion": {
                    "completed_task_count": completion["completed_task_count"],
                    "total_task_count": completion["total_task_count"],
                    "completed_epochs": completion["completed_epochs"],
                    "final_evaluation_performed": completion[
                        "final_evaluation_performed"
                    ],
                },
                "headline": record["headline"]["summary"],
                "latest_checkpoint": {
                    "checkpoint_id": latest_checkpoint["checkpoint_id"],
                    "stage": latest_checkpoint["stage"],
                    "completed_epochs": latest_checkpoint["completed_epochs"],
                    "tasks": latest_checkpoint["tasks"],
                },
                "comparability_claim": record["comparability"]["claim"],
                "record_path": _portable_path(record_path),
                "repository_files": files,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "curated_experiment_registry",
        "generated_from": "docs/experiments/records/*/record.json",
        "record_count": len(records),
        "storage_policy": {
            "included": [
                "structured provenance",
                "raw per-task evaluation summaries",
                "derived metric summaries with formulas or source references",
                "small human-readable evaluation log excerpts",
                "source artifact SHA256 checksums",
            ],
            "excluded": [
                "model weights and inference snapshots",
                "training and optimizer checkpoints",
                "replay storage",
                "TensorBoard event files",
                "videos and full generated run directories",
                "full training logs",
            ],
        },
        "records": records,
    }


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_results_index(registry: dict[str, Any]) -> str:
    """Render a compact human index; detailed interpretation lives in README."""

    lines = [
        "# Curated Experiment Results",
        "",
        "This file is generated by `scripts/experiment_registry.py`. Do not edit it",
        "by hand. See [README.md](README.md) for storage rules and interpretation.",
        "",
        "> Raw returns are listed per task. They must not be averaged across games.",
        "> Rows with different protocols, evaluator cohorts, task awareness, or budgets",
        "> are not direct rankings.",
        "",
        "| Record | Method | Evidence | Status | Scope | Task identity | Headline |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for entry in registry["records"]:
        completion = entry["completion"]
        scope = f"{completion['completed_task_count']}/{completion['total_task_count']} tasks"
        task_identity = {
            "task_aware": "exposed",
            "task_agnostic": "hidden",
            "not_applicable": "n/a",
        }[entry["task_awareness"]]
        # RESULTS.md is part of the fixed repository layout. Keep its links
        # portable even when tests validate records from a temporary root.
        record_link = Path("records") / entry["record_id"] / "record.json"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"[`{_markdown_escape(entry['record_id'])}`]({record_link.as_posix()})",
                    _markdown_escape(entry["method"]),
                    _markdown_escape(entry["evidence_level"]),
                    _markdown_escape(entry["status"]),
                    scope,
                    task_identity,
                    _markdown_escape(entry["headline"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"Registry contains **{registry['record_count']}** curated run records.",
            "",
        ]
    )
    return "\n".join(lines)


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def write_outputs(records_root: Path, registry_path: Path, results_path: Path) -> None:
    entries = load_records(records_root)
    registry = build_registry(entries)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(_canonical_json(registry), encoding="utf-8")
    results_path.write_text(render_results_index(registry), encoding="utf-8")


def check_outputs(records_root: Path, registry_path: Path, results_path: Path) -> None:
    entries = load_records(records_root)
    registry = build_registry(entries)
    expected_registry = _canonical_json(registry)
    expected_results = render_results_index(registry)
    problems: list[str] = []
    for path, expected in (
        (registry_path, expected_registry),
        (results_path, expected_results),
    ):
        if not path.is_file():
            problems.append(f"missing {path}")
        elif path.read_text(encoding="utf-8") != expected:
            problems.append(f"stale {path}")
    if problems:
        joined = "; ".join(problems)
        raise RecordValidationError(
            f"Experiment registry check failed: {joined}. Run with mode 'write'."
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("write", "check"))
    parser.add_argument("--records-root", type=Path, default=DEFAULT_RECORDS_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_INDEX)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.mode == "write":
            write_outputs(args.records_root, args.registry, args.results)
        else:
            check_outputs(args.records_root, args.registry, args.results)
    except RecordValidationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"experiment registry {args.mode} passed for {_portable_path(args.records_root)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
