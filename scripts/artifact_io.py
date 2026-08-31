"""Dependency-free file helpers shared by experiment and audit scripts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Return a file SHA-256 without loading the whole artifact into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text_atomic(path: Path, text: str) -> None:
    """Write UTF-8 text through a same-directory temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        temporary.write(text)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Preserve the audit JSON format: UTF-8 values, two-space indentation."""
    write_text_atomic(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_json_atomic_sorted(path: Path, value: Mapping[str, Any]) -> None:
    """Write deterministic key-sorted JSON used by post-hoc model probes."""
    write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_sha256_sidecar(path: Path) -> str:
    """Write the repository's two-space SHA-256 sidecar format."""
    digest = sha256_file(path)
    write_text_atomic(
        path.with_suffix(path.suffix + ".sha256"), f"{digest}  {path.name}\n"
    )
    return digest
