#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Launch D-AutoRoute: original D with independent MLPs and label-free inference.

This independent entry point fixes the research topology. It composes the
existing launcher/trainer rather than copying them. Training remains task-aware;
only interaction and evaluation select routes from first-frame reconstruction.
All relative paths resolve from the repository root, independent of the working
directory. Absolute paths and user-home paths are preserved.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from run_evolving_atomic_rssm import (
    D_AUTOROUTE_METHOD as METHOD_KEY,
    D_AUTOROUTE_PROTOCOL as PROTOCOL,
    PRIVATE_MLP_AUTOROUTE_BEHAVIOR,
    ROOT,
    SEEDS,
    main as _launch,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--seed", type=int, choices=range(len(SEEDS)), default=0)
    parser.add_argument("--classification", choices=("pilot",), default="pilot")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--replay-mmap-root", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cpu-threads", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None, *, benchmark: str = "atari") -> int:
    args = _parser().parse_args(argv)
    def resolved(path: Path, *, follow_symlinks: bool = True) -> str:
        expanded = path.expanduser()
        rooted = expanded if expanded.is_absolute() else ROOT / expanded
        # A venv Python is often a symlink; dereferencing it discards the venv.
        return str(rooted.resolve() if follow_symlinks else Path(os.path.abspath(rooted)))

    command = [
        "--benchmark", benchmark,
        "--task-order", "arrow-original-six",
        "--task0-profile", "fixed_v1",
        "--prediction-head-profile", "shared_distilled",
        "--adaptive-qfp-compression",
        "--behavior-profile", PRIVATE_MLP_AUTOROUTE_BEHAVIOR,
        "--seed", str(args.seed),
        "--classification", args.classification,
        "--python", resolved(args.python, follow_symlinks=False),
        "--cpu-threads", str(args.cpu_threads),
    ]
    for name in ("output_dir", "replay_mmap_root"):
        value = getattr(args, name)
        if value is not None:
            command.extend(("--" + name.replace("_", "-"), resolved(value)))
    if args.dry_run:
        command.append("--dry-run")
    return _launch(command)


if __name__ == "__main__":
    raise SystemExit(main())
