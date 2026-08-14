"""Git provenance checks shared by reproducible training launchers."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_state(root: Path) -> dict[str, int | str | bool]:
    """Return the local commit and its relation to the configured upstream."""
    commit = _git(root, "rev-parse", "HEAD")
    dirty = bool(_git(root, "status", "--porcelain=v1", "--untracked-files=normal"))
    upstream = _git(root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    behind_ahead = _git(root, "rev-list", "--left-right", "--count", f"{upstream}...HEAD")
    behind, ahead = (int(value) for value in behind_ahead.split())
    return {
        "commit": commit,
        "dirty": dirty,
        "upstream": upstream,
        "behind": behind,
        "ahead": ahead,
    }


def require_synced_training_git_state(root: Path) -> dict[str, int | str | bool]:
    """Refuse a training launch unless code is committed and pushed first."""
    state = git_state(root)
    if state["dirty"]:
        raise RuntimeError(
            "Refusing to train from a dirty worktree. Commit and push the exact "
            "code and protocol before launching a run."
        )
    if state["behind"] or state["ahead"]:
        raise RuntimeError(
            "Refusing to train unless HEAD matches its upstream. Run `git fetch "
            "--prune`, reconcile any remote changes, and push the launch commit. "
            f"upstream={state['upstream']} behind={state['behind']} ahead={state['ahead']}"
        )
    return state
