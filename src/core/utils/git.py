"""Lightweight git helpers for reproducibility artefacts.

Used by :class:`BaseExperiment` to record a commit SHA + dirty patch
alongside every parent run.
"""

from __future__ import annotations

import subprocess


def _run(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def commit_sha() -> str:
    try:
        return _run("rev-parse", "HEAD")
    except Exception:
        return "unknown"


def branch() -> str:
    try:
        return _run("rev-parse", "--abbrev-ref", "HEAD")
    except Exception:
        return "unknown"


def is_dirty() -> bool:
    try:
        out = _run("status", "--porcelain")
        return bool(out)
    except Exception:
        return False


def dirty_patch() -> str:
    """Diff vs HEAD, empty string if clean / git missing."""
    try:
        return subprocess.check_output(["git", "diff", "HEAD"], text=True)
    except Exception:
        return ""


def repo_state() -> dict[str, str | bool]:
    return {
        "commit": commit_sha(),
        "branch": branch(),
        "dirty": is_dirty(),
    }
