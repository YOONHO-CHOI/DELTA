"""Locate a user checkout for external configurations, not executable code."""
from __future__ import annotations

import os
from pathlib import Path


def repository_root() -> Path:
    explicit = os.environ.get("DELTA_REPOSITORY_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[2]
    return checkout if (checkout / "configs").is_dir() else Path.cwd().resolve()
