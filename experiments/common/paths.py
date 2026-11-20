"""Machine-independent runtime path resolution."""

from __future__ import annotations

import os
from pathlib import Path


def env_path(name: str, default: str | Path | None = None) -> Path:
    """Resolve a path from an environment variable with an optional default."""

    raw = os.environ.get(name)
    if raw:
        return Path(raw).expanduser().resolve()
    if default is None:
        raise RuntimeError(f"{name} is required")
    return Path(default).expanduser().resolve()


def project_root() -> Path:
    """Canonical workspace containing data, experiments, and results."""

    return env_path("DELTA_PROJECT_ROOT", Path.cwd())


def source_root() -> Path:
    """Raw/source data root."""

    return env_path("DELTA_SOURCE_ROOT", project_root() / "source_data")


def huggingface_home() -> Path:
    """Hugging Face cache root."""

    return env_path("HF_HOME", Path.home() / ".cache" / "huggingface")


def pathorob_root() -> Path:
    """External PathoROB checkout."""

    return env_path("PATHOROB_ROOT")
