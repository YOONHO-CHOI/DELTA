"""Hash maintained source files without importing optional models."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from typing import Iterable

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_identity(module_names: Iterable[str]) -> dict[str, dict[str, str]]:
    """Hash retained evaluation source without importing optional model modules."""
    package = importlib.import_module("experiments")
    package_root = Path(package.__file__).resolve().parent
    identities = {}
    for name in sorted(set(module_names)):
        parts = str(name).split(".")
        if parts[0] != "experiments" or not all(part.isidentifier() for part in parts):
            raise ValueError(f"not a retained experiment module: {name!r}")
        path = package_root.joinpath(*parts[1:])
        path = path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
        path = path.resolve()
        if not path.is_relative_to(package_root) or not path.is_file():
            raise FileNotFoundError(f"evaluation source missing: {name}")
        identities[str(name)] = {"path": str(path), "sha256": sha256_file(path)}
    return identities
