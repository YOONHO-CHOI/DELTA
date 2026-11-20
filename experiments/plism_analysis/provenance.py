"""Public evaluation-source receipts, independent of the private operator code."""
from __future__ import annotations
import hashlib
import importlib
import json
import os
from pathlib import Path


def analysis_root() -> Path:
    return Path(os.environ.get("DELTA_PLISM_ANALYSIS_ROOT", Path(__file__).resolve().parent)).expanduser().resolve()


def canonical_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def verify_release_manifest(path: Path | None = None) -> dict:
    manifest_path = Path(path) if path is not None else analysis_root() / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("hash_policy") != "sha256-lf-canonical-bytes-v1":
        raise ValueError("unsupported evaluation hash policy")
    package = importlib.import_module("experiments")
    root = Path(package.__file__).resolve().parent
    for relative, expected in manifest["files"].items():
        source = Path(relative)
        if source.is_absolute() or source.parts[:1] != ("experiments",):
            raise ValueError(f"unsupported evaluation source scope: {relative}")
        target = root.joinpath(*source.parts[1:]).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError(f"invalid evaluation source: {relative}")
        if canonical_sha256(target) != expected:
            raise ValueError(f"evaluation code changed: {relative}")
    return manifest


def bind_release_output(value):
    if not isinstance(value, dict):
        return value
    verify_release_manifest()
    return {**value, "runtime_provenance": "current-release",
            "release_manifest_sha256": canonical_sha256(analysis_root() / "release_manifest.json")}


def validate_current_release_output(marker: dict) -> None:
    expected = canonical_sha256(analysis_root() / "release_manifest.json")
    if marker.get("runtime_provenance") != "current-release" or marker.get("release_manifest_sha256") != expected:
        raise ValueError("current output is not bound to the current evaluation release")
    verify_release_manifest()


def validate_source_protocol(marker: dict, current_protocol: Path, historical_hashes) -> bool:
    """Validate explicitly supplied historical identities without private imports."""
    digest = marker["protocol_sha256"]
    if digest == canonical_sha256(current_protocol):
        validate_current_release_output(marker)
        return True
    if digest not in historical_hashes:
        raise ValueError("unrecognized historical source protocol identity")
    return False
