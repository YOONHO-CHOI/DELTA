"""Resolve external prediction and truth paths without a training pipeline."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def statistics_input_roots(config: dict[str, Any], run_root: Path) -> tuple[Path | None, Path]:
    """Retain existing explicit and environment/default prepared-asset contracts."""
    paths = config["paths"]
    if "defaults" in paths:
        environment = paths.get("prepared_asset_root_env")
        recorded = (os.environ.get(environment) if environment else None) or paths["defaults"].get("prepared_asset_root")
    else:
        recorded = paths.get("prepared_asset_root") or paths.get("triplex_asset_root")
    if not recorded:
        raise ValueError("Prepared STP-Bench asset root is absent from the statistics configuration")
    prepared = Path(recorded).expanduser().resolve()
    recorded_raw = paths.get("official_raw_result_root")
    raw = Path(recorded_raw).expanduser().resolve() if recorded_raw else None
    return raw, prepared
