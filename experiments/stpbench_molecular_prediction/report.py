#!/usr/bin/env python3
"""Build deterministic STP-Bench molecular-prediction result tables.

This is a CPU-only finalizer. It reads only atomically completed shard
manifests, fails closed until the experiment's frozen fold x panel x arm grid
is present, and never edits training artifacts or derived embeddings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd
import yaml

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_text(path, frame.to_csv(index=False, lineterminator="\n"))


def metric_payload(result: dict[str, Any], fold: int, predictor: str) -> dict[str, float]:
    metrics = result["summary"][predictor]["metrics"]
    payload = metrics.get(str(fold), metrics.get(fold))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Missing {predictor} metrics for fold {fold}")
    pcc = float(payload["test_PearsonCorrCoef"])
    mae = float(payload["test_MeanAbsoluteError"])
    if not math.isfinite(pcc) or not math.isfinite(mae):
        raise RuntimeError(f"Non-finite metric for fold {fold}: PCC={pcc}, MAE={mae}")
    return {"pcc": pcc, "mae": mae}


def expected_shards(config: dict[str, Any]) -> list[tuple[str, int, str, str]]:
    rows: list[tuple[str, int, str, str]] = []
    frozen_folds = config.get("evaluation", {}).get("internal_folds")
    for dataset, spec in config["internal_datasets"].items():
        folds = [int(value) for value in frozen_folds] if frozen_folds is not None else range(int(spec["n_folds"]))
        for fold in folds:
            for panel in config["panels"]:
                for arm in config["arms"]:
                    rows.append((dataset, fold, panel, arm))
    return rows


def load_complete_shard(
    run_root: Path,
    dataset: str,
    fold: int,
    panel: str,
    arm: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    shard = run_root / "shards" / dataset / f"fold_{fold}" / panel / arm
    complete_path = shard / "_COMPLETE.json"
    workflow_path = shard / "workflow_result.json"
    if not complete_path.is_file() or not workflow_path.is_file():
        raise FileNotFoundError(f"Incomplete expected shard: {shard}")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    expected = {
        "status": "complete",
        "dataset": dataset,
        "fold": fold,
        "panel": panel,
        "arm": arm,
        "canary": False,
    }
    for key, value in expected.items():
        if complete.get(key) != value or workflow.get(key) != value:
            raise RuntimeError(
                f"Shard contract mismatch at {shard}: {key} "
                f"complete={complete.get(key)!r}, workflow={workflow.get(key)!r}, expected={value!r}"
            )
    if complete.get("checkpoint_sha256") != workflow.get("checkpoint_sha256"):
        raise RuntimeError(f"Checkpoint hash mismatch: {shard}")
    return complete, workflow, workflow_path


def build_report(config_path: Path, run_root: Path, output_root: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    predictor = str(config["upstream"]["predictor"])
    internal_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []

    external_by_source = {
        (str(spec["source"]), int(spec["source_fold"])): []
        for spec in config["external_pairs"].values()
    }
    for pair, spec in config["external_pairs"].items():
        external_by_source[(str(spec["source"]), int(spec["source_fold"]))].append(pair)

    for dataset, fold, panel, arm in expected_shards(config):
        complete, workflow, workflow_path = load_complete_shard(
            run_root, dataset, fold, panel, arm
        )
        internal_metric = metric_payload(workflow["internal"], fold, predictor)
        internal_rows.append(
            {
                "dataset": dataset,
                "fold": fold,
                "panel": panel,
                "arm": arm,
                **internal_metric,
                "checkpoint_sha256": complete["checkpoint_sha256"],
                "completed_at": complete["completed_at"],
            }
        )
        artifact_rows.append(
            {
                "dataset": dataset,
                "fold": fold,
                "panel": panel,
                "arm": arm,
                "workflow_result": str(workflow_path),
                "workflow_sha256": sha256_file(workflow_path),
                "checkpoint_sha256": complete["checkpoint_sha256"],
                "source_workflow_sha256": complete.get("source_workflow_sha256"),
            }
        )

        expected_external = sorted(external_by_source.get((dataset, fold), []))
        observed_external = sorted(workflow.get("external", {}))
        if observed_external != expected_external:
            raise RuntimeError(
                f"External result mismatch for {dataset} fold {fold} {panel} {arm}: "
                f"observed={observed_external}, expected={expected_external}"
            )
        for pair in expected_external:
            external_metric = metric_payload(workflow["external"][pair], fold, predictor)
            external_rows.append(
                {
                    "pair": pair,
                    "source_dataset": dataset,
                    "source_fold": fold,
                    "target_dataset": config["external_pairs"][pair]["target"],
                    "panel": panel,
                    "arm": arm,
                    **external_metric,
                    "checkpoint_sha256": complete["checkpoint_sha256"],
                }
            )

    internal = pd.DataFrame(internal_rows).sort_values(
        ["dataset", "fold", "panel", "arm"]
    )
    external = pd.DataFrame(external_rows).sort_values(["pair", "panel", "arm"])
    artifacts = pd.DataFrame(artifact_rows).sort_values(
        ["dataset", "fold", "panel", "arm"]
    )
    expected_internal = len(expected_shards(config))
    expected_external = len(config["external_pairs"]) * len(config["panels"]) * len(config["arms"])
    if len(internal) != expected_internal or len(external) != expected_external:
        raise RuntimeError(
            f"Unexpected final row counts: internal={len(internal)} external={len(external)}"
        )

    internal_summary = (
        internal.groupby(["dataset", "panel", "arm"], as_index=False)
        .agg(n_folds=("fold", "nunique"), pcc_mean=("pcc", "mean"), mae_mean=("mae", "mean"))
        .sort_values(["dataset", "panel", "arm"])
    )
    internal_wide = internal.pivot(
        index=["dataset", "fold", "panel"], columns="arm", values=["pcc", "mae"]
    )
    if internal_wide.isna().any().any():
        raise RuntimeError("Internal RAW/DELTA pairing is incomplete")
    internal_paired = internal_wide.index.to_frame(index=False)
    for metric in ("pcc", "mae"):
        for arm in ("delta", "raw"):
            internal_paired[f"{metric}_{arm}"] = internal_wide[(metric, arm)].to_numpy()
    internal_paired["delta_minus_raw_pcc"] = (
        internal_paired["pcc_delta"] - internal_paired["pcc_raw"]
    )
    internal_paired["raw_minus_delta_mae"] = (
        internal_paired["mae_raw"] - internal_paired["mae_delta"]
    )

    external_wide = external.pivot(
        index=["pair", "source_dataset", "source_fold", "target_dataset", "panel"],
        columns="arm",
        values=["pcc", "mae"],
    )
    if external_wide.isna().any().any():
        raise RuntimeError("External RAW/DELTA pairing is incomplete")
    external_paired = external_wide.index.to_frame(index=False)
    for metric in ("pcc", "mae"):
        for arm in ("delta", "raw"):
            external_paired[f"{metric}_{arm}"] = external_wide[(metric, arm)].to_numpy()
    external_paired["delta_minus_raw_pcc"] = (
        external_paired["pcc_delta"] - external_paired["pcc_raw"]
    )
    external_paired["raw_minus_delta_mae"] = (
        external_paired["mae_raw"] - external_paired["mae_delta"]
    )

    output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "internal_fold_metrics.csv": internal,
        "internal_dataset_summary.csv": internal_summary,
        "internal_paired_differences.csv": internal_paired,
        "external_pair_metrics.csv": external,
        "external_paired_differences.csv": external_paired,
        "artifact_ledger.csv": artifacts,
    }
    for name, frame in outputs.items():
        atomic_csv(output_root / name, frame)

    audit = {
        "schema": "delta-stpbench-molecular-report-v1",
        "status": "complete",
        "predictor": predictor,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "run_root": str(run_root),
        "internal_rows": len(internal),
        "external_rows": len(external),
        "expected_shards": len(expected_shards(config)),
        "all_metrics_finite": bool(
            all(math.isfinite(float(value)) for value in internal[["pcc", "mae"]].to_numpy().ravel())
            and all(math.isfinite(float(value)) for value in external[["pcc", "mae"]].to_numpy().ravel())
        ),
        "outputs": {
            name: {"path": str(output_root / name), "sha256": sha256_file(output_root / name)}
            for name in outputs
        },
    }
    atomic_text(output_root / "audit.json", json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(
        f"STPBENCH_MOLECULAR_REPORT_COMPLETE output={output_root} "
        f"internal={len(internal)} external={len(external)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    output_root = args.output_root or args.run_root / "results"
    build_report(args.config, args.run_root, output_root)


if __name__ == "__main__":
    main()
