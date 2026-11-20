#!/usr/bin/env python3
"""Build task-specific paired statistics from saved STP-Bench predictions.

The analysis unit is a tissue section within its frozen fold.  Internal
datasets and external source-to-target transfers are never pooled into one
inferential population.  Each dataset/transfer x gene-panel task receives
its own paired RAW-versus-DELTA confidence interval and randomization test.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import warnings
from typing import Any


import numpy as np
import pandas as pd
from scipy import sparse
import yaml

from experiments.stpbench_molecular_prediction.report import atomic_csv, atomic_text, load_complete_shard, sha256_file
from .paths import statistics_input_roots


def timestamp() -> str:
    from datetime import datetime

    return datetime.now().astimezone().isoformat(timespec="seconds")


def stable_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "little") ^ int(seed)) % (2**63 - 1)


def dense_float32(value: Any) -> np.ndarray:
    array = value.toarray() if sparse.issparse(value) else np.asarray(value)
    return np.asarray(array, dtype=np.float32)


def prediction_dir(
    result: dict[str, Any],
    fold: int,
    official_raw_result_root: Path | None = None,
    dataset: str | None = None,
    panel: str | None = None,
) -> Path:
    key = str(fold)
    missing: list[Path] = []
    for item in result.get("results", []):
        folds = item.get("artifacts", {}).get("folds", {})
        payload = folds.get(key, folds.get(fold))
        if isinstance(payload, dict) and payload.get("prediction_dir"):
            path = Path(payload["prediction_dir"])
            if path.is_dir():
                return path
            missing.append(path)
            if official_raw_result_root is not None and dataset and panel:
                marker = f"/01_runs/{dataset}/{panel}/"
                recorded = path.as_posix()
                if marker in recorded:
                    suffix = recorded.split(marker, 1)[1]
                    relocated = (
                        official_raw_result_root / "01_runs" / dataset / panel / suffix
                    )
                    if relocated.is_dir():
                        return relocated
                    missing.append(relocated)
    if missing:
        raise FileNotFoundError(
            "Prediction directory is absent; checked: "
            + ", ".join(str(path) for path in missing)
        )
    raise RuntimeError(f"Prediction directory is absent for fold {fold}")


def truth_root(result: dict[str, Any], fold: int, fallback: Path | None = None) -> Path:
    missing: list[Path] = []
    for payload in result.get("plan", {}).get("payloads", []):
        if int(payload.get("fold", -1)) != int(fold):
            continue
        value = payload.get("runtime_config", {}).get("DATA", {}).get("data_dir")
        if value:
            path = Path(value)
            if path.is_dir():
                return path
            missing.append(path)
    if fallback is not None and fallback.is_dir():
        return fallback
    if missing:
        raise FileNotFoundError(
            "Truth data root is absent; checked: "
            + ", ".join(str(path) for path in missing)
        )
    raise RuntimeError(f"Truth data root is absent for fold {fold}")


def truth_path(root: Path, name: str) -> Path:
    for subdirectory in ("st", "adata"):
        path = root / subdirectory / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"Truth H5AD is absent under {root}: {name}")


def section_metric(truth: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    import torch
    from torchmetrics.regression import PearsonCorrCoef

    if truth.shape != prediction.shape or truth.ndim != 2:
        raise ValueError(f"Invalid truth/prediction shapes: {truth.shape} {prediction.shape}")
    metric = PearsonCorrCoef(num_outputs=truth.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        per_gene = metric(torch.from_numpy(prediction), torch.from_numpy(truth))
    pcc = float(torch.nanmean(per_gene).item())
    mae = float(np.mean(np.abs(truth - prediction), dtype=np.float64))
    if not math.isfinite(pcc) or not math.isfinite(mae):
        raise RuntimeError(f"Non-finite section metric: PCC={pcc} MAE={mae}")
    return pcc, mae


def paired_section_rows(
    raw_result: dict[str, Any],
    delta_result: dict[str, Any],
    fold: int,
    fixed_fields: dict[str, Any],
    official_raw_result_root: Path,
    triplex_asset_root: Path,
    dataset: str,
    panel: str,
) -> list[dict[str, Any]]:
    import anndata as ad

    raw_dir = prediction_dir(
        raw_result,
        fold,
        official_raw_result_root=official_raw_result_root,
        dataset=dataset,
        panel=panel,
    )
    delta_dir = prediction_dir(delta_result, fold)
    raw_root = truth_root(raw_result, fold, fallback=triplex_asset_root)
    delta_root = truth_root(delta_result, fold, fallback=triplex_asset_root)
    raw_files = {path.name: path for path in raw_dir.glob("*.h5ad")}
    delta_files = {path.name: path for path in delta_dir.glob("*.h5ad")}
    if not raw_files or raw_files.keys() != delta_files.keys():
        raise RuntimeError(
            f"RAW/DELTA section inventory mismatch: raw={sorted(raw_files)} delta={sorted(delta_files)}"
        )

    rows: list[dict[str, Any]] = []
    for name in sorted(raw_files):
        raw = ad.read_h5ad(raw_files[name])
        delta = ad.read_h5ad(delta_files[name])
        if not raw.obs_names.equals(delta.obs_names) or not raw.var_names.equals(delta.var_names):
            raise RuntimeError(f"RAW/DELTA H5AD axes differ: {name}")
        raw_truth_path = truth_path(raw_root, name)
        delta_truth_path = truth_path(delta_root, name)
        source = ad.read_h5ad(raw_truth_path)
        mirror = source if raw_truth_path.resolve() == delta_truth_path.resolve() else ad.read_h5ad(delta_truth_path)
        if not source.obs_names.equals(mirror.obs_names) or not source.var_names.equals(mirror.var_names):
            raise RuntimeError(f"RAW/DELTA truth-mirror axes differ: {name}")
        if not raw.obs_names.isin(source.obs_names).all() or not raw.var_names.isin(source.var_names).all():
            raise RuntimeError(f"Prediction axes are not contained in truth H5AD: {name}")
        selected = source[raw.obs_names, raw.var_names]
        mirror_selected = mirror[raw.obs_names, raw.var_names]
        truth_counts = dense_float32(selected.X)
        mirror_counts = dense_float32(mirror_selected.X)
        if not np.array_equal(truth_counts, mirror_counts):
            raise RuntimeError(f"RAW/DELTA truth-mirror expression differs: {name}")
        y = np.log1p(truth_counts)
        raw_x = dense_float32(raw.X)
        delta_x = dense_float32(delta.X)
        raw_pcc, raw_mae = section_metric(y, raw_x)
        delta_pcc, delta_mae = section_metric(y, delta_x)
        rows.append(
            {
                **fixed_fields,
                "fold": int(fold),
                "section": Path(name).stem,
                "n_spots": int(y.shape[0]),
                "n_genes": int(y.shape[1]),
                "pcc_raw": raw_pcc,
                "pcc_delta": delta_pcc,
                "delta_minus_raw_pcc": delta_pcc - raw_pcc,
                "mae_raw": raw_mae,
                "mae_delta": delta_mae,
                "raw_minus_delta_mae": raw_mae - delta_mae,
            }
        )
    return rows


def bootstrap_effect(
    values_by_fold: list[np.ndarray], repetitions: int, rng: np.random.Generator
) -> np.ndarray:
    draws = np.zeros(repetitions, dtype=np.float64)
    for values in values_by_fold:
        indices = rng.integers(0, len(values), size=(repetitions, len(values)))
        draws += values[indices].mean(axis=1) / len(values_by_fold)
    return draws


def sign_flip_p(
    values_by_fold: list[np.ndarray], repetitions: int, rng: np.random.Generator
) -> tuple[float, str]:
    values: list[float] = []
    weights: list[float] = []
    n_folds = len(values_by_fold)
    for fold_values in values_by_fold:
        values.extend(float(value) for value in fold_values)
        weights.extend([1.0 / (n_folds * len(fold_values))] * len(fold_values))
    values_array = np.asarray(values, dtype=np.float64)
    weighted = values_array * np.asarray(weights, dtype=np.float64)
    observed = abs(float(weighted.sum()))
    n_sections = len(values_array)
    tolerance = 1e-15
    if n_sections <= 20:
        exceed = 0
        total = 2**n_sections
        for signs in itertools.product((-1.0, 1.0), repeat=n_sections):
            if abs(float(np.dot(np.asarray(signs), weighted))) >= observed - tolerance:
                exceed += 1
        return exceed / total, "exact_paired_section_sign_flip"

    exceed = 0
    completed = 0
    chunk = 10_000
    while completed < repetitions:
        size = min(chunk, repetitions - completed)
        signs = rng.integers(0, 2, size=(size, n_sections), dtype=np.int8)
        signs = signs.astype(np.float64) * 2.0 - 1.0
        statistics = np.abs(signs @ weighted)
        exceed += int(np.count_nonzero(statistics >= observed - tolerance))
        completed += size
    return (exceed + 1) / (repetitions + 1), f"monte_carlo_paired_section_sign_flip_{repetitions}"


def holm(values: pd.Series) -> pd.Series:
    array = values.to_numpy(dtype=float)
    order = np.argsort(array)
    adjusted = np.empty(len(array), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(array) - rank) * array[index])
        running = max(running, candidate)
        adjusted[index] = running
    return pd.Series(adjusted, index=values.index)


def aggregate_point(frame: pd.DataFrame, column: str) -> float:
    return float(frame.groupby("fold", sort=True)[column].mean().mean())


def task_statistics(
    frame: pd.DataFrame,
    label: str,
    bootstrap_repetitions: int,
    permutation_repetitions: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "n_folds": int(frame["fold"].nunique()),
        "n_sections": int(len(frame)),
        "n_spots": int(frame["n_spots"].sum()),
    }
    for metric, raw_column, delta_column, effect_column in (
        ("pcc", "pcc_raw", "pcc_delta", "delta_minus_raw_pcc"),
        ("mae", "mae_raw", "mae_delta", "raw_minus_delta_mae"),
    ):
        values_by_fold = [
            group[effect_column].to_numpy(dtype=np.float64)
            for _, group in frame.groupby("fold", sort=True)
        ]
        rng_boot = np.random.default_rng(stable_seed(seed, f"{label}:{metric}:bootstrap"))
        rng_perm = np.random.default_rng(stable_seed(seed, f"{label}:{metric}:permutation"))
        draws = bootstrap_effect(values_by_fold, bootstrap_repetitions, rng_boot)
        p_value, p_method = sign_flip_p(values_by_fold, permutation_repetitions, rng_perm)
        result[f"{metric}_raw"] = aggregate_point(frame, raw_column)
        result[f"{metric}_delta"] = aggregate_point(frame, delta_column)
        result[f"{metric}_effect"] = aggregate_point(frame, effect_column)
        result[f"{metric}_ci_low"] = float(np.quantile(draws, 0.025))
        result[f"{metric}_ci_high"] = float(np.quantile(draws, 0.975))
        result[f"{metric}_p"] = float(p_value)
        result[f"{metric}_p_method"] = p_method
        result[f"{metric}_delta_wins_sections"] = int((frame[effect_column] > 0).sum())
    return result


def check_official_metrics(
    section_frame: pd.DataFrame, official_internal: pd.DataFrame, official_external: pd.DataFrame
) -> dict[str, float]:
    errors = {"internal_pcc": 0.0, "internal_mae": 0.0, "external_pcc": 0.0, "external_mae": 0.0}
    internal = section_frame[section_frame["domain"] == "internal"]
    for (dataset, fold, panel), group in internal.groupby(["task", "fold", "panel"], sort=True):
        for arm in ("raw", "delta"):
            expected = official_internal[
                (official_internal["dataset"] == dataset)
                & (official_internal["fold"] == fold)
                & (official_internal["panel"] == panel)
                & (official_internal["arm"] == arm)
            ]
            if len(expected) != 1:
                raise RuntimeError(f"Missing official internal row: {dataset} {fold} {panel} {arm}")
            for metric in ("pcc", "mae"):
                observed = float(group[f"{metric}_{arm}"].mean())
                target = float(expected.iloc[0][metric])
                errors[f"internal_{metric}"] = max(errors[f"internal_{metric}"], abs(observed - target))

    external = section_frame[section_frame["domain"] == "external"]
    for (pair, fold, panel), group in external.groupby(["task", "fold", "panel"], sort=True):
        for arm in ("raw", "delta"):
            expected = official_external[
                (official_external["pair"] == pair)
                & (official_external["source_fold"] == fold)
                & (official_external["panel"] == panel)
                & (official_external["arm"] == arm)
            ]
            if len(expected) != 1:
                raise RuntimeError(f"Missing official external row: {pair} {fold} {panel} {arm}")
            for metric in ("pcc", "mae"):
                observed = float(group[f"{metric}_{arm}"].mean())
                target = float(expected.iloc[0][metric])
                errors[f"external_{metric}"] = max(errors[f"external_{metric}"], abs(observed - target))
    if max(errors.values()) > 5e-6:
        raise RuntimeError(f"Section reconstruction does not reproduce official metrics: {errors}")
    return errors


def build_statistics(
    config_path: Path,
    run_root: Path,
    output_root: Path,
    bootstrap_repetitions: int,
    permutation_repetitions: int,
    seed: int,
    only: str | None,
) -> None:
    import torch

    torch.set_num_threads(1)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    official_raw_result_root, prepared_asset_root = statistics_input_roots(config, run_root)
    evaluation = config.get("evaluation", {})
    if evaluation.get("paired_unit") != "section_within_frozen_fold":
        raise RuntimeError(f"Unexpected paired unit: {evaluation.get('paired_unit')}")
    official_internal = pd.read_csv(run_root / "results" / "internal_fold_metrics.csv")
    official_external = pd.read_csv(run_root / "results" / "external_pair_metrics.csv")

    section_rows: list[dict[str, Any]] = []
    internal_tasks: list[tuple[str, str]] = [
        (dataset, panel) for dataset in config["internal_datasets"] for panel in config["panels"]
    ]
    external_tasks: list[tuple[str, str]] = [
        (pair, panel) for pair in config["external_pairs"] for panel in config["panels"]
    ]

    for dataset, panel in internal_tasks:
        label = f"internal:{dataset}:{panel}"
        if only is not None and only != label:
            continue
        print(f"STPBENCH_STATS_TASK_START timestamp={timestamp()} task={label}", flush=True)
        frozen_folds = config.get("evaluation", {}).get("internal_folds")
        folds = (
            [int(value) for value in frozen_folds]
            if frozen_folds is not None
            else range(int(config["internal_datasets"][dataset]["n_folds"]))
        )
        for fold in folds:
            _, raw_workflow, _ = load_complete_shard(run_root, dataset, fold, panel, "raw")
            _, delta_workflow, _ = load_complete_shard(run_root, dataset, fold, panel, "delta")
            section_rows.extend(
                paired_section_rows(
                    raw_workflow["internal"],
                    delta_workflow["internal"],
                    fold,
                    {"domain": "internal", "task": dataset, "panel": panel},
                    official_raw_result_root,
                    prepared_asset_root,
                    dataset,
                    panel,
                )
            )
        print(f"STPBENCH_STATS_TASK_INPUT_COMPLETE timestamp={timestamp()} task={label}", flush=True)

    for pair, panel in external_tasks:
        label = f"external:{pair}:{panel}"
        if only is not None and only != label:
            continue
        print(f"STPBENCH_STATS_TASK_START timestamp={timestamp()} task={label}", flush=True)
        spec = config["external_pairs"][pair]
        dataset = str(spec["source"])
        fold = int(spec["source_fold"])
        _, raw_workflow, _ = load_complete_shard(run_root, dataset, fold, panel, "raw")
        _, delta_workflow, _ = load_complete_shard(run_root, dataset, fold, panel, "delta")
        section_rows.extend(
            paired_section_rows(
                raw_workflow["external"][pair],
                delta_workflow["external"][pair],
                fold,
                {
                    "domain": "external",
                    "task": pair,
                    "panel": panel,
                    "source_dataset": dataset,
                    "target_dataset": str(spec["target"]),
                },
                official_raw_result_root,
                prepared_asset_root,
                dataset,
                panel,
            )
        )
        print(f"STPBENCH_STATS_TASK_INPUT_COMPLETE timestamp={timestamp()} task={label}", flush=True)

    if not section_rows:
        raise RuntimeError(f"No task matched --only={only!r}")
    sections = pd.DataFrame(section_rows).sort_values(
        ["domain", "task", "panel", "fold", "section"]
    )
    reconstruction_errors = check_official_metrics(sections, official_internal, official_external)

    internal_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    for (domain, task, panel), frame in sections.groupby(["domain", "task", "panel"], sort=True):
        label = f"{domain}:{task}:{panel}"
        summary = task_statistics(
            frame,
            label,
            bootstrap_repetitions,
            permutation_repetitions,
            seed,
        )
        row = {"task": task, "panel": panel, **summary}
        if domain == "internal":
            row["dataset"] = task
            internal_rows.append(row)
        else:
            row["pair"] = task
            row["source_dataset"] = str(frame["source_dataset"].iloc[0])
            row["target_dataset"] = str(frame["target_dataset"].iloc[0])
            external_rows.append(row)

    internal_frame = pd.DataFrame(internal_rows)
    external_frame = pd.DataFrame(external_rows)
    for frame in (internal_frame, external_frame):
        if frame.empty:
            continue
        for panel, indices in frame.groupby("panel").groups.items():
            for metric in ("pcc", "mae"):
                frame.loc[indices, f"{metric}_holm_p"] = holm(frame.loc[indices, f"{metric}_p"])
                frame.loc[indices, f"{metric}_holm_family"] = (
                    f"{panel}:{metric}:tasks_within_domain"
                )

    if only is None and (len(internal_frame) != 14 or len(external_frame) != 12):
        raise RuntimeError(
            f"Unexpected task counts: internal={len(internal_frame)} external={len(external_frame)}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, pd.DataFrame] = {"task_statistics_section_metrics.csv": sections}
    if not internal_frame.empty:
        outputs["internal_task_statistics.csv"] = internal_frame.sort_values(["dataset", "panel"])
    if not external_frame.empty:
        outputs["external_task_statistics.csv"] = external_frame.sort_values(["pair", "panel"])
    for name, frame in outputs.items():
        atomic_csv(output_root / name, frame)

    audit = {
        "schema": "delta-stpbench-task-specific-statistics-v1",
        "status": "complete",
        "analysis_unit": "tissue_section_within_frozen_fold",
        "tasks_are_not_pooled": True,
        "pcc_effect": "DELTA minus RAW; positive favors DELTA",
        "mae_effect": "RAW minus DELTA; positive favors DELTA",
        "bootstrap": {
            "method": "paired_section_bootstrap_stratified_by_frozen_fold",
            "repetitions": int(bootstrap_repetitions),
            "seed": int(seed),
        },
        "randomization": {
            "method": "paired_section_sign_flip_stratified_by_frozen_fold",
            "monte_carlo_repetitions": int(permutation_repetitions),
            "exact_when_n_sections_at_most": 20,
        },
        "multiplicity": "Holm separately within domain x panel x metric across tasks",
        "constant_or_near_constant_gene_pcc": "NaN excluded by torch.nanmean, matching stored STP-Bench metrics",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "run_root": str(run_root),
        "source_report_audit_sha256": sha256_file(run_root / "results" / "audit.json"),
        "only": only,
        "internal_tasks": int(len(internal_frame)),
        "external_tasks": int(len(external_frame)),
        "paired_sections": int(len(sections)),
        "all_finite": bool(
            np.isfinite(
                sections[
                    [
                        "pcc_raw",
                        "pcc_delta",
                        "delta_minus_raw_pcc",
                        "mae_raw",
                        "mae_delta",
                        "raw_minus_delta_mae",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "official_metric_reconstruction_max_abs_error": reconstruction_errors,
        "outputs": {
            name: {"path": str(output_root / name), "sha256": sha256_file(output_root / name)}
            for name in outputs
        },
    }
    atomic_text(output_root / "task_statistics_audit.json", json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(
        f"STPBENCH_TASK_STATISTICS_COMPLETE output={output_root} "
        f"internal={len(internal_frame)} external={len(external_frame)} sections={len(sections)}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--permutation-repetitions", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--only",
        help="Optional canary task: internal:<dataset>:<panel> or external:<pair>:<panel>",
    )
    args = parser.parse_args()
    output_root = args.output_root or args.run_root / "results"
    if args.bootstrap_repetitions < 100 or args.permutation_repetitions < 100:
        raise ValueError("At least 100 bootstrap and permutation repetitions are required")
    build_statistics(
        args.config,
        args.run_root,
        output_root,
        args.bootstrap_repetitions,
        args.permutation_repetitions,
        args.seed,
        args.only,
    )


if __name__ == "__main__":
    main()
