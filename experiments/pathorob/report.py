"""Numerical patch-level and bank-coverage reports from frozen results."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from experiments.common.resources import repository_root
from .workbook import write_workbook

PROJECT_ROOT = Path(os.environ.get('DELTA_PROJECT_ROOT', repository_root())).expanduser().resolve()
PRIMARY_EXPERIMENT_RELATIVE = Path(
    "runs/pathorob_bank_coverage/p01_p50"
)
BASELINE_RELATIVE = Path(
    "runs/pathorob_representation_and_roi/historical_comparator_backing/results.csv"
)
TCGA2_DELTA_RELATIVE = Path(
    "runs/pathorob_representation_and_roi/representation/representation_results.csv"
)
TCGA2_BASELINE_RELATIVE = Path(
    "runs/pathorob_representation_and_roi/tcga2x2_completion/"
    "representation/baselines/baseline_results.csv"
)
P2_OUTPUT_RELATIVE = Path(
    "results/pathorob_representation-apd__20backbone__delta-v1-3-p2"
)
COVERAGE_OUTPUT_RELATIVE = Path(
    "results/pathorob_bank-coverage__20backbone__delta-v1-3__p01-p50"
)
RUN_P2 = True
RUN_COVERAGE = True
PRIMARY_BANK_COVERAGE = 2
BANK_COVERAGES = [1, 2, 5, 10, 20, 30, 40, 50]
PREDICTIVE_DATASETS = ["camelyon", "tcga_4x4", "tolkach_esca"]
REPRESENTATION_INVENTORY_DATASETS = [
    "camelyon", "tcga_4x4", "tolkach_esca", "tcga_2x2",
]
REPRESENTATION_DATASETS_BY_ENDPOINT = {
    "biological_knn": ["camelyon", "tcga_2x2", "tolkach_esca"],
    "confounder_insensitivity": ["camelyon", "tcga_2x2", "tolkach_esca"],
    "robustness_index": ["camelyon", "tcga_2x2", "tolkach_esca"],
    "clustering_score": ["camelyon", "tcga_4x4", "tolkach_esca"],
}
COVERAGE_DATASETS = ["camelyon", "tcga_4x4", "tolkach_esca"]
BASE_SPLITS = {"camelyon": 8, "tcga_4x4": 7, "tolkach_esca": 4}
BOOTSTRAP_REPETITIONS = 10_000
RANDOM_SEED = 20260819
DISPLAY_NAMES = {
    "RAW": "RAW", "DANN": "DANN", "DELTA-v1.3 p2": "DELTA",
    "R-RAW": "Reinhard-RAW", "R-DANN": "Reinhard-DANN",
    "R-DELTA-v1.3 p2": "Reinhard-DELTA",
    "ComBat-T": "ComBat (transductive)",
    "R-ComBat-T": "Reinhard-ComBat (transductive)",
}
METHOD_ORDER = [
    "RAW", "DANN", "DELTA-v1.3 p2",
    "R-RAW", "R-DANN", "R-DELTA-v1.3 p2",
    "ComBat-T", "R-ComBat-T",
]
CONDITION = {
    method: ("Transductive" if "ComBat-T" in method else "Inductive")
    for method in METHOD_ORDER
}
METHOD_MAP = {
    "RAW": "RAW",
    "DANN": "DANN",
    "ComBat": "ComBat-T",
    "Reinhard-LUSC500": "R-RAW",
    "Reinhard-LUSC500-DANN": "R-DANN",
    "Reinhard-LUSC500-ComBat": "R-ComBat-T",
    "Reinhard-RAW": "R-RAW",
    "Reinhard-ComBat": "R-ComBat-T",
    "DELTA-v1.3-p2": "DELTA-v1.3 p2",
    "Reinhard-DELTA-v1.3-p2": "R-DELTA-v1.3 p2",
}
BASELINE_METHODS = [
    "RAW", "DANN", "ComBat", "Reinhard-LUSC500",
    "Reinhard-LUSC500-DANN", "Reinhard-LUSC500-ComBat",
]
EXPECTED_BACKBONES = [
    "BEPH", "CONCH", "CONCH v1.5", "CTransPath", "Ciga",
    "EXAONE-Path-2.5", "GPFM", "H-optimus-0", "H0-mini", "HIPT",
    "Kaiko ViT-B/8", "MOOZY patch encoder (Lunit DINO)", "MUSK",
    "Original UNI", "Phikon-v2", "Prov-GigaPath", "RetCCL", "UNI2-h",
    "Virchow", "Virchow2",
]
REP_ENDPOINTS = [
    "biological_knn", "clustering_score",
    "confounder_insensitivity", "robustness_index",
]

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def holm_adjust(pvalues: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(pvalues), dtype=float)
    order = np.argsort(values)
    ranked = np.maximum.accumulate((len(values) - np.arange(len(values))) * values[order])
    ranked = np.minimum(ranked, 1.0)
    result = np.empty_like(ranked)
    result[order] = ranked
    return result

def finite_complete(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    complete = frame[frame["status"].eq("complete")].copy()
    if complete.empty:
        raise ValueError(f"no complete rows: {source}")
    if not np.isfinite(complete["value"].to_numpy(float)).all():
        raise ValueError(f"non-finite complete rows: {source}")
    return complete

def load_candidate(root: Path, evaluation: str) -> tuple[pd.DataFrame, list[Path]]:
    paths = sorted((root / evaluation / "results").glob("*.csv"))
    if len(paths) != 20:
        raise ValueError(f"{evaluation}: expected 20 CSVs, found {len(paths)}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    frame = finite_complete(frame, root / evaluation / "results")
    # The canonical per-backbone files now contain the complete p1--p50
    # coverage sweep.  This manuscript report is intentionally p2-only, so
    # select the frozen primary coverage explicitly instead of assuming that
    # each input file contains only p2.
    frame = frame.loc[
        frame["bank_fraction_percent"].astype(int).eq(PRIMARY_BANK_COVERAGE)
    ].copy()
    if frame.empty:
        raise ValueError(f"{evaluation}: no p{PRIMARY_BANK_COVERAGE} rows")
    if set(frame["backbone_label"]) != set(EXPECTED_BACKBONES):
        raise ValueError(f"{evaluation}: backbone inventory mismatch")
    if set(frame["bank_fraction_percent"].astype(int)) != {PRIMARY_BANK_COVERAGE}:
        raise ValueError(f"{evaluation}: p2-only report received another coverage")
    frame["backbone_std"] = frame["backbone_label"].astype(str)
    frame["method_std"] = frame["method"].map(METHOD_MAP)
    if frame["method_std"].isna().any():
        raise ValueError(f"{evaluation}: unmapped candidate method")
    return frame, paths

def load_coverage_candidate(root: Path, evaluation: str) -> tuple[pd.DataFrame, Path]:
    path = root / f"{evaluation}_results.csv"
    frame = finite_complete(pd.read_csv(path), path)
    observed_coverages = set(frame["bank_fraction_percent"].astype(int))
    if observed_coverages != set(BANK_COVERAGES):
        raise ValueError(
            f"{evaluation}: coverage mismatch {sorted(observed_coverages)}"
        )
    if set(frame["backbone_label"]) != set(EXPECTED_BACKBONES):
        raise ValueError(f"{evaluation}: backbone inventory mismatch")
    method = frame["method"].astype(str)
    frame["method_std"] = np.where(
        method.str.startswith("Reinhard-"), "Reinhard-DELTA", "DELTA"
    )
    frame["backbone_std"] = frame["backbone_label"].astype(str)
    frame["coverage"] = frame["bank_fraction_percent"].astype(int)
    return frame, path

def load_baseline(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = finite_complete(frame, path)
    frame = frame[
        frame["backbone"].isin(EXPECTED_BACKBONES)
        & frame["method"].isin(BASELINE_METHODS)
    ].copy()
    frame["backbone_std"] = frame["backbone"].astype(str)
    frame["method_std"] = frame["method"].map(METHOD_MAP)
    return frame

def load_canonical_representation(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    keys = ["backbone", "dataset", "method", "endpoint"]
    if len(frame) != 640 or frame.duplicated(keys).any():
        raise ValueError("canonical representation inventory must contain 640 unique rows")
    if set(frame["dataset"]) != set(REPRESENTATION_INVENTORY_DATASETS):
        raise ValueError("canonical representation dataset inventory mismatch")
    tcga2 = frame.loc[frame["dataset"].eq("tcga_2x2")]
    clustering = tcga2.loc[tcga2["endpoint"].eq("clustering_score")]
    if len(clustering) != 40 or set(clustering["status"]) != {"not_applicable"}:
        raise ValueError("TCGA2x2 clustering must be explicitly not applicable")
    if clustering["value"].notna().any():
        raise ValueError("TCGA2x2 clustering contains a fabricated value")
    frame = finite_complete(frame, path)
    if len(frame) != 600:
        raise ValueError("canonical representation finite endpoint inventory mismatch")
    if set(frame["backbone_label"]) != set(EXPECTED_BACKBONES):
        raise ValueError("TCGA2x2 DELTA backbone inventory mismatch")
    frame["backbone_std"] = frame["backbone_label"].astype(str)
    frame["method_std"] = frame["method"].map(METHOD_MAP)
    if frame["method_std"].isna().any():
        raise ValueError("TCGA2x2 DELTA contains an unmapped method")
    return frame

def load_tcga2_baseline(path: Path) -> pd.DataFrame:
    frame = finite_complete(pd.read_csv(path), path)
    keys = ["backbone", "dataset", "method", "endpoint"]
    expected_methods = {"RAW", "ComBat", "Reinhard-RAW", "Reinhard-ComBat"}
    expected_endpoints = {
        "biological_knn", "confounder_insensitivity", "robustness_index",
    }
    if len(frame) != 240 or frame.duplicated(keys).any():
        raise ValueError("TCGA2x2 baseline inventory must contain 240 unique rows")
    if set(frame["backbone"]) != set(EXPECTED_BACKBONES):
        raise ValueError("TCGA2x2 baseline backbone inventory mismatch")
    if set(frame["method"]) != expected_methods:
        raise ValueError("TCGA2x2 baseline method inventory mismatch")
    if set(frame["endpoint"]) != expected_endpoints:
        raise ValueError("TCGA2x2 baseline endpoint inventory mismatch")
    frame["backbone_std"] = frame["backbone"].astype(str)
    frame["method_std"] = frame["method"].map(METHOD_MAP)
    frame["domain"] = "FULL"
    frame["track"] = "A"
    if frame["method_std"].isna().any():
        raise ValueError("TCGA2x2 baseline contains an unmapped method")
    return frame

def combine_apd(baseline: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    base = baseline[
        baseline["track"].eq("B")
        & baseline["endpoint"].eq("balanced_accuracy")
        & baseline["domain"].isin(["ID", "OOD"])
    ][[
        "backbone_std", "dataset", "split_index", "repetition_seed",
        "method_std", "domain", "value",
    ]]
    new = candidate[[
        "backbone_std", "dataset", "split_index", "repetition_seed",
        "method_std", "domain", "value",
    ]]
    result = pd.concat([base, new], ignore_index=True)
    result = result[result["dataset"].isin(PREDICTIVE_DATASETS)].copy()
    return result

def combine_representation(baseline: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    base = baseline[
        baseline["track"].eq("A")
        & baseline["domain"].eq("FULL")
        & baseline["endpoint"].isin(REP_ENDPOINTS)
    ][["backbone_std", "dataset", "method_std", "endpoint", "value"]]
    new = candidate[["backbone_std", "dataset", "method_std", "endpoint", "value"]]
    result = pd.concat([base, new], ignore_index=True)
    return result[result["dataset"].isin(REPRESENTATION_INVENTORY_DATASETS)].copy()

def backbone_balanced_accuracy(apd_rows: pd.DataFrame) -> pd.DataFrame:
    dataset = apd_rows.groupby(
        ["backbone_std", "dataset", "method_std", "domain"], as_index=False
    )["value"].mean()
    result = dataset.groupby(
        ["backbone_std", "method_std", "domain"], as_index=False
    )["value"].mean()
    result["metric"] = "balanced_accuracy"
    return result

def raw_balanced_apd_references(apd_rows: pd.DataFrame) -> pd.Series:
    """Shared RAW split-0 mean over repetitions, separately per encoder/task/domain."""
    raw = apd_rows.loc[
        apd_rows['method_std'].eq('RAW')
        & apd_rows['dataset'].isin(BASE_SPLITS)
        & apd_rows['domain'].isin(['ID','OOD'])
    ].copy()
    if 'endpoint' in raw:
        raw = raw.loc[raw['endpoint'].eq('balanced_accuracy')]
    if 'track' in raw:
        raw = raw.loc[raw['track'].eq('B')]
    if raw.empty:
        raise ValueError('APD requires the common RAW balanced-split reference scores')
    keys = ['backbone_std','dataset','domain','repetition_seed','split_index']
    if raw.duplicated(keys).any() or not np.isfinite(raw['value']).all():
        raise ValueError('RAW APD references contain duplicate keys or non-finite values')
    raw = raw.loc[raw['split_index'].astype(int).mod(raw['dataset'].map(BASE_SPLITS)).eq(0)]
    references = raw.groupby(['backbone_std','dataset','domain'])['value'].mean()
    if references.empty or not np.isfinite(references).all() or references.le(0).any():
        raise ValueError('Missing or non-positive RAW balanced-split APD reference')
    return references

def backbone_average_performance_drop(apd_rows: pd.DataFrame) -> pd.DataFrame:
    """PathoROB robustification Eq. 8; never normalize by each method's own score."""
    references = raw_balanced_apd_references(apd_rows)
    rows = []
    keys = ["backbone_std", "dataset", "method_std", "domain", "repetition_seed"]
    for key, group in apd_rows.groupby(keys, sort=False):
        backbone, dataset, method, domain, seed = key
        base_splits = BASE_SPLITS[str(dataset)]
        step = group.assign(correlation_step=group["split_index"].astype(int) % base_splits)
        values = step.groupby("correlation_step")["value"].mean()
        if set(values.index) != set(range(base_splits)) or len(step) != base_splits:
            raise ValueError(f'incomplete or duplicated APD curve: {key}')
        reference = float(references.loc[(backbone,dataset,domain)])
        rows.append({
            "backbone_std": backbone,
            "dataset": dataset,
            "method_std": method,
            "domain": domain,
            "repetition_seed": seed,
            "value": float((values.loc[1:] / reference).mean() - 1.0),
        })
    repetition = pd.DataFrame(rows)
    if repetition.empty:
        raise ValueError("average performance drop could not be reconstructed")
    dataset = repetition.groupby(
        ["backbone_std", "dataset", "method_std", "domain"], as_index=False
    )["value"].mean()
    result = dataset.groupby(
        ["backbone_std", "method_std", "domain"], as_index=False
    )["value"].mean()
    result["metric"] = "average_performance_drop"
    return result

def backbone_representation(rep_rows: pd.DataFrame) -> pd.DataFrame:
    expected = pd.DataFrame([
        {"endpoint": endpoint, "dataset": dataset}
        for endpoint, datasets in REPRESENTATION_DATASETS_BY_ENDPOINT.items()
        for dataset in datasets
    ])
    selected = rep_rows.merge(
        expected, on=["endpoint", "dataset"], how="inner", validate="many_to_one",
    )
    dataset = selected.groupby(
        ["backbone_std", "dataset", "method_std", "endpoint"], as_index=False
    )["value"].mean()
    result = dataset.groupby(
        ["backbone_std", "method_std", "endpoint"], as_index=False
    )["value"].mean()
    counts = dataset.groupby(
        ["backbone_std", "method_std", "endpoint"], as_index=False
    )["dataset"].nunique().rename(columns={"dataset": "datasets_averaged"})
    return result.merge(
        counts, on=["backbone_std", "method_std", "endpoint"],
        validate="one_to_one",
    )

def coverage_balanced_accuracy(rows: pd.DataFrame) -> pd.DataFrame:
    dataset = rows.groupby(
        ["coverage", "backbone_std", "dataset", "method_std", "domain"],
        as_index=False,
    )["value"].mean()
    result = dataset.groupby(
        ["coverage", "backbone_std", "method_std", "domain"], as_index=False
    )["value"].mean()
    result["metric"] = "balanced_accuracy"
    return result

def coverage_average_performance_drop(rows: pd.DataFrame, raw_rows: pd.DataFrame) -> pd.DataFrame:
    """Use the same frozen RAW references for every coverage and DELTA branch."""
    references = raw_balanced_apd_references(raw_rows)
    output = []
    keys = [
        "coverage", "backbone_std", "dataset", "method_std", "domain",
        "repetition_seed",
    ]
    for key, group in rows.groupby(keys, sort=False):
        coverage, backbone, dataset, method, domain, seed = key
        base_splits = BASE_SPLITS[str(dataset)]
        step = group.assign(
            correlation_step=group["split_index"].astype(int) % base_splits
        )
        values = step.groupby("correlation_step")["value"].mean()
        if set(values.index) != set(range(base_splits)) or len(step) != base_splits:
            raise ValueError(f'incomplete or duplicated coverage APD curve: {key}')
        reference = float(references.loc[(backbone,dataset,domain)])
        output.append({
            "coverage": int(coverage),
            "backbone_std": backbone,
            "dataset": dataset,
            "method_std": method,
            "domain": domain,
            "repetition_seed": seed,
            "value": float((values.loc[1:] / reference).mean() - 1.0),
        })
    repetition = pd.DataFrame(output)
    if repetition.empty:
        raise ValueError("coverage APD could not be reconstructed")
    dataset = repetition.groupby(
        ["coverage", "backbone_std", "dataset", "method_std", "domain"],
        as_index=False,
    )["value"].mean()
    result = dataset.groupby(
        ["coverage", "backbone_std", "method_std", "domain"], as_index=False
    )["value"].mean()
    result["metric"] = "average_performance_drop"
    return result

def coverage_representation(rows: pd.DataFrame) -> pd.DataFrame:
    dataset = rows.groupby(
        ["coverage", "backbone_std", "dataset", "method_std", "endpoint"],
        as_index=False,
    )["value"].mean()
    return dataset.groupby(
        ["coverage", "backbone_std", "method_std", "endpoint"], as_index=False
    )["value"].mean()

def bootstrap_summary(
    frame: pd.DataFrame, group_columns: list[str], seed_offset: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + seed_offset)
    rows = []
    for key, group in frame.groupby(group_columns, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        values = group.sort_values("backbone_std")["value"].to_numpy(float)
        draws = values[rng.integers(0, len(values), size=(BOOTSTRAP_REPETITIONS, len(values)))].mean(axis=1)
        lo, hi = np.quantile(draws, [0.025, 0.975])
        row = dict(zip(group_columns, key))
        row.update({
            "mean": float(values.mean()), "median": float(np.median(values)),
            "ci_low": float(lo), "ci_high": float(hi), "n_backbones": len(values),
        })
        rows.append(row)
    return pd.DataFrame(rows)

class ExactSignFlip:
    def __init__(self, backbones: list[str]):
        self.backbones = backbones
        n = len(backbones)
        indices = np.arange(1 << n, dtype=np.uint32)
        bits = ((indices[:, None] >> np.arange(n, dtype=np.uint32)) & 1).astype(np.int8)
        self.signs = bits * 2 - 1

    def pvalue(self, differences: np.ndarray) -> float:
        observed = abs(float(differences.sum()))
        permuted = np.abs(self.signs @ differences)
        return float(np.mean(permuted >= observed - 1e-14))

def paired_statistics(
    frame: pd.DataFrame,
    metric_columns: list[str],
    axis_column: str,
    candidate: str,
    controls: list[str],
    test: ExactSignFlip,
    family_prefix: str,
) -> pd.DataFrame:
    rows = []
    grouped = frame.groupby(metric_columns, sort=False) if metric_columns else [((), frame)]
    for metric_values, metric_group in grouped:
        metric_values = metric_values if isinstance(metric_values, tuple) else (metric_values,)
        metric_meta = dict(zip(metric_columns, metric_values))
        for axis_value, axis_group in metric_group.groupby(axis_column, sort=False):
            pivot = axis_group.pivot(index="backbone_std", columns="method_std", values="value")
            pivot = pivot.reindex(test.backbones)
            for control in controls:
                if candidate not in pivot or control not in pivot:
                    continue
                paired = pivot[[candidate, control]].dropna()
                if len(paired) != len(test.backbones):
                    raise ValueError(f"unmatched paired units: {candidate} vs {control}")
                difference = (paired[candidate] - paired[control]).to_numpy(float)
                rng = np.random.default_rng(RANDOM_SEED + len(rows))
                draws = difference[
                    rng.integers(0, len(difference), size=(BOOTSTRAP_REPETITIONS, len(difference)))
                ].mean(axis=1)
                lo, hi = np.quantile(draws, [0.025, 0.975])
                try:
                    wilcoxon_p = float(wilcoxon(difference, method="auto").pvalue)
                except ValueError:
                    wilcoxon_p = 1.0
                rows.append({
                    **metric_meta,
                    axis_column: axis_value,
                    "candidate": candidate,
                    "control": control,
                    "mean_difference": float(difference.mean()),
                    "median_difference": float(np.median(difference)),
                    "ci_low": float(lo), "ci_high": float(hi),
                    "wins": int((difference > 1e-12).sum()),
                    "ties": int((np.abs(difference) <= 1e-12).sum()),
                    "losses": int((difference < -1e-12).sum()),
                    "exact_signflip_p": test.pvalue(difference),
                    "wilcoxon_p": wilcoxon_p,
                    "family": f"{family_prefix}:{metric_meta}:{candidate}",
                })
    result = pd.DataFrame(rows)
    adjusted = []
    adjusted_w = []
    for _, group in result.groupby("family", sort=False):
        adjusted.extend(zip(group.index, holm_adjust(group["exact_signflip_p"])))
        adjusted_w.extend(zip(group.index, holm_adjust(group["wilcoxon_p"])))
    result["holm_exact_p"] = np.nan
    result["holm_wilcoxon_p"] = np.nan
    for index, value in adjusted:
        result.loc[index, "holm_exact_p"] = value
    for index, value in adjusted_w:
        result.loc[index, "holm_wilcoxon_p"] = value
    return result.drop(columns="family")

def assert_inventory(frame: pd.DataFrame, method_column: str = "method_std") -> None:
    if set(frame["backbone_std"]) != set(EXPECTED_BACKBONES):
        raise ValueError("backbone inventory mismatch after aggregation")
    if set(frame[method_column]) != set(METHOD_ORDER):
        raise ValueError("method inventory mismatch after aggregation")
    if not np.isfinite(frame["value"]).all():
        raise ValueError("non-finite aggregate")

def apd_summary_table(ba_summary: pd.DataFrame, apd_summary: pd.DataFrame) -> pd.DataFrame:
    def wide(source: pd.DataFrame, prefix: str) -> pd.DataFrame:
        pieces = []
        for statistic in ["mean", "median", "ci_low", "ci_high"]:
            part = source.pivot(index="method_std", columns="domain", values=statistic)
            part.columns = [f"{domain}_{prefix}_{statistic}" for domain in part.columns]
            pieces.append(part)
        return pd.concat(pieces, axis=1)

    result = pd.concat([wide(ba_summary, "BA"), wide(apd_summary, "APD")], axis=1)
    result = result.reindex(METHOD_ORDER).reset_index().rename(columns={"method_std": "Method"})
    result.insert(1, "Condition", result["Method"].map(CONDITION))
    result.insert(1, "Display method", result["Method"].map(DISPLAY_NAMES))
    result["ID_OOD_BA_gap"] = result["ID_BA_mean"] - result["OOD_BA_mean"]
    return result

def representation_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for statistic in ["mean", "median", "ci_low", "ci_high"]:
        part = summary.pivot(index="method_std", columns="endpoint", values=statistic)
        part.columns = [f"{endpoint}_{statistic}" for endpoint in part.columns]
        pieces.append(part)
    result = pd.concat(pieces, axis=1).reindex(METHOD_ORDER).reset_index()
    result = result.rename(columns={"method_std": "Method"})
    result.insert(1, "Condition", result["Method"].map(CONDITION))
    result.insert(1, "Display method", result["Method"].map(DISPLAY_NAMES))
    return result

def json_records(frame: pd.DataFrame) -> list[list[object]]:
    clean = frame.replace({np.nan: None, np.inf: None, -np.inf: None})
    return clean.astype(object).where(pd.notna(clean), None).values.tolist()

def workbook_sheet(name: str, frame: pd.DataFrame, formats: dict[str, str] | None = None) -> dict:
    return {
        "name": name,
        "columns": [
            {"name": str(column), "format": (formats or {}).get(str(column), "General")}
            for column in frame.columns
        ],
        "rows": json_records(frame),
    }

def build_payload(
    output: Path,
    apd_summary: pd.DataFrame,
    apd_paired: pd.DataFrame,
    ba_backbone: pd.DataFrame,
    apd_backbone: pd.DataFrame,
    rep_summary: pd.DataFrame,
    rep_paired: pd.DataFrame,
    rep_backbone: pd.DataFrame,
    provenance: pd.DataFrame,
) -> Path:
    percent_columns = {
        column: "0.000%"
        for frame in [apd_summary, apd_paired, ba_backbone, apd_backbone]
        for column in frame.columns
        if any(token in str(column) for token in ["BA", "APD", "difference", "ci_", "value", "gap"])
        and not any(token in str(column) for token in ["p", "wins", "ties", "losses"])
    }
    p_formats = {column: "0.000000" for column in apd_paired.columns if column.endswith("_p")}
    rep_formats = {
        column: "0.0000"
        for frame in [rep_summary, rep_paired, rep_backbone]
        for column in frame.columns
        if any(token in str(column) for token in ["mean", "median", "ci_", "difference", "value"])
    }
    rep_p_formats = {column: "0.000000" for column in rep_paired.columns if column.endswith("_p")}

    overview = pd.DataFrame([
        ["Report", "PathoROB DELTA-v1.3 p2 final results"],
        ["Backbones", 20],
        ["Representation datasets", 4],
        ["Predictive datasets", 3],
        ["Primary method", "DELTA-v1.3 p2"],
        ["Statistical unit", "Backbone; datasets averaged within backbone"],
        ["Confidence interval", "10,000 backbone bootstrap resamples"],
        ["Hypothesis test", "Exact paired sign-flip; Wilcoxon corroboration"],
        ["Multiplicity", "Holm correction within declared comparison family"],
        ["Coverage status", "p2 primary report; complete p1--p50 sweep retained separately"],
    ], columns=["Field", "Value"])

    payload = {
        "title": "PathoROB DELTA-v1.3 p2",
        "sheets": [
            workbook_sheet("Overview", overview),
            workbook_sheet("APD Summary", apd_summary, percent_columns),
            workbook_sheet("APD Paired", apd_paired, {**percent_columns, **p_formats}),
            workbook_sheet("BA Backbone", ba_backbone, percent_columns),
            workbook_sheet("APD Backbone", apd_backbone, percent_columns),
            workbook_sheet("Representation Summary", rep_summary, rep_formats),
            workbook_sheet("Representation Paired", rep_paired, {**rep_formats, **rep_p_formats}),
            workbook_sheet("Representation Backbone", rep_backbone, rep_formats),
            workbook_sheet("Provenance", provenance),
        ],
    }
    path = output / "workbook_payload.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path

def write_csv_tables(output: Path, tables: dict[str, pd.DataFrame]) -> list[Path]:
    directory = output / "tables"
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, frame in tables.items():
        path = directory / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths.append(path)
    return paths

def write_readme(output: Path) -> Path:
    path = output / "README.md"
    path.write_text(
        "# PathoROB DELTA-v1.3 p2 report\n\n"
        "This directory contains the frozen 20-backbone p2 paper report.\n\n"
        "- `PathoROB_DELTA_v1_3_p2_results.xlsx`: formal workbook.\n"
        "- `tables/`: auditable CSV tables used by the workbook.\n"
        "- `report_manifest.json`: input hashes, protocol, and output inventory.\n\n"
        "This is the frozen p2 primary report; the complete p1--p50 coverage sweep is retained separately.\n",
        encoding="utf-8",
    )
    return path

def build_coverage_payload(
    output: Path,
    predictive_summary: pd.DataFrame,
    representation_summary: pd.DataFrame,
    predictive_backbone: pd.DataFrame,
    representation_backbone: pd.DataFrame,
    provenance: pd.DataFrame,
) -> Path:
    overview = pd.DataFrame([
        ["Report", "PathoROB DELTA-v1.3 bank-coverage sensitivity"],
        ["Backbones", 20],
        ["Datasets", 3],
        ["Coverage per bank role", "1, 2, 5, 10, 20, 30, 40, 50%"],
        ["Bank rule", "Disjoint style/fitting banks; one-to-one pairing"],
        ["Statistical unit", "Backbone; datasets averaged within backbone"],
        ["Confidence interval", "10,000 backbone bootstrap resamples"],
        ["Selection status", "p2 was frozen before the corrected coverage sweep"],
    ], columns=["Field", "Value"])
    percent_columns = {
        column: "0.000%"
        for frame in [predictive_summary, predictive_backbone]
        for column in frame.columns
        if column in {"mean", "median", "ci_low", "ci_high", "value"}
    }
    score_columns = {
        column: "0.0000"
        for frame in [representation_summary, representation_backbone]
        for column in frame.columns
        if column in {"mean", "median", "ci_low", "ci_high", "value"}
    }
    payload = {
        "title": "PathoROB DELTA-v1.3 bank coverage",
        "sheets": [
            workbook_sheet("Overview", overview),
            workbook_sheet("Predictive Summary", predictive_summary, percent_columns),
            workbook_sheet("Predictive Backbone", predictive_backbone, percent_columns),
            workbook_sheet("Representation Summary", representation_summary, score_columns),
            workbook_sheet("Representation Backbone", representation_backbone, score_columns),
            workbook_sheet("Provenance", provenance),
        ],
    }
    path = output / "workbook_payload.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path

def run_coverage_report(root: Path, output_dir: Path | None=None) -> None:
    experiment = root / PRIMARY_EXPERIMENT_RELATIVE
    output = output_dir if output_dir is not None else root / COVERAGE_OUTPUT_RELATIVE
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Use a new or empty output directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    print('COVERAGE_REPORT_START', flush=True)
    apd_rows, apd_path = load_coverage_candidate(experiment, 'apd')
    rep_rows, rep_path = load_coverage_candidate(experiment, 'representation')
    ba_backbone = coverage_balanced_accuracy(apd_rows)
    raw_reference_rows = load_baseline(root / BASELINE_RELATIVE)
    apd_backbone = coverage_average_performance_drop(apd_rows, raw_reference_rows)
    rep_backbone = coverage_representation(rep_rows)
    expected_predictive = 20 * len(BANK_COVERAGES) * 2 * 2
    expected_representation = 20 * len(BANK_COVERAGES) * 2 * len(REP_ENDPOINTS)
    if len(ba_backbone) != expected_predictive or len(apd_backbone) != expected_predictive:
        raise ValueError('coverage predictive inventory mismatch')
    if len(rep_backbone) != expected_representation:
        raise ValueError('coverage representation inventory mismatch')
    ba_summary = bootstrap_summary(ba_backbone, ['coverage', 'method_std', 'domain'], 31)
    ba_summary['panel'] = 'BA_' + ba_summary['domain'].astype(str)
    apd_summary = bootstrap_summary(apd_backbone, ['coverage', 'method_std', 'domain'], 32)
    apd_summary['panel'] = 'APD_' + apd_summary['domain'].astype(str)
    predictive_summary = pd.concat([ba_summary, apd_summary], ignore_index=True)
    rep_summary = bootstrap_summary(rep_backbone, ['coverage', 'method_std', 'endpoint'], 33)
    rep_summary['panel'] = rep_summary['endpoint'].astype(str)
    print('COVERAGE_REPORT_STATISTICS_COMPLETE', flush=True)
    tables = {'predictive_summary': predictive_summary, 'predictive_by_backbone': pd.concat([ba_backbone, apd_backbone], ignore_index=True), 'representation_summary': rep_summary, 'representation_by_backbone': rep_backbone}
    table_paths = write_csv_tables(output, tables)
    provenance = pd.DataFrame([['Primary experiment', str(experiment.resolve())], ['Coverage set', ','.join(map(str, BANK_COVERAGES))], ['Backbones', 20], ['Datasets', ','.join(COVERAGE_DATASETS)], ['Bank rule', 'Disjoint one-to-one style/fitting banks'], ['Rank rule', 'Permutation-null-adjusted energy, 90% cumulative threshold'], ['APD source SHA256', sha256_file(apd_path)], ['Representation source SHA256', sha256_file(rep_path)]], columns=['Field', 'Value'])
    payload = build_coverage_payload(output, predictive_summary, rep_summary, tables['predictive_by_backbone'], rep_backbone, provenance)
    workbook = write_workbook(payload, output / 'PathoROB_DELTA_v1_3_bank_coverage.xlsx')
    (output / 'README.md').write_text('# PathoROB DELTA-v1.3 bank-coverage report\n\nComplete corrected p1, p2, p5, p10, p20, p30, p40, and p50 sensitivity analysis.\n', encoding='utf-8')
    manifest = {'status': 'complete', 'backbones': 20, 'coverages': BANK_COVERAGES, 'generated_files': sorted((str(path.relative_to(output)) for path in [*table_paths, payload, workbook, output / 'README.md'])), 'apd_source_sha256': sha256_file(apd_path), 'representation_source_sha256': sha256_file(rep_path)}
    (output / 'report_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
    print('COVERAGE_REPORT_COMPLETE', flush=True)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--output-dir', type=Path, help='New or empty report directory')
    parser.add_argument('--mode', choices=['p2', 'coverage'], default='p2')
    args = parser.parse_args()
    if args.mode == 'coverage':
        if not RUN_COVERAGE:
            raise RuntimeError('Coverage reporting is disabled')
        run_coverage_report(args.project_root, args.output_dir)
        return
    if not RUN_P2:
        raise RuntimeError('p2 reporting is disabled')
    root = args.project_root
    experiment = root / PRIMARY_EXPERIMENT_RELATIVE
    baseline_path = root / BASELINE_RELATIVE
    tcga2_delta_path = root / TCGA2_DELTA_RELATIVE
    tcga2_baseline_path = root / TCGA2_BASELINE_RELATIVE
    output = args.output_dir if args.output_dir is not None else root / P2_OUTPUT_RELATIVE
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Use a new or empty output directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    print('P2_REPORT_START', flush=True)
    candidate_apd, candidate_apd_paths = load_candidate(experiment, 'apd')
    candidate_rep = load_canonical_representation(tcga2_delta_path)
    baseline = load_baseline(baseline_path)
    tcga2_baseline = load_tcga2_baseline(tcga2_baseline_path)
    print('P2_REPORT_INPUTS_COMPLETE', flush=True)
    apd_rows = combine_apd(baseline, candidate_apd)
    rep_rows = combine_representation(pd.concat([baseline, tcga2_baseline], ignore_index=True), candidate_rep)
    ba_backbone = backbone_balanced_accuracy(apd_rows)
    apd_backbone = backbone_average_performance_drop(apd_rows)
    rep_backbone = backbone_representation(rep_rows)
    for frame in [ba_backbone, apd_backbone, rep_backbone]:
        assert_inventory(frame)
    print('P2_REPORT_AGGREGATION_COMPLETE', flush=True)
    ba_summary = bootstrap_summary(ba_backbone, ['method_std', 'domain'], 1)
    apd_metric_summary = bootstrap_summary(apd_backbone, ['method_std', 'domain'], 2)
    rep_metric_summary = bootstrap_summary(rep_backbone, ['method_std', 'endpoint'], 3)
    test = ExactSignFlip(sorted(EXPECTED_BACKBONES))
    controls = ['RAW', 'DANN', 'R-RAW', 'R-DANN', 'ComBat-T', 'R-ComBat-T']
    apd_stats_frames = []
    for metric_name, frame in [('balanced_accuracy', ba_backbone), ('average_performance_drop', apd_backbone)]:
        for candidate in ['DELTA-v1.3 p2', 'R-DELTA-v1.3 p2']:
            stats = paired_statistics(frame, ['metric'], 'domain', candidate, controls, test, f'apd:{metric_name}')
            apd_stats_frames.append(stats)
    apd_paired = pd.concat(apd_stats_frames, ignore_index=True)
    rep_stats_frames = []
    representation_controls = ['RAW', 'R-RAW', 'ComBat-T', 'R-ComBat-T']
    for candidate in ['DELTA-v1.3 p2', 'R-DELTA-v1.3 p2']:
        rep_stats_frames.append(paired_statistics(rep_backbone, [], 'endpoint', candidate, representation_controls, test, 'representation'))
    rep_paired = pd.concat(rep_stats_frames, ignore_index=True)
    print('P2_REPORT_STATISTICS_COMPLETE', flush=True)
    apd_summary = apd_summary_table(ba_summary, apd_metric_summary)
    rep_summary = representation_summary_table(rep_metric_summary)
    ba_export = ba_backbone.rename(columns={'backbone_std': 'Backbone', 'method_std': 'Method', 'domain': 'Domain', 'value': 'BA_value'})[['Backbone', 'Method', 'Domain', 'BA_value']]
    apd_export = apd_backbone.rename(columns={'backbone_std': 'Backbone', 'method_std': 'Method', 'domain': 'Domain', 'value': 'APD_value'})[['Backbone', 'Method', 'Domain', 'APD_value']]
    rep_export = rep_backbone.rename(columns={'backbone_std': 'Backbone', 'method_std': 'Method', 'endpoint': 'Endpoint', 'value': 'Score_value', 'datasets_averaged': 'Datasets_averaged'})[['Backbone', 'Method', 'Endpoint', 'Score_value', 'Datasets_averaged']]
    sources = [baseline_path, tcga2_delta_path, tcga2_baseline_path, *candidate_apd_paths]
    provenance_rows = [['Protocol SHA256', '0426287241a71974587481cf21bcdcfaae9612d8fe462f121c88353960913856'], ['Baseline source', str(baseline_path.resolve())], ['TCGA2x2 baseline source', str(tcga2_baseline_path.resolve())], ['TCGA2x2 DELTA source', str(tcga2_delta_path.resolve())], ['Primary experiment', str(experiment.resolve())], ['Primary coverage', '2% per disjoint bank role'], ['Bank rule', 'One-to-one style/fitting pairing; disjoint banks'], ['Rank rule', 'Permutation-null-adjusted energy, 90% cumulative threshold'], ['Statistical unit', 'Backbone (n=20); datasets averaged within backbone'], ['Representation datasets by endpoint', json.dumps(REPRESENTATION_DATASETS_BY_ENDPOINT, sort_keys=True)], ['Predictive datasets', ','.join(PREDICTIVE_DATASETS)], ['TCGA2x2 exclusions', 'clustering and DANN are not applicable'], ['APD formula', 'PathoROB Eq.8: mean_r mean_s>0(score_method[r,s] / mean_r(score_RAW[r,0])) - 1; reference separate per backbone/dataset/domain'], ['Input aggregate SHA256', hashlib.sha256(''.join((sha256_file(p) for p in sources)).encode()).hexdigest()], ['Coverage report', 'Complete p1,2,5,10,20,30,40,50 sweep retained separately']]
    provenance = pd.DataFrame(provenance_rows, columns=['Field', 'Value'])
    tables = {'apd_summary': apd_summary, 'apd_paired_statistics': apd_paired, 'balanced_accuracy_by_backbone': ba_export, 'average_performance_drop_by_backbone': apd_export, 'representation_summary': rep_summary, 'representation_paired_statistics': rep_paired, 'representation_by_backbone': rep_export, 'provenance': provenance}
    table_paths = write_csv_tables(output, tables)
    payload = build_payload(output, apd_summary, apd_paired, ba_export, apd_export, rep_summary, rep_paired, rep_export, provenance)
    workbook = write_workbook(payload, output / 'PathoROB_DELTA_v1_3_p2_results.xlsx')
    readme = write_readme(output)
    manifest = {'status': 'complete', 'generated_files': sorted((str(path.relative_to(output)) for path in [*table_paths, payload, workbook, readme])), 'backbones': 20, 'representation_datasets_by_endpoint': REPRESENTATION_DATASETS_BY_ENDPOINT, 'predictive_datasets': PREDICTIVE_DATASETS, 'coverage': PRIMARY_BANK_COVERAGE, 'coverage_report_enabled': False, 'baseline_sha256': sha256_file(baseline_path), 'input_aggregate_sha256': provenance.loc[provenance['Field'].eq('Input aggregate SHA256'), 'Value'].iloc[0]}
    (output / 'report_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
    print('P2_REPORT_COMPLETE', flush=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
