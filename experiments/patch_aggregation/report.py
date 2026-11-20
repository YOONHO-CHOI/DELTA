"""Numerical patch-aggregation tables and paired inference."""
from __future__ import annotations
import argparse
import hashlib
import math
import os
import platform
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.dimensions import ColumnDimension
from scipy.stats import wilcoxon
from experiments.common.resources import repository_root
from .statistics import exact_signed_rank_p, holm_adjust as strict_holm_adjust

TARGET_METRIC = os.environ.get("PATHROB_PATCH_AGG_TARGET_METRIC", "balanced_accuracy")
SUPPORTED_METRICS = {
    "balanced_accuracy": {"label": "Balanced accuracy", "higher_is_better": True},
    "macro_f1": {"label": "Macro-F1", "higher_is_better": True},
    "auroc": {"label": "AUROC", "higher_is_better": True},
    "auprc": {"label": "AUPRC", "higher_is_better": True},
    "brier": {"label": "Brier score", "higher_is_better": False},
    "ece_10bin": {"label": "ECE (10 bins)", "higher_is_better": False},
}
TASK_ORDER = [
    "camelyon_slide_tumor",
    "tcga4_four_class",
    "tcga_nsclc_subtype",
]
TASK_LABELS = {
    "camelyon_slide_tumor": "Camelyon tumor",
    "tcga4_four_class": "TCGA 4x4",
    "tcga_nsclc_subtype": "TCGA NSCLC subtype",
}
BACKBONE_ORDER = [
    "Original UNI",
    "Phikon-v2",
    "Virchow2",
    "CONCH",
    "CONCH v1.5",
    "UNI2-h",
    "H-optimus-0",
    "EXAONE-Path-2.5",
    "GPFM",
    "MOOZY patch encoder (Lunit DINO)",
    "Prov-GigaPath",
    "MUSK",
    "H0-mini",
    "RetCCL",
    "BEPH",
    "HIPT",
    "Kaiko ViT-B/8",
    "Ciga",
    "Virchow",
    "CTransPath",
]
METHOD_ORDER = [
    "RAW",
    "Reinhard-RAW",
    "DANN",
    "Reinhard-DANN",
    "DELTA",
    "Reinhard-DELTA",
    "ComBat-transductive",
    "Reinhard-ComBat-transductive",
]
TRANSDUCTIVE_METHODS = {"ComBat-transductive", "Reinhard-ComBat-transductive"}
BRANCH_COMPARATORS = {
    "DELTA": ("RAW", "Reinhard-RAW", "DANN", "ComBat-transductive"),
    "Reinhard-DELTA": ("RAW", "Reinhard-RAW", "Reinhard-DANN", "Reinhard-ComBat-transductive"),
}
RANK_ELIGIBLE_METHODS = {
    "RAW",
    "Reinhard-RAW",
    "DANN",
    "Reinhard-DANN",
    "DELTA",
    "Reinhard-DELTA",
}
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 42
CI_LEVEL = 0.95
EXCEL_STYLE = {
    "font": "Arial",
    "body_size": 9,
    "header_size": 9,
    "title_size": 14,
    "caption_size": 9,
    "thin_color": "808080",
    "black": "000000",
    "white": "FFFFFF",
    "na_text": "NA",
    "first_place_bold": True,
    "second_place_underline": True,
}
REQUIRED_COLUMNS = {
    "completed_at_kst",
    "backbone",
    "backbone_key",
    "task_id",
    "dataset",
    "split_index",
    "split_id",
    "seed",
    "method",
    "domain",
    "metric",
    "value",
    "status",
}
RAW_METHOD_MAP = {
    "RAW": "RAW",
    "Reinhard-LUSC500": "Reinhard-RAW",
    "Reinhard-RAW": "Reinhard-RAW",
    "DANN": "DANN",
    "Reinhard-LUSC500-DANN": "Reinhard-DANN",
    "Reinhard-DANN": "Reinhard-DANN",
    "ComBat-inductive": "ComBat-inductive",
    "Reinhard-LUSC500-ComBat": "Reinhard-ComBat-inductive",
    "ComBat-transductive": "ComBat-transductive",
    "Reinhard-ComBat-transductive": "Reinhard-ComBat-transductive",
    "DELTA-v1.3-p2": "DELTA",
    "Reinhard-DELTA-v1.3-p2": "Reinhard-DELTA",
}
EXPECTED_SOURCE_METHODS = set(METHOD_ORDER)
EXPECTED_SOURCE_ROWS = 20 * 3 * 100 * 2 * 6 * len(METHOD_ORDER)
THIN = Side(style="thin", color=EXCEL_STYLE["thin_color"])
MEDIUM = Side(style="medium", color=EXCEL_STYLE["black"])
NO_FILL = PatternFill(fill_type=None)

def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()

def canonical_method(backbone: str, raw_method: str) -> str | None:
    """Return the publication method under the universal projection contract."""
    del backbone
    return RAW_METHOD_MAP.get(raw_method)

def method_protocol(method: str) -> str:
    if method == "RAW":
        return "Unmodified representation"
    if method == "Reinhard-RAW":
        return "Image normalization; no feature correction"
    if method in {"DANN", "Reinhard-DANN"}:
        return "Train-only inductive"
    if method in {"ComBat-inductive", "Reinhard-ComBat-inductive"}:
        return "Train-only; unseen-center applicability required"
    if method in {"DELTA", "Reinhard-DELTA"}:
        return "Train-only, label-free; universal centered Euclidean projection"
    if method in TRANSDUCTIVE_METHODS:
        return "Transductive evaluation-cohort adjustment; included in prespecified Holm family; not deployment-equivalent"
    return ""

def scientific_key_columns(frame: pd.DataFrame) -> list[str]:
    candidates = [
        "backbone",
        "backbone_key",
        "task_id",
        "dataset",
        "split_index",
        "split_id",
        "seed",
        "method",
        "domain",
        "metric",
    ]
    return [c for c in candidates if c in frame.columns]

def load_source(path: Path) -> tuple[pd.DataFrame, dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Canonical downstream CSV not found: {path}\n"
            "Set PATHROB_PATCH_AGG_INPUT_CSV to a verified copy."
        )
    frame = pd.read_csv(path, low_memory=False)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Required columns missing: {sorted(missing)}")

    mapped_methods = frame["method"].map(RAW_METHOD_MAP)
    if mapped_methods.isna().any():
        raise ValueError("Unmapped source method; canonical transductive labels must be explicit")
    frame = frame[mapped_methods.isin(METHOD_ORDER)].copy()
    observed_source_methods = set(frame["method"].map(RAW_METHOD_MAP).unique())
    if observed_source_methods != EXPECTED_SOURCE_METHODS:
        raise ValueError(
            "Canonical v1.3-p2 eight-method inventory mismatch; explicit transductive ComBat is required in both branches, never substituted by inductive rows. "
            f"missing={sorted(EXPECTED_SOURCE_METHODS-observed_source_methods)}; "
            f"extra={sorted(observed_source_methods-EXPECTED_SOURCE_METHODS)}"
        )
    if len(frame) != EXPECTED_SOURCE_ROWS:
        raise ValueError(
            f"Canonical v1.3-p2 row inventory mismatch: {len(frame)}/{EXPECTED_SOURCE_ROWS}"
        )

    source_rows = len(frame)
    frame["_completed_at"] = pd.to_datetime(
        frame["completed_at_kst"], errors="coerce"
    )
    frame["_status_priority"] = frame["status"].eq("complete").astype(int)
    keys = scientific_key_columns(frame)
    frame = (
        frame.sort_values(["_status_priority", "_completed_at"], kind="stable")
        .drop_duplicates(keys, keep="last")
        .reset_index(drop=True)
    )
    deduplicated_rows = len(frame)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["display_method"] = [
        canonical_method(str(b), str(m))
        for b, m in zip(frame["backbone"], frame["method"])
    ]
    unknown_methods = sorted(
        frame.loc[frame["display_method"].isna(), "method"].astype(str).unique()
    )
    if unknown_methods:
        raise ValueError(f"Unmapped canonical methods: {unknown_methods}")

    complete = frame[
        frame["status"].eq("complete") & np.isfinite(frame["value"])
    ].copy()
    counts = complete.groupby(["backbone", "task_id", "display_method", "domain", "metric"])["split_id"].nunique()
    if len(counts) != 20 * 3 * len(METHOD_ORDER) * 2 * 6 or not counts.eq(100).all():
        raise ValueError("Canonical inference requires a complete paired 20-encoder, three-task, 100-split panel for all eight methods")
    observed_backbones = list(dict.fromkeys(frame["backbone"].astype(str)))
    observed_tasks = set(frame["task_id"].astype(str))
    if set(BACKBONE_ORDER) != set(observed_backbones):
        raise ValueError(
            "The frozen 20-backbone contract does not match the source. "
            f"missing={sorted(set(BACKBONE_ORDER)-set(observed_backbones))}; "
            f"extra={sorted(set(observed_backbones)-set(BACKBONE_ORDER))}"
        )
    if set(TASK_ORDER) != observed_tasks:
        raise ValueError(
            f"Task contract mismatch: observed={sorted(observed_tasks)}"
        )

    qc = {
        "source_rows": source_rows,
        "deduplicated_rows": deduplicated_rows,
        "duplicate_or_superseded_rows": source_rows - deduplicated_rows,
        "complete_finite_rows": len(complete),
        "nonfinite_complete_rows": int(
            (frame["status"].eq("complete") & ~np.isfinite(frame["value"])).sum()
        ),
        "not_applicable_rows": int(frame["status"].eq("not_applicable").sum()),
        "backbone_count": frame["backbone"].nunique(),
        "task_count": frame["task_id"].nunique(),
        "split_count_min": int(
            frame.groupby(["backbone", "task_id"])["split_index"].nunique().min()
        ),
        "split_count_max": int(
            frame.groupby(["backbone", "task_id"])["split_index"].nunique().max()
        ),
    }
    return frame, qc

def bootstrap_ci(
    values: Sequence[float],
    *,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
    level: float = CI_LEVEL,
) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (math.nan, math.nan)
    if x.size == 1:
        return (float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    batch = min(1000, repetitions)
    means = []
    remaining = repetitions
    while remaining > 0:
        current = min(batch, remaining)
        sampled = rng.integers(0, x.size, size=(current, x.size))
        means.append(x[sampled].mean(axis=1))
        remaining -= current
    boot = np.concatenate(means)
    alpha = (1.0 - level) / 2.0
    return tuple(np.quantile(boot, [alpha, 1.0 - alpha]).astype(float))

def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    finite_idx = np.flatnonzero(np.isfinite(p))
    if finite_idx.size == 0:
        return out
    order = finite_idx[np.argsort(p[finite_idx])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        adjusted = min(1.0, (m - rank) * p[idx])
        running = max(running, adjusted)
        out[idx] = running
    return out

def build_paired_frame(complete: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "backbone",
        "backbone_key",
        "task_id",
        "dataset",
        "split_index",
        "split_id",
        "seed",
        "domain",
        "metric",
    ]
    raw = (
        complete[complete["display_method"].eq("RAW")][keys + ["value"]]
        .rename(columns={"value": "raw_value"})
        .drop_duplicates(keys)
    )
    paired = complete.merge(raw, on=keys, how="left", validate="many_to_one")
    paired = paired[np.isfinite(paired["raw_value"])].copy()
    higher = paired["metric"].map(
        {m: cfg["higher_is_better"] for m, cfg in SUPPORTED_METRICS.items()}
    )
    if higher.isna().any():
        raise ValueError(
            "Direction missing for metrics: "
            f"{sorted(paired.loc[higher.isna(), 'metric'].unique())}"
        )
    direction = np.where(higher.to_numpy(bool), 1.0, -1.0)
    paired["improvement"] = direction * (
        paired["value"].to_numpy(float) - paired["raw_value"].to_numpy(float)
    )
    paired["improvement_pp"] = 100.0 * paired["improvement"]
    denominator = paired["raw_value"].abs().to_numpy(float)
    paired["matched_relative_gain_pct"] = np.where(
        denominator > 0,
        100.0 * paired["improvement"].to_numpy(float) / denominator,
        np.nan,
    )
    return paired

def build_condition_means(paired: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "backbone",
        "backbone_key",
        "task_id",
        "dataset",
        "display_method",
        "domain",
        "metric",
    ]
    condition = (
        paired.groupby(group_cols, sort=False)
        .agg(
            split_n=("value", "size"),
            value_mean=("value", "mean"),
            value_std=("value", "std"),
            value_median=("value", "median"),
            raw_mean=("raw_value", "mean"),
            improvement_pp=("improvement_pp", "mean"),
            relative_gain_pct=("matched_relative_gain_pct", "mean"),
        )
        .reset_index()
        .rename(columns={"display_method": "method"})
    )
    condition["task_label"] = condition["task_id"].map(TASK_LABELS)
    condition["protocol"] = condition["method"].map(method_protocol)
    condition["rank_eligible"] = condition["method"].isin(RANK_ELIGIBLE_METHODS)
    condition["transductive"] = condition["method"].isin(TRANSDUCTIVE_METHODS)
    return condition

def summarize_scope(
    condition: pd.DataFrame,
    metric: str,
    task_id: str | None = None,
) -> pd.DataFrame:
    x = condition[condition["metric"].eq(metric)].copy()
    if task_id is not None:
        x = x[x["task_id"].eq(task_id)]
    else:
        # The inferential unit is the backbone, after equal weighting of the
        # three prespecified tasks.  Do not treat backbone x task conditions
        # as 60 independent observations.
        x = (
            x.groupby(
                ["backbone", "backbone_key", "method", "domain", "metric"],
                sort=False,
            )
            .agg(
                split_n=("split_n", "sum"),
                value_mean=("value_mean", "mean"),
                improvement_pp=("improvement_pp", "mean"),
                relative_gain_pct=("relative_gain_pct", "mean"),
            )
            .reset_index()
        )
    rows = []
    for method in METHOD_ORDER:
        for domain in ("ID", "OOD"):
            g = x[x["method"].eq(method) & x["domain"].eq(domain)]
            value = g["value_mean"].to_numpy(float)
            delta = g["improvement_pp"].to_numpy(float)
            relative = g["relative_gain_pct"].to_numpy(float)
            value_ci = bootstrap_ci(value, seed=BOOTSTRAP_SEED + len(rows) * 17)
            delta_ci = bootstrap_ci(delta, seed=BOOTSTRAP_SEED + len(rows) * 17 + 1)
            relative_ci = bootstrap_ci(
                relative, seed=BOOTSTRAP_SEED + len(rows) * 17 + 2
            )
            expected = len(BACKBONE_ORDER)
            rows.append(
                {
                    "scope": TASK_LABELS.get(task_id, "All tasks"),
                    "task_id": task_id or "ALL",
                    "method": method,
                    "protocol": method_protocol(method),
                    "domain": domain,
                    "condition_n": len(g),
                    "expected_condition_n": expected,
                    "coverage_pct": 100.0 * len(g) / expected if expected else math.nan,
                    "value_mean": float(np.mean(value)) if len(value) else math.nan,
                    "value_ci_low": value_ci[0],
                    "value_ci_high": value_ci[1],
                    "delta_pp_mean": float(np.mean(delta)) if len(delta) else math.nan,
                    "delta_pp_median": float(np.median(delta)) if len(delta) else math.nan,
                    "delta_pp_ci_low": delta_ci[0],
                    "delta_pp_ci_high": delta_ci[1],
                    "wins": int(np.sum(delta > 0)) if len(delta) else 0,
                    "ties": int(np.sum(delta == 0)) if len(delta) else 0,
                    "losses": int(np.sum(delta < 0)) if len(delta) else 0,
                    "relative_gain_mean_pct": (
                        float(np.mean(relative)) if len(relative) else math.nan
                    ),
                    "relative_gain_ci_low_pct": relative_ci[0],
                    "relative_gain_ci_high_pct": relative_ci[1],
                    "rank_eligible": method in RANK_ELIGIBLE_METHODS,
                    "transductive": method in TRANSDUCTIVE_METHODS,
                }
            )
    return pd.DataFrame(rows)

def pairwise_statistics(condition: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Final S14: four comparators per branch; 12 contrasts for task families."""
    current = condition[condition["metric"].eq(metric)]
    scopes = [("All tasks", None)] + [(TASK_LABELS[t], t) for t in TASK_ORDER]
    rows = []
    for scope_label, task_id in scopes:
        selected = current if task_id is None else current[current["task_id"].eq(task_id)]
        if selected.duplicated(["backbone", "task_id", "method", "domain"]).any():
            raise ValueError("Duplicate encoder/task/method keys before inference")
        for target_index, (target, comparators) in enumerate(BRANCH_COMPARATORS.items()):
            for domain_index, domain in enumerate(("ID", "OOD")):
                x = selected[selected["domain"].eq(domain)]
                for comparator_index, reference in enumerate(comparators):
                    wide = x.pivot(index=["backbone", "task_id"], columns="method", values="value_mean")
                    if target not in wide or reference not in wide:
                        raise ValueError(f"Missing canonical branch comparison: {target} vs {reference}; never substitute inductive ComBat")
                    paired = wide[[target, reference]].dropna()
                    expected_tasks = 3 if task_id is None else 1
                    if len(paired) != 20 * expected_tasks or not paired.groupby(level="backbone").size().eq(expected_tasks).all():
                        raise ValueError(f"Incomplete matched task/encoder panel: {target} vs {reference}")
                    effect = (paired[target] - paired[reference]).groupby(level="backbone").mean().to_numpy(float)
                    seed = 42 if task_id is None else (4200 + TASK_ORDER.index(task_id) * 100 +
                        comparator_index * 10 + target_index + domain_index * 1000)
                    ci = bootstrap_ci(effect * 100, seed=seed)
                    rows.append({
                        "scope": scope_label, "task_id": task_id or "ALL", "domain": domain,
                        "metric": metric, "target": target, "reference": reference,
                        "comparison": f"{target} vs {reference}", "method": reference,
                        "protocol": method_protocol(reference), "condition_n": 20,
                        "expected_condition_n": 20,
                        "mean_delta_pp": float(effect.mean() * 100),
                        "median_delta_pp": float(np.median(effect) * 100),
                        "ci_low_pp": ci[0], "ci_high_pp": ci[1],
                        "wins": int((effect > 0).sum()), "ties": int((effect == 0).sum()),
                        "losses": int((effect < 0).sum()),
                        "favourable_side": "positive" if SUPPORTED_METRICS[metric]["higher_is_better"] else "negative",
                        "wilcoxon_p": exact_signed_rank_p(effect),
                        "rank_eligible": reference in RANK_ELIGIBLE_METHODS,
                        "transductive": reference in TRANSDUCTIVE_METHODS,
                    })
    result = pd.DataFrame(rows)
    result["holm_p"] = np.nan
    result["holm_family_size"] = 0
    result["family_scope"] = np.where(result.task_id.eq("ALL"), "pooled", "three_tasks")
    for (_, _, family_scope), indices in result.groupby(["target", "domain", "family_scope"]).groups.items():
        expected = 4 if family_scope == "pooled" else 12
        if len(indices) != expected:
            raise ValueError("Incomplete prespecified branchwise Holm family")
        result.loc[indices, "holm_p"] = strict_holm_adjust(result.loc[indices, "wilcoxon_p"])
        result.loc[indices, "holm_family_size"] = expected
    return result

def backbone_summary(condition: pd.DataFrame, metric: str) -> pd.DataFrame:
    x = condition[condition["metric"].eq(metric)].copy()
    rows = []
    for backbone in BACKBONE_ORDER:
        for method in METHOD_ORDER:
            row = {
                "backbone": backbone,
                "method": method,
                "protocol": method_protocol(method),
                "rank_eligible": method in RANK_ELIGIBLE_METHODS,
                "transductive": method in TRANSDUCTIVE_METHODS,
            }
            for domain in ("ID", "OOD"):
                g = x[
                    x["backbone"].eq(backbone)
                    & x["method"].eq(method)
                    & x["domain"].eq(domain)
                ]
                row[f"{domain}_task_n"] = len(g)
                row[f"{domain}_split_n"] = int(g["split_n"].sum()) if len(g) else 0
                row[f"{domain}_value_mean"] = (
                    float(g["value_mean"].mean()) if len(g) else math.nan
                )
                row[f"{domain}_delta_pp"] = (
                    float(g["improvement_pp"].mean()) if len(g) else math.nan
                )
                row[f"{domain}_APD_gain_pct"] = (
                    float(g["relative_gain_pct"].mean()) if len(g) else math.nan
                )
            if np.isfinite(row["ID_value_mean"]) and np.isfinite(row["OOD_value_mean"]):
                row["absolute_ID_OOD_gap"] = abs(
                    row["ID_value_mean"] - row["OOD_value_mean"]
                )
            else:
                row["absolute_ID_OOD_gap"] = math.nan
            rows.append(row)
    return pd.DataFrame(rows)

def task_summary(condition: pd.DataFrame, metric: str) -> pd.DataFrame:
    pieces = [summarize_scope(condition, metric, task) for task in TASK_ORDER]
    return pd.concat(pieces, ignore_index=True)

def paired_p_against_delta(
    condition: pd.DataFrame, metric: str, domain: str,
) -> dict[str, tuple[float, int]]:
    """Reuse the four-comparator DELTA family, including transductive ComBat."""
    statistics = pairwise_statistics(condition, metric)
    family = statistics[(statistics.target == "DELTA") &
                        (statistics.domain == domain) & statistics.task_id.eq("ALL")]
    return {row.reference: (float(row.holm_p), int(row.condition_n))
            for row in family.itertuples()}

def safe_excel_value(value):
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value

def write_dataframe(ws, frame: pd.DataFrame, start_row: int, start_col: int = 1) -> tuple[int, int]:
    for j, column in enumerate(frame.columns, start=start_col):
        ws.cell(start_row, j, str(column))
    for i, row in enumerate(frame.itertuples(index=False, name=None), start=start_row + 1):
        for j, value in enumerate(row, start=start_col):
            ws.cell(i, j, safe_excel_value(value))
    return start_row + len(frame), start_col + len(frame.columns) - 1

def apply_table_style(
    ws,
    min_row: int,
    max_row: int,
    min_col: int,
    max_col: int,
    *,
    header_rows: int = 1,
    transductive_row: int | None = None,
):
    for row in ws.iter_rows(
        min_row=min_row,
        max_row=max_row,
        min_col=min_col,
        max_col=max_col,
    ):
        for cell in row:
            cell.fill = NO_FILL
            cell.font = Font(name=EXCEL_STYLE["font"], size=EXCEL_STYLE["body_size"])
            cell.alignment = Alignment(vertical="center", wrap_text=True)
            cell.border = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
    for r in range(min_row, min_row + header_rows):
        for cell in ws[r][min_col - 1 : max_col]:
            cell.font = Font(
                name=EXCEL_STYLE["font"],
                size=EXCEL_STYLE["header_size"],
                bold=True,
            )
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = Border(top=MEDIUM, bottom=MEDIUM, left=THIN, right=THIN)
    for cell in ws[max_row][min_col - 1 : max_col]:
        cell.border = Border(bottom=MEDIUM, left=THIN, right=THIN, top=THIN)
    if transductive_row is not None:
        for cell in ws[transductive_row][min_col - 1 : max_col]:
            cell.border = Border(top=MEDIUM, bottom=THIN, left=THIN, right=THIN)

def set_column_widths(ws, widths: dict[int, float], max_col: int):
    for col in range(1, max_col + 1):
        ws.column_dimensions[get_column_letter(col)].width = widths.get(col, 13)

def set_print_layout(ws, landscape: bool = True):
    ws.sheet_view.showGridLines = False
    ws.page_setup.orientation = "landscape" if landscape else "portrait"
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_margins.left = 0.25
    ws.page_margins.right = 0.25
    ws.page_margins.top = 0.45
    ws.page_margins.bottom = 0.45
    ws.oddFooter.center.text = "Page &P of &N"
    ws.oddFooter.center.size = 8

def fmt_ci(low: float, high: float, digits: int = 3, signed: bool = False) -> str | None:
    if not np.isfinite(low) or not np.isfinite(high):
        return None
    spec = f"{'+' if signed else ''}.{digits}f"
    return f"[{format(low, spec)}, {format(high, spec)}]"

def fmt_p(value: float) -> str | None:
    if not np.isfinite(value):
        return None
    if value < 0.0001:
        return "<0.0001"
    return f"{value:.4f}"

def rank_style_main_table(ws, method_rows: dict[str, int], summary: pd.DataFrame):
    # Rank only fully covered inductive rows. First = bold; second = underline.
    for domain, ba_col in (("ID", 4), ("OOD", 10)):
        candidates = summary[
            summary["domain"].eq(domain)
            & summary["rank_eligible"]
            & summary["coverage_pct"].eq(100.0)
            & np.isfinite(summary["value_mean"])
        ].sort_values("value_mean", ascending=False)
        for place, (_, row) in enumerate(candidates.head(2).iterrows(), start=1):
            excel_row = method_rows[str(row["method"])]
            current = ws.cell(excel_row, ba_col).font
            ws.cell(excel_row, ba_col).font = Font(
                name=EXCEL_STYLE["font"],
                size=EXCEL_STYLE["body_size"],
                bold=(place == 1 and EXCEL_STYLE["first_place_bold"]),
                underline=("single" if place == 2 and EXCEL_STYLE["second_place_underline"] else None),
            )

def make_readme_sheet(
    wb: Workbook,
    input_path: Path,
    input_sha: str,
    qc: dict,
    generated_at: str,
):
    ws = wb.active
    ws.title = "README"
    ws.sheet_view.showGridLines = False
    ws.merge_cells("A1:F1")
    ws["A1"] = "PathoROB Patch-set Aggregation - 20-backbone Manuscript Tables"
    ws["A1"].font = Font(name=EXCEL_STYLE["font"], size=EXCEL_STYLE["title_size"], bold=True)
    ws["A1"].alignment = Alignment(vertical="center")
    ws.row_dimensions[1].height = 25

    rows = [
        ("Generated", generated_at),
        ("Canonical source", str(input_path)),
        ("Source SHA-256", input_sha),
        ("Source rows", qc["source_rows"]),
        ("Backbones", qc["backbone_count"]),
        ("Tasks", qc["task_count"]),
        ("Splits per backbone-task", f"{qc['split_count_min']} to {qc['split_count_max']}"),
        ("Primary table metric", TARGET_METRIC),
        ("Primary unit", "backbone after split averaging and equal weighting of the three tasks; n=20"),
        ("Figure effect", "100 x direction-adjusted (method - matched RAW)"),
        ("APD in this workbook", "100 x direction-adjusted (method - matched RAW) / abs(matched RAW), paired on backbone/task/split/seed/domain"),
        ("DELTA projection", "The same centered Euclidean operator is used for every backbone"),
        ("Backbone exceptions", "None: the final centered Euclidean projection applies to all 20 backbones"),
        ("ComBat-transductive", "Both input branches included in prespecified Holm families; not deployment-equivalent; excluded only from inductive ranking"),
        ("ComBat-inductive OOD", "Not applicable under unseen-center policy; no value is imputed"),
        ("Ranking", "Only 100%-covered inductive methods; first bold, second underline"),
        ("Missingness", "NA means scientifically not applicable or unavailable; never converted to zero"),
    ]
    ws["A3"] = "Item"
    ws["B3"] = "Definition"
    for i, (key, value) in enumerate(rows, start=4):
        ws.cell(i, 1, key)
        ws.cell(i, 2, value)
    apply_table_style(ws, 3, 3 + len(rows), 1, 2)
    ws.column_dimensions["A"].width = 31
    ws.column_dimensions["B"].width = 115

    start = 23
    ws.cell(start, 1, "Publication method")
    ws.cell(start, 2, "Protocol classification")
    for i, method in enumerate(METHOD_ORDER, start=start + 1):
        ws.cell(i, 1, method)
        ws.cell(i, 2, method_protocol(method))
    apply_table_style(
        ws,
        start,
        start + len(METHOD_ORDER),
        1,
        2,
        transductive_row=start + 1 + METHOD_ORDER.index("ComBat-transductive"),
    )
    ws.freeze_panes = "A4"
    set_print_layout(ws, landscape=False)

def make_main_table_sheet(wb: Workbook, summary: pd.DataFrame):
    ws = wb.create_sheet("Table 1 - BA")
    ws.merge_cells("A1:O1")
    ws["A1"] = "Table 1. Patch-set aggregation balanced accuracy across 20 pathology foundation models"
    ws["A1"].font = Font(name=EXCEL_STYLE["font"], size=EXCEL_STYLE["title_size"], bold=True)
    ws.merge_cells("A2:O2")
    ws["A2"] = (
        "Values are equal-weighted backbone means after split averaging and equal weighting of the three tasks. "
        "Delta is percentage-point improvement over matched RAW. CI uses 10,000 backbone-bootstrap replicates. "
        "Transductive ComBat is separated "
        "and excluded from inductive ranking."
    )
    ws["A2"].font = Font(name=EXCEL_STYLE["font"], size=EXCEL_STYLE["caption_size"], italic=True)
    ws["A2"].alignment = Alignment(wrap_text=True)
    ws.row_dimensions[2].height = 34

    ws.merge_cells("A4:A5")
    ws.merge_cells("B4:B5")
    ws.merge_cells("C4:C5")
    ws.merge_cells("D4:I4")
    ws.merge_cells("J4:N4")
    ws["A4"], ws["B4"], ws["C4"] = "Method", "Protocol", "Rank eligible"
    ws["D4"], ws["J4"] = "ID", "OOD"
    id_headers = ["BA", "95% CI", "Delta BA (pp)", "Delta 95% CI", "Wins", "Coverage"]
    ood_headers = ["BA", "95% CI", "Delta BA (pp)", "Delta 95% CI", "Wins", "Coverage"]
    for c, value in enumerate(id_headers, start=4):
        ws.cell(5, c, value)
    for c, value in enumerate(ood_headers, start=10):
        ws.cell(5, c, value)

    method_rows = {}
    for r, method in enumerate(METHOD_ORDER, start=6):
        method_rows[method] = r
        ws.cell(r, 1, method)
        ws.cell(r, 2, method_protocol(method))
        ws.cell(r, 3, "Yes" if method in RANK_ELIGIBLE_METHODS else "No")
        for domain, start_col in (("ID", 4), ("OOD", 10)):
            s = summary[
                summary["method"].eq(method) & summary["domain"].eq(domain)
            ].iloc[0]
            ws.cell(r, start_col, safe_excel_value(s["value_mean"]))
            ws.cell(r, start_col + 1, fmt_ci(s["value_ci_low"], s["value_ci_high"]))
            ws.cell(r, start_col + 2, safe_excel_value(s["delta_pp_mean"]))
            ws.cell(
                r,
                start_col + 3,
                fmt_ci(s["delta_pp_ci_low"], s["delta_pp_ci_high"], digits=2, signed=True),
            )
            ws.cell(r, start_col + 4, f"{int(s['wins'])}/{int(s['condition_n'])}" if s["condition_n"] else None)
            ws.cell(r, start_col + 5, f"{int(s['condition_n'])}/{int(s['expected_condition_n'])}")

    last_row = 5 + len(METHOD_ORDER)
    apply_table_style(
        ws,
        4,
        last_row,
        1,
        15,
        header_rows=2,
        transductive_row=method_rows["ComBat-transductive"],
    )
    for r in range(6, last_row + 1):
        for c in (4, 10):
            ws.cell(r, c).number_format = "0.000"
        for c in (6, 12):
            ws.cell(r, c).number_format = "+0.00;-0.00;0.00"
    rank_style_main_table(ws, method_rows, summary)
    set_column_widths(
        ws,
        {
            1: 31,
            2: 52,
            3: 13,
            4: 11,
            5: 19,
            6: 14,
            7: 22,
            8: 11,
            9: 12,
            10: 11,
            11: 19,
            12: 14,
            13: 22,
            14: 11,
            15: 12,
        },
        15,
    )
    ws.freeze_panes = "D6"
    ws.auto_filter.ref = f"A5:O{last_row}"
    set_print_layout(ws)

def make_apd_table_sheet(wb: Workbook, summary: pd.DataFrame):
    ws = wb.create_sheet("Table 2 - APD")
    ws.merge_cells("A1:J1")
    ws["A1"] = "Table 2. Matched relative balanced-accuracy gain (APD convention)"
    ws["A1"].font = Font(name=EXCEL_STYLE["font"], size=EXCEL_STYLE["title_size"], bold=True)
    ws.merge_cells("A2:J2")
    ws["A2"] = (
        "APD gain (%) = 100 x (method BA - matched RAW BA) / abs(matched RAW BA), paired on the same "
        "backbone, task, split, seed, and domain. Positive values favor the method."
    )
    ws["A2"].font = Font(name=EXCEL_STYLE["font"], size=EXCEL_STYLE["caption_size"], italic=True)
    ws["A2"].alignment = Alignment(wrap_text=True)
    headers = [
        "Method",
        "Protocol",
        "ID APD gain (%)",
        "ID 95% CI",
        "ID wins",
        "ID coverage",
        "OOD APD gain (%)",
        "OOD 95% CI",
        "OOD wins",
        "OOD coverage",
    ]
    for c, value in enumerate(headers, start=1):
        ws.cell(4, c, value)
    method_rows = {}
    for r, method in enumerate(METHOD_ORDER, start=5):
        method_rows[method] = r
        ws.cell(r, 1, method)
        ws.cell(r, 2, method_protocol(method))
        for domain, start_col in (("ID", 3), ("OOD", 7)):
            s = summary[
                summary["method"].eq(method) & summary["domain"].eq(domain)
            ].iloc[0]
            ws.cell(r, start_col, safe_excel_value(s["relative_gain_mean_pct"]))
            ws.cell(
                r,
                start_col + 1,
                fmt_ci(
                    s["relative_gain_ci_low_pct"],
                    s["relative_gain_ci_high_pct"],
                    digits=2,
                    signed=True,
                ),
            )
            ws.cell(r, start_col + 2, f"{int(s['wins'])}/{int(s['condition_n'])}" if s["condition_n"] else None)
            ws.cell(r, start_col + 3, f"{int(s['condition_n'])}/{int(s['expected_condition_n'])}")
    last_row = 4 + len(METHOD_ORDER)
    apply_table_style(
        ws,
        4,
        last_row,
        1,
        10,
        transductive_row=method_rows["ComBat-transductive"],
    )
    for r in range(5, last_row + 1):
        ws.cell(r, 3).number_format = "+0.00;-0.00;0.00"
        ws.cell(r, 7).number_format = "+0.00;-0.00;0.00"
    set_column_widths(ws, {1: 31, 2: 52, 3: 17, 4: 21, 5: 11, 6: 12, 7: 18, 8: 21, 9: 11, 10: 13}, 10)
    ws.freeze_panes = "C5"
    ws.auto_filter.ref = f"A4:J{last_row}"
    set_print_layout(ws)

def add_dataframe_sheet(
    wb: Workbook,
    name: str,
    title: str,
    caption: str,
    frame: pd.DataFrame,
    widths: dict[str, float] | None = None,
):
    ws = wb.create_sheet(name)
    max_col = max(1, len(frame.columns))
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max_col)
    ws.cell(1, 1, title)
    ws.cell(1, 1).font = Font(name=EXCEL_STYLE["font"], size=EXCEL_STYLE["title_size"], bold=True)
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=max_col)
    ws.cell(2, 1, caption)
    ws.cell(2, 1).font = Font(name=EXCEL_STYLE["font"], size=EXCEL_STYLE["caption_size"], italic=True)
    ws.cell(2, 1).alignment = Alignment(wrap_text=True)
    ws.row_dimensions[2].height = 32
    last_row, last_col = write_dataframe(ws, frame, 4)
    trans_row = None
    if "method" in frame.columns and "ComBat-transductive" in set(frame["method"].astype(str)):
        first_idx = frame.index[frame["method"].astype(str).eq("ComBat-transductive")][0]
        trans_row = 5 + int(first_idx)
    apply_table_style(ws, 4, last_row, 1, last_col, transductive_row=trans_row)
    ws.freeze_panes = "A5"
    ws.auto_filter.ref = f"A4:{get_column_letter(last_col)}{last_row}"
    for c, column in enumerate(frame.columns, start=1):
        default = 14
        if widths and column in widths:
            default = widths[column]
        elif any(token in str(column).lower() for token in ("protocol", "comparison")):
            default = 46
        elif any(token in str(column).lower() for token in ("backbone", "method", "task")):
            default = 27
        ws.column_dimensions[get_column_letter(c)].width = default
        if any(token in str(column).lower() for token in ("mean", "median", "std", "delta", "gain", "ci_low", "ci_high")):
            for r in range(5, last_row + 1):
                ws.cell(r, c).number_format = "+0.000;-0.000;0.000" if ("delta" in str(column).lower() or "gain" in str(column).lower()) else "0.000"
        if "p" == str(column).lower()[-1:] or "p_value" in str(column).lower() or str(column).lower().endswith("_p"):
            for r in range(5, last_row + 1):
                ws.cell(r, c).number_format = "0.0000"
    set_print_layout(ws)
    return ws

def make_qc_sheet(
    wb: Workbook,
    source: pd.DataFrame,
    qc: dict,
    input_path: Path,
    input_sha: str,
):
    ws = wb.create_sheet("QC and Provenance")
    ws.merge_cells("A1:F1")
    ws["A1"] = "QC and Provenance"
    ws["A1"].font = Font(name=EXCEL_STYLE["font"], size=EXCEL_STYLE["title_size"], bold=True)
    qc_rows = pd.DataFrame(
        [(k, v) for k, v in qc.items()]
        + [
            ("input_path", str(input_path)),
            ("input_sha256", input_sha),
            ("expected_backbones", len(BACKBONE_ORDER)),
            ("expected_tasks", len(TASK_ORDER)),
            ("expected_splits_per_backbone_task", 100),
            ("raw_source_read_only", True),
        ],
        columns=["check", "value"],
    )
    last, _ = write_dataframe(ws, qc_rows, 3)
    apply_table_style(ws, 3, last, 1, 2)

    status = (
        source.groupby(["method", "domain", "status"], dropna=False)
        .size()
        .rename("row_count")
        .reset_index()
    )
    start = last + 3
    last2, last_col = write_dataframe(ws, status, start)
    apply_table_style(ws, start, last2, 1, last_col)
    ws.column_dimensions["A"].width = 62
    ws.column_dimensions["B"].width = 23
    ws.column_dimensions["C"].width = 20
    ws.column_dimensions["D"].width = 16
    ws.freeze_panes = "A4"
    set_print_layout(ws)

def build_workbook(
    output_path: Path,
    source: pd.DataFrame,
    complete: pd.DataFrame,
    condition: pd.DataFrame,
    overall_summary: pd.DataFrame,
    backbone_table: pd.DataFrame,
    task_table: pd.DataFrame,
    pairwise: pd.DataFrame,
    input_path: Path,
    input_sha: str,
    qc: dict,
    generated_at: str,
):
    wb = Workbook()
    make_readme_sheet(wb, input_path, input_sha, qc, generated_at)
    make_main_table_sheet(wb, overall_summary)
    make_apd_table_sheet(wb, overall_summary)

    add_dataframe_sheet(
        wb,
        "Backbone Summary",
        "Backbone-level summary",
        "Each value is the equal-task mean across the three frozen patch-set aggregation tasks. "
        "ComBat estimates adjustment parameters from each evaluation cohort's embedding statistics and medical-center labels.",
        backbone_table,
    )
    add_dataframe_sheet(
        wb,
        "Task Summary",
        "Task-level summary",
        "Backbones are equally weighted within task. CI uses a condition bootstrap.",
        task_table,
    )

    pairwise_export = pairwise.copy()
    pairwise_export["wilcoxon_p_display"] = pairwise_export["wilcoxon_p"].map(fmt_p)
    pairwise_export["holm_p_display"] = pairwise_export["holm_p"].map(fmt_p)
    add_dataframe_sheet(
        wb,
        "Pairwise Statistics",
        "Paired backbone-level DELTA comparisons in both input branches",
        "The inferential unit is the backbone. Split results are averaged within task and the three tasks are "
        "weighted equally within backbone before testing. Exact signed-rank tests round native differences to 12 decimals, "
        "use average tied ranks and exclude zeros. Holm covers four comparators per branch/metric/domain for pooled results, "
        "or twelve contrasts across comparators and tasks; both canonical transductive ComBat branches are included.",
        pairwise_export,
    )

    condition_export = condition[condition["metric"].eq(TARGET_METRIC)].sort_values(
        ["backbone", "task_id", "domain", "method", "metric"], kind="stable"
    ).reset_index(drop=True)
    add_dataframe_sheet(
        wb,
        "Condition Means",
        "Backbone x task condition means",
        "Auditable analysis table used by the workbook and overall statistics.",
        condition_export,
    )

    all_metrics = condition.sort_values(
        ["backbone", "task_id", "domain", "method", "metric"], kind="stable"
    ).reset_index(drop=True)
    add_dataframe_sheet(
        wb,
        "All Metrics",
        "All six downstream metrics",
        "Positive improvement_pp and relative_gain_pct always favor the method; Brier and ECE directions are reversed before computing improvement.",
        all_metrics,
    )
    make_qc_sheet(wb, source, qc, input_path, input_sha)

    # Workbook-wide font cleanup and formal white background.
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                if cell.value is not None and cell.font.name != EXCEL_STYLE["font"]:
                    cell.font = Font(
                        name=EXCEL_STYLE["font"],
                        size=cell.font.sz or EXCEL_STYLE["body_size"],
                        bold=cell.font.bold,
                        italic=cell.font.italic,
                        underline=cell.font.underline,
                    )

    legacy_partial = output_path.with_name(output_path.name + ".partial")
    if legacy_partial.exists():
        legacy_partial.unlink()
    partial = output_path.with_name(
        output_path.stem + ".partial" + output_path.suffix
    )
    wb.save(partial)
    wb.close()
    # Reopen before atomic publication.
    check = load_workbook(partial, data_only=False, read_only=True)
    expected_sheets = {
        "README",
        "Table 1 - BA",
        "Table 2 - APD",
        "Backbone Summary",
        "Task Summary",
        "Pairwise Statistics",
        "Condition Means",
        "All Metrics",
        "QC and Provenance",
    }
    if set(check.sheetnames) != expected_sheets:
        check.close()
        raise ValueError(f"Workbook sheet contract failed: {check.sheetnames}")
    check.close()
    os.replace(partial, output_path)

PROJECT_ROOT = Path(os.environ.get("DELTA_PROJECT_ROOT", repository_root())).expanduser().resolve()
INPUT_CSV = Path(os.environ.get(
    "PATHROB_PATCH_AGG_INPUT_CSV",
    PROJECT_ROOT / "runs/pathorob_patch_aggregation/main/results/all_methods_v1_3_p2_results_long.csv",
)).expanduser()
OUTPUT_STEM = "PathoROB_patch_aggregation_20backbone_all_metrics"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if TARGET_METRIC != "balanced_accuracy":
        raise ValueError("The primary workbook uses balanced_accuracy; all six metrics are exported in CSV.")
    output = args.output_dir
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Use a new or empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source, qc = load_source(args.input_csv)
    complete = source[source.status.eq("complete") & np.isfinite(source.value)].copy()
    paired = build_paired_frame(complete)
    condition = build_condition_means(paired)
    all_pairwise = pd.concat([pairwise_statistics(condition, metric) for metric in SUPPORTED_METRICS], ignore_index=True)
    tables = {
        "condition_means": condition,
        "paired_statistics": all_pairwise,
        "backbone_summary": pd.concat([backbone_summary(condition, metric) for metric in SUPPORTED_METRICS], ignore_index=True),
        "task_summary": pd.concat([task_summary(condition, metric) for metric in SUPPORTED_METRICS], ignore_index=True),
    }
    for name, frame in tables.items():
        frame.to_csv(output / (name + ".csv"), index=False)
    overall = summarize_scope(condition, TARGET_METRIC)
    backbone = backbone_summary(condition, TARGET_METRIC)
    tasks = task_summary(condition, TARGET_METRIC)
    pairwise = pairwise_statistics(condition, TARGET_METRIC)
    workbook = output / f"{OUTPUT_STEM}_tables.xlsx"
    build_workbook(workbook, source, complete, condition, overall, backbone, tasks, pairwise,
                   args.input_csv, sha256_file(args.input_csv), qc,
                   datetime.now().astimezone().isoformat(timespec="seconds"))
    print(f"AGGREGATION_REPORT_COMPLETE {workbook}")


if __name__ == "__main__":
    main()
