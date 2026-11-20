"""Export canonical WSI survival bank-coverage tables without plotting.

The five-backbone, two-branch, eight-fraction results remain read-only inputs.
Mean time-dependent AUC is the arithmetic mean at the locked q25/q50/q75
horizons. A fail-closed check excludes noncanonical rank rules and fixed caps.
"""
from __future__ import annotations
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BACKBONES = ["conch", "musk", "h0mini", "uni", "virchow2"]
BACKBONE_LABELS = {
    "conch": "CONCH",
    "musk": "MUSK",
    "h0mini": "H-optimus-0-mini",
    "uni": "UNI",
    "virchow2": "Virchow2",
}
BANK_FRACTIONS = [1, 2, 5, 10, 20, 30, 40, 50]
CANONICAL_RANK_RULE = "null_adjusted_90pct"
CREATE_XLSX_WORKBOOK = True
ENDPOINTS = {
    "uno_c_index": {
        "label": "Uno C-index",
        "title": "Uno C-index",
        "ylabel": "Uno C-index",
        "role": "Primary",
        "higher": True,
    },
    "integrated_brier_score": {
        "label": "Integrated Brier score",
        "title": "Integrated Brier score",
        "ylabel": "IBS",
        "role": "Primary",
        "higher": False,
    },
    "harrell_c_index": {
        "label": "Harrell C-index",
        "title": "Harrell C-index",
        "ylabel": "Harrell C-index",
        "role": "Secondary",
        "higher": True,
    },
    "mean_time_dependent_auc": {
        "label": "Mean time-dependent AUC",
        "title": "Mean AUC",
        "ylabel": "Mean AUC",
        "role": "Secondary",
        "higher": True,
    },
}
DOMAIN_LABELS = {
    "held_out_center": "OOD (held-out center)",
    "in_center": "ID (in-center)",
}
BRANCH_LABELS = {
    "raw": "DELTA-v1.3",
    "reinhard": "Reinhard-DELTA-v1.3",
}
MATCHED_BASELINE = {"raw": "RAW", "reinhard": "Reinhard-RAW"}
RAW_METRICS = [
    "uno_c_index",
    "integrated_brier_score",
    "harrell_c_index",
    "td_auc_q25",
    "td_auc_q50",
    "td_auc_q75",
]
TD_AUC_COMPONENTS = ["td_auc_q25", "td_auc_q50", "td_auc_q75"]
REQUIRED_A_COLUMNS = {
    "backbone",
    "branch",
    "bank_fraction_percent",
    "split_id",
    "method",
    "domain",
    "metric",
    "value",
    "status",
}
REQUIRED_REFERENCE_COLUMNS = {
    "backbone",
    "split_id",
    "method",
    "domain",
    "metric",
    "value",
    "status",
}
REQUIRED_STRUCTURE_COLUMNS = {
    "backbone",
    "branch",
    "bank_fraction_percent",
    "applied_rank",
    "intrinsic_rank",
    "rank_rule",
    "cap_free",
}

BRANCHES = ["raw", "reinhard"]
DOMAINS = ["held_out_center", "in_center"]
EXPERIMENT_ROOT = Path()
CANONICAL_ROOT = Path()
OUTPUT_DIR = Path()


def require_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def validate_finite_complete(frame: pd.DataFrame, source: Path) -> None:
    if frame.empty:
        raise ValueError(f"{source} is empty")
    values = pd.to_numeric(frame["value"], errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{source} contains non-finite values")
    if not frame["status"].astype(str).str.lower().eq("complete").all():
        raise ValueError(f"{source} contains non-complete rows")


def add_mean_time_dependent_auc(frame: pd.DataFrame) -> pd.DataFrame:
    """Append the arithmetic mean of the three locked AUC horizons."""
    existing = frame.loc[frame["metric"].eq("mean_time_dependent_auc")]
    if not existing.empty:
        return frame.copy()
    component = frame.loc[frame["metric"].isin(TD_AUC_COMPONENTS)].copy()
    if component.empty:
        raise ValueError("time-dependent AUC components are absent")
    group_columns = [column for column in frame.columns if column not in {"metric", "value"}]
    counts = component.groupby(group_columns, dropna=False)["metric"].nunique()
    if not counts.eq(len(TD_AUC_COMPONENTS)).all():
        raise ValueError("a row group does not contain all three locked AUC horizons")
    derived = component.groupby(group_columns, dropna=False, as_index=False)["value"].mean()
    derived["metric"] = "mean_time_dependent_auc"
    return pd.concat([frame, derived[frame.columns]], ignore_index=True)


def validate_canonical_rank_contract(backbone: str) -> None:
    """Reject any noncanonical rank rule before tables are built."""

    for branch in ("raw", "reinhard"):
        source = EXPERIMENT_ROOT / "artifacts" / backbone / branch / "bank_structure.csv"
        if not source.is_file():
            raise FileNotFoundError(f"canonical rank structure is missing: {source}")
        structure = pd.read_csv(source)
        require_columns(structure, REQUIRED_STRUCTURE_COLUMNS, source)
        if set(structure["bank_fraction_percent"].astype(int)) != set(BANK_FRACTIONS):
            raise ValueError(f"{source} does not contain the eight frozen bank fractions")
        if set(structure["branch"].astype(str)) != {branch}:
            raise ValueError(f"{source} contains a nonmatching branch")
        if not structure["rank_rule"].astype(str).eq(CANONICAL_RANK_RULE).all():
            rules = sorted(structure["rank_rule"].astype(str).unique().tolist())
            raise ValueError(
                f"noncanonical rank rule in {source}: {rules}; "
                f"required={CANONICAL_RANK_RULE}"
            )
        cap_free = structure["cap_free"].astype(str).str.lower().isin({"true", "1"})
        if not cap_free.all():
            raise ValueError(f"fixed rank cap detected in canonical structure: {source}")
        if (structure["applied_rank"].astype(int) <= 0).any():
            raise ValueError(f"nonpositive canonical rank in {source}")


def read_candidate_results() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for backbone in BACKBONES:
        validate_canonical_rank_contract(backbone)
        source = EXPERIMENT_ROOT / "artifacts" / backbone / "ratio_results.csv"
        if not source.is_file():
            raise FileNotFoundError(f"official null-adjusted-90% result is missing: {source}")
        frame = pd.read_csv(source)
        require_columns(frame, REQUIRED_A_COLUMNS, source)
        validate_finite_complete(frame, source)
        if set(frame["bank_fraction_percent"].astype(int)) != set(BANK_FRACTIONS):
            raise ValueError(f"{source} does not contain the eight frozen bank fractions")
        if set(frame["branch"].astype(str)) != {"raw", "reinhard"}:
            raise ValueError(f"{source} does not contain both official branches")
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    return add_mean_time_dependent_auc(combined)


def read_reference_results() -> pd.DataFrame:
    controls: list[pd.DataFrame] = []
    for backbone in BACKBONES:
        directory = CANONICAL_ROOT / "results" / backbone
        control_source = directory / "survival_metrics_long.csv"
        control = pd.read_csv(control_source)
        require_columns(control, REQUIRED_REFERENCE_COLUMNS, control_source)
        # Some unrelated comparison methods can contain scientific NA values.
        # Validate only the exact matched controls used by this report; never
        # drop rows from the DELTA-v1.3 candidate results.
        control = control.loc[control["method"].isin(MATCHED_BASELINE.values())].copy()
        validate_finite_complete(control, control_source)
        controls.append(control)
    return add_mean_time_dependent_auc(pd.concat(controls, ignore_index=True))


def endpoint_value(
    frame: pd.DataFrame,
    *,
    backbone: str,
    domain: str,
    metric: str,
    method: str | None = None,
) -> float:
    selected = frame.loc[
        frame["backbone"].eq(backbone)
        & frame["domain"].eq(domain)
        & frame["metric"].eq(metric)
    ]
    if method is not None:
        selected = selected.loc[selected["method"].eq(method)]
    if len(selected) != 1:
        raise ValueError(
            f"expected one value: backbone={backbone}, domain={domain}, "
            f"metric={metric}, method={method}; found={len(selected)}"
        )
    return float(selected.iloc[0]["value"])


def candidate_curve(
    candidates: pd.DataFrame,
    *,
    backbone: str,
    branch: str,
    domain: str,
    metric: str,
) -> pd.Series:
    selected = candidates.loc[
        candidates["backbone"].eq(backbone)
        & candidates["branch"].eq(branch)
        & candidates["domain"].eq(domain)
        & candidates["metric"].eq(metric)
    ].copy()
    selected["bank_fraction_percent"] = selected["bank_fraction_percent"].astype(int)
    curve = selected.set_index("bank_fraction_percent")["value"].reindex(BANK_FRACTIONS)
    if curve.isna().any() or len(curve) != len(BANK_FRACTIONS):
        raise ValueError(f"incomplete curve: {backbone}/{branch}/{domain}/{metric}")
    return curve.astype(float)


def source_table(
    candidates: pd.DataFrame,
    controls: pd.DataFrame,
) -> pd.DataFrame:
    candidate = candidates.loc[candidates["metric"].isin(ENDPOINTS)].copy()
    candidate["record_type"] = "bank_coverage"
    baseline_rows: list[dict[str, object]] = []
    for branch in BRANCHES:
        for backbone in BACKBONES:
            for domain in DOMAINS:
                for metric in ENDPOINTS:
                    baseline_rows.append(
                        {
                            "backbone": backbone,
                            "branch": branch,
                            "bank_fraction_percent": np.nan,
                            "split_id": candidate["split_id"].iloc[0],
                            "method": MATCHED_BASELINE[branch],
                            "domain": domain,
                            "metric": metric,
                            "value": endpoint_value(
                                controls,
                                backbone=backbone,
                                domain=domain,
                                metric=metric,
                                method=MATCHED_BASELINE[branch],
                            ),
                            "status": "complete",
                            "record_type": "matched_baseline",
                        }
                    )
    return pd.concat(
        [candidate, pd.DataFrame(baseline_rows)],
        ignore_index=True,
    )


def coverage_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    selected = candidates.loc[candidates["metric"].isin(ENDPOINTS)].copy()
    summary = (
        selected.groupby(
            ["branch", "domain", "metric", "bank_fraction_percent"],
            as_index=False,
        )["value"]
        .agg(["mean", "median", "min", "max", "count"])
    )
    return summary


def write_table_sheet(writer, name: str, frame: pd.DataFrame) -> None:
    from openpyxl.styles import Alignment, Border, Font, Side
    from openpyxl.utils import get_column_letter

    frame.to_excel(writer, sheet_name=name, index=False)
    sheet = writer.sheets[name]
    header_font = Font(name="Arial", size=9, bold=True)
    body_font = Font(name="Arial", size=9)
    header_border = Border(bottom=Side(style="thin", color="000000"))
    for cell in sheet[1]:
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center")
        cell.border = header_border
    for column_index, column in enumerate(frame.columns, start=1):
        is_numeric = pd.api.types.is_numeric_dtype(frame[column]) and not pd.api.types.is_bool_dtype(
            frame[column]
        )
        values = frame[column].astype(str).head(2000)
        width = min(36, max(10, len(str(column)) + 1, *(len(str(value)) + 1 for value in values)))
        sheet.column_dimensions[get_column_letter(column_index)].width = width
        for cell in sheet.iter_cols(
            min_col=column_index,
            max_col=column_index,
            min_row=2,
            max_row=max(2, len(frame) + 1),
        ):
            for item in cell:
                item.font = body_font
                if is_numeric:
                    item.number_format = "0.0000"
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    sheet.sheet_view.showGridLines = False
    if name == "README":
        # Keep the compact two-column layout, but make every protocol note
        # visible in Excel and rendered previews instead of clipping at the
        # former generic 36-character width cap.
        sheet.column_dimensions["A"].width = 24
        sheet.column_dimensions["B"].width = 72
        for row_index in range(2, len(frame) + 2):
            sheet.cell(row=row_index, column=1).alignment = Alignment(
                horizontal="left",
                vertical="top",
            )
            description_cell = sheet.cell(row=row_index, column=2)
            description_cell.alignment = Alignment(
                horizontal="left",
                vertical="top",
                wrap_text=True,
            )
            text_length = len(str(description_cell.value or ""))
            wrapped_lines = max(1, int(np.ceil(text_length / 68)))
            sheet.row_dimensions[row_index].height = 15 * wrapped_lines + 3


def build_workbook(
    candidates: pd.DataFrame,
    controls: pd.DataFrame,
    source: pd.DataFrame,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    readme = pd.DataFrame(
        {
            "Item": [
                "Analysis",
                "Backbones",
                "Bank fractions",
                "Primary endpoints",
                "Secondary endpoints",
                "Main baseline rule",
                "Domains",
                "Outcome use",
                "Canonical rank rule",
            ],
            "Description": [
                "DELTA-v1.3 Full-WSI bank-coverage ablation (official null-adjusted-90% results)",
                ", ".join(BACKBONE_LABELS[b] for b in BACKBONES),
                ", ".join(map(str, BANK_FRACTIONS)) + "% per disjoint Style/Fitting role",
                "Uno C-index; integrated Brier score",
                "Harrell C-index; mean of locked q25/q50/q75 time-dependent AUCs",
                "DELTA uses RAW; Reinhard-DELTA uses Reinhard-RAW",
                "ID=in-center; OOD=held-out center",
                "Outcomes are used only by the locked downstream evaluation",
                "Leading permutation-null-supported interval; 90% cumulative null-adjusted spectral mass; no fixed rank cap",
            ],
        }
    )
    qc = pd.DataFrame(
        {
            "Check": [
                "five_backbones",
                "eight_fractions",
                "two_branches",
                "two_domains",
                "four_endpoints",
                "all_finite",
                "all_complete",
            ],
            "Result": [
                candidates["backbone"].nunique() == 5,
                set(candidates["bank_fraction_percent"].astype(int)) == set(BANK_FRACTIONS),
                set(candidates["branch"]) == {"raw", "reinhard"},
                set(candidates["domain"]) == {"in_center", "held_out_center"},
                set(ENDPOINTS).issubset(set(candidates["metric"])),
                np.isfinite(source["value"].to_numpy(dtype=float)).all(),
                source["status"].astype(str).str.lower().eq("complete").all(),
            ],
        }
    )
    qc["Result"] = np.where(qc["Result"], "PASS", "FAIL")
    partial = output.with_name(f".{output.name}.partial.xlsx")
    try:
        with pd.ExcelWriter(partial, engine="openpyxl") as writer:
            write_table_sheet(writer, "README", readme)
            write_table_sheet(writer, "Coverage_Summary", coverage_summary(candidates))
            write_table_sheet(writer, "Figure_Source", source)
            write_table_sheet(writer, "Candidate_Long", candidates)
            write_table_sheet(
                writer,
                "Matched_Controls",
                controls.loc[controls["metric"].isin(ENDPOINTS)],
            )
            write_table_sheet(writer, "QC", qc)
        os.replace(partial, output)
    finally:
        partial.unlink(missing_ok=True)


def main() -> None:
    global EXPERIMENT_ROOT, CANONICAL_ROOT, OUTPUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    EXPERIMENT_ROOT = args.experiment_root.expanduser().resolve()
    CANONICAL_ROOT = args.canonical_root.expanduser().resolve()
    OUTPUT_DIR = args.output_dir.expanduser().resolve()
    candidates = read_candidate_results()
    controls = read_reference_results()
    source = source_table(candidates, controls)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(OUTPUT_DIR / "metrics_long.csv", index=False)
    source.to_csv(OUTPUT_DIR / "source_long.csv", index=False)
    coverage_summary(candidates).to_csv(OUTPUT_DIR / "coverage_summary.csv", index=False)
    workbook = OUTPUT_DIR / "DELTA_v1_3_bank_coverage.xlsx"
    if CREATE_XLSX_WORKBOOK:
        build_workbook(candidates, controls, source, workbook)
    manifest = {
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backbones": BACKBONES,
        "bank_fractions_percent_per_role": BANK_FRACTIONS,
        "endpoints": list(ENDPOINTS),
        "domains": DOMAINS,
        "branches": BRANCHES,
        "matched_baseline": MATCHED_BASELINE,
        "canonical_rank_rule": CANONICAL_RANK_RULE,
        "diagnostic_full_null_included": False,
        "experiment_root": str(EXPERIMENT_ROOT),
        "canonical_root": str(CANONICAL_ROOT),
        "workbook": str(workbook),
        "workbook_generated_in_this_python_run": CREATE_XLSX_WORKBOOK,
        "metrics": str(OUTPUT_DIR / "metrics_long.csv"),
        "source": str(OUTPUT_DIR / "source_long.csv"),
    }
    partial = OUTPUT_DIR / "coverage_report_manifest.json.partial"
    partial.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, OUTPUT_DIR / "coverage_report_manifest.json")
    print(f"DELTA_BANK_COVERAGE_REPORT_COMPLETE output={OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
