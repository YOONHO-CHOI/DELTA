"""Export the frozen TCGA-BRCA manuscript comparisons without rendering figures.

Paired effects retain their original confidence intervals and exact sign-flip P
values. Holm correction covers four prespecified comparators separately within
each endpoint, domain and DELTA input branch. IBS differences are exported in
the direct target-minus-control direction, for which lower is favourable.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import openpyxl

TCGA_METHODS = ('RAW', 'Reinhard-RAW', 'DELTA-v1.3', 'Reinhard-DELTA-v1.3')
SURVIVAL_METHODS = ('RAW', 'DANN', 'ComBat-transductive', 'DELTA-v1.3', 'Reinhard-RAW', 'Reinhard-DANN', 'Reinhard-ComBat-transductive', 'Reinhard-DELTA-v1.3')
METRICS_TCGA = (('uno_c_index', 'Uno C-index'), ('harrell_c_index', 'Harrell C-index'), ('td_auc_mean', 'Mean tdAUC'), ('integrated_brier_score', 'IBS ↓'))
DOMAINS = (('in_center', 'ID'), ('held_out_center', 'OOD'))
TCGA_COMPARATORS = (('DELTA-v1.3', ('RAW', 'Reinhard-RAW', 'DANN', 'ComBat-transductive')), ('Reinhard-DELTA-v1.3', ('RAW', 'Reinhard-RAW', 'Reinhard-DANN', 'Reinhard-ComBat-transductive')))
TCGA_EFFECTS = (('DELTA-v1.3', 'RAW'), ('Reinhard-DELTA-v1.3', 'Reinhard-RAW'))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_sheet(book: openpyxl.Workbook, name: str, header: int = 0) -> pd.DataFrame:
    records = list(book[name].values)
    return pd.DataFrame(records[header + 1:], columns=records[header])


def tcga_holm_pairwise(pairwise: pd.DataFrame) -> pd.DataFrame:
    """Adjust frozen P values without changing effects, intervals or source rows."""
    result = pairwise.copy()
    keys = ["target", "control", "domain", "metric"]
    if result.duplicated(keys).any():
        raise ValueError("Duplicate TCGA paired contrast before Holm correction")
    result["p_holm_within_family"] = np.nan
    result["holm_family_size"] = 0
    result["holm_family"] = ""
    for target, controls in TCGA_COMPARATORS:
        expected_controls = set(controls)
        for domain, _ in DOMAINS:
            for metric, _ in METRICS_TCGA:
                mask = (result["target"].eq(target) & result["domain"].eq(domain) &
                        result["metric"].eq(metric) &
                        result["control"].isin(expected_controls))
                family = result.loc[mask]
                if len(family) != 4 or set(family["control"]) != expected_controls:
                    raise ValueError(f"Incomplete TCGA Holm family: {target}/{domain}/{metric}")
                p = pd.to_numeric(family["p_raw"], errors="raise").to_numpy(float)
                if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                    raise ValueError(f"Invalid TCGA P values: {target}/{domain}/{metric}")
                positions = np.argsort(p, kind="stable")
                adjusted = np.empty_like(p)
                adjusted[positions] = np.minimum(
                    1.0, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[positions]))
                result.loc[mask, "p_holm_within_family"] = adjusted
                result.loc[mask, "holm_family_size"] = len(p)
                result.loc[mask, "holm_family"] = f"{target}/{domain}/{metric}"
    return result


def load_tcga_inputs(workbook: Path):
    import openpyxl

    book = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    required = {"SUPP_06_Backbones", "SUPP_08_Analysis_Long", "SUPP_04_Paired_Summary",
                "SUPP_02_DELTA_Pairwise", "SUPP_09_ReDELTA_Pairwise"}
    if not required.issubset(book.sheetnames):
        raise ValueError(f"Unified TCGA workbook missing {required - set(book.sheetnames)}")
    backbone_frame = read_sheet(book, "SUPP_06_Backbones")
    order = backbone_frame["backbone"].astype(str).tolist()
    names = dict(zip(order, backbone_frame["backbone_display"].astype(str)))
    names["moozy_patch"] = "MOOZY"
    if len(order) != 20 or len(set(order)) != 20 or "openpath" in order:
        raise ValueError("The report requires the frozen 20-backbone TCGA panel")
    analysis = read_sheet(book, "SUPP_08_Analysis_Long")
    summary = read_sheet(book, "SUPP_04_Paired_Summary")
    tcga_pairwise = pd.concat((read_sheet(book, "SUPP_02_DELTA_Pairwise"),
                               read_sheet(book, "SUPP_09_ReDELTA_Pairwise")),
                              ignore_index=True)
    book.close()
    tcga_pairwise = tcga_holm_pairwise(tcga_pairwise)
    required_metrics = {key for key, _ in METRICS_TCGA}
    selected = analysis[analysis["method"].isin(TCGA_METHODS) &
                        analysis["metric"].isin(required_metrics)].copy()
    selected["value"] = pd.to_numeric(selected["value"], errors="raise")
    expected = 20 * 4 * 2 * 4
    key = ["backbone", "method", "domain", "metric"]
    if (len(selected) != expected or selected.duplicated(key).any() or
            set(selected["backbone"]) != set(order) or
            set(selected["domain"]) != {d for d, _ in DOMAINS} or
            not selected["status"].eq("complete").all() or
            not np.isfinite(selected["value"]).all()):
        raise ValueError("Incomplete/non-finite TCGA 20 × 4 × 2 × 4 inventory")
    observed = selected.groupby(["method", "domain", "metric"])["value"].mean()
    for row in summary.itertuples(index=False):
        name = (row.method, row.domain, row.metric)
        if name in observed.index and not math.isclose(float(row.mean), float(observed[name]), abs_tol=1e-11):
            raise ValueError(f"TCGA summary/long-table mismatch: {name}")
    selected = selected.set_index(key)["value"]
    return order, names, selected, tcga_pairwise, {"tcga_workbook": sha256(workbook)}


def load_tcga_dense_scores(workbook: Path, order: list[str], dense_path: Path):
    """Validate dense patient-level re-evaluation against frozen q25/q50/q75."""
    import openpyxl

    manifest_path = dense_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    horizons = tuple(range(25, 76, 5))
    if (manifest.get("status") != "complete" or
            manifest.get("data_sha256") != sha256(dense_path) or
            set(manifest.get("backbones", [])) != set(order) or
            set(manifest.get("methods", [])) != set(SURVIVAL_METHODS) or
            manifest.get("quantile_percent") != list(horizons) or
            manifest.get("canonical_q25_q50_q75_max_abs_error", math.inf) > 1e-8):
        raise ValueError("Dense AUC manifest does not match the frozen 20-backbone analysis")
    dense = pd.read_csv(dense_path)
    keys = ["backbone", "method", "domain", "quantile_percent"]
    expected_rows = 20 * len(SURVIVAL_METHODS) * len(DOMAINS) * len(horizons)
    if (len(dense) != expected_rows or dense.duplicated(keys).any() or
            set(dense["backbone"]) != set(order) or
            set(dense["method"]) != set(SURVIVAL_METHODS) or
            set(dense["domain"]) != {domain for domain, _ in DOMAINS} or
            set(dense["quantile_percent"]) != set(horizons) or
            not np.isfinite(dense[["time_days", "td_auc"]].to_numpy(dtype=float)).all()):
        raise ValueError("Incomplete or non-finite dense TCGA AUC inventory")
    if not dense["td_auc"].between(0, 1).all():
        raise ValueError("Dense AUC outside [0, 1]")
    day_ranges = dense.groupby("quantile_percent")["time_days"].agg(["min", "max"])
    if not np.allclose(day_ranges["min"], day_ranges["max"], rtol=0, atol=1e-10):
        raise ValueError("Training event-time quantiles differ across methods/backbones")
    book = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    analysis = read_sheet(book, "SUPP_08_Analysis_Long")
    book.close()
    frozen_horizons = tuple(f"td_auc_q{q}" for q in (25, 50, 75))
    selected = analysis[analysis["method"].isin(SURVIVAL_METHODS) &
                        analysis["metric"].isin(frozen_horizons)].copy()
    key = ["backbone", "method", "domain", "metric"]
    if (len(selected) != 20 * len(SURVIVAL_METHODS) * 2 * 3 or selected.duplicated(key).any() or
            not selected["status"].eq("complete").all() or
            set(selected["backbone"]) != set(order) or
            set(selected["domain"]) != {d for d, _ in DOMAINS}):
        raise ValueError("Incomplete TCGA time-dependent AUC inventory")
    selected["value"] = pd.to_numeric(selected["value"], errors="raise")
    if not np.isfinite(selected["value"].to_numpy()).all():
        raise ValueError("Non-finite TCGA time-dependent AUC")
    values = selected.set_index(key)["value"]
    dense_values = dense.set_index(keys)["td_auc"]
    for model in order:
        for domain, _ in DOMAINS:
            for method in SURVIVAL_METHODS:
                for q in (25, 50, 75):
                    if not math.isclose(
                            float(dense_values.loc[(model, method, domain, q)]),
                            float(values.loc[(model, method, domain, f"td_auc_q{q}")]),
                            rel_tol=0, abs_tol=1e-8):
                        raise ValueError(f"Dense/workbook AUC mismatch: {model}/{method}/{domain}/q{q}")
    rng = np.random.default_rng(42)
    bootstrap_indices = rng.integers(0, len(order), size=(20_000, len(order)))
    results = {}
    for domain, _ in DOMAINS:
        for method in SURVIVAL_METHODS:
            for horizon in horizons:
                scores = np.array([dense_values.loc[(model, method, domain, horizon)]
                                   for model in order], dtype=float)
                draws = scores[bootstrap_indices].mean(axis=1)
                low, high = np.quantile(draws, [.025, .975])
                results[(domain, method, horizon)] = {
                    "mean": float(scores.mean()),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                }
    times = {int(q): float(day_ranges.loc[q, "min"]) for q in horizons}
    return horizons, results, times, manifest


def tcga_effects(pairwise: pd.DataFrame):
    selected = pairwise.set_index(["target", "control", "domain", "metric"])
    results = {}
    for branch, (target, control) in enumerate(TCGA_EFFECTS):
        for domain, _ in DOMAINS:
            for metric, _ in METRICS_TCGA:
                row = selected.loc[(target, control, domain, metric)]
                raw = float(row["target_mean"] - row["control_mean"])
                if metric == "integrated_brier_score":
                    low, high = -float(row["ci95_high"]), -float(row["ci95_low"])
                    if not math.isclose(raw, -float(row["favorable_delta_mean"]), abs_tol=1e-11):
                        raise ValueError("IBS paired-effect sign mismatch")
                else:
                    low, high = float(row["ci95_low"]), float(row["ci95_high"])
                    if not math.isclose(raw, float(row["favorable_delta_mean"]), abs_tol=1e-11):
                        raise ValueError("TCGA paired-effect mismatch")
                results[(branch, domain, metric)] = (
                    raw, low, high, float(row["p_holm_within_family"]))
    return results


def tcga_comparison_rows(pairwise: pd.DataFrame, order: list[str]):
    """Read only requested frozen 20-backbone paired contrasts."""
    frame = pairwise.set_index(["target", "control", "domain", "metric"])
    if frame.index.has_duplicates:
        raise ValueError("Duplicate TCGA pairwise contrast")
    rows = {}
    for domain, _ in DOMAINS:
        domain_rows = []
        for branch, (target, controls) in enumerate(TCGA_COMPARATORS):
            for control in controls:
                effects = {}
                for metric, _ in METRICS_TCGA:
                    key = (target, control, domain, metric)
                    if key not in frame.index:
                        raise ValueError(f"Missing TCGA paired contrast: {key}")
                    row = frame.loc[key]
                    n = int(row["n_backbones"])
                    common = set(str(row["common_backbones"]).split(";"))
                    favourable, low, high, p, p_holm = (float(row[field]) for field in
                                                ("favorable_delta_mean", "ci95_low",
                                                 "ci95_high", "p_raw", "p_holm_within_family"))
                    mean = float(row["target_mean"]) - float(row["control_mean"])
                    if metric == "integrated_brier_score":
                        favourable, low, high = -favourable, -high, -low
                    if (n != 20 or common != set(order) or
                            not all(math.isfinite(v) for v in (mean, low, high, p, p_holm)) or
                            low > high or not 0 <= p <= p_holm <= 1 or
                            int(row["holm_family_size"]) != 4 or
                            not math.isclose(mean, favourable, abs_tol=1e-11)):
                        raise ValueError(f"Invalid paired contrast: {key}")
                    effects[metric] = {"mean_pp": 100 * mean,
                                       "ci95_low_pp": 100 * low,
                                       "ci95_high_pp": 100 * high,
                                       "p_unadjusted": p,
                                       "p_holm_within_family": p_holm,
                                       "holm_family_size": int(row["holm_family_size"]),
                                       "holm_family": str(row["holm_family"])}
                domain_rows.append({"branch": branch, "target": target,
                                    "control": control, "effects": effects})
        rows[domain] = domain_rows
    if {domain: len(domain_rows) for domain, domain_rows in rows.items()} != {
            "in_center": 8, "held_out_center": 8}:
        raise ValueError("Unexpected TCGA comparison inventory")
    return rows


def export_summary(workbook: Path, output_dir: Path, dense_auc: Path | None = None) -> dict:
    order, names, selected, pairwise, hashes = load_tcga_inputs(workbook)
    rows = tcga_comparison_rows(pairwise, order)
    tcga_effects(pairwise)  # Fail closed on incorrect signed IBS intervals.
    comparisons = pd.DataFrame([
        {"domain": domain, "branch": row["branch"], "target": row["target"],
         "control": row["control"], "metric": metric, "n_backbones": len(order),
         **effect}
        for domain, entries in rows.items() for row in entries
        for metric, effect in row["effects"].items()
    ])
    output_dir.mkdir(parents=True, exist_ok=True)
    pairwise.to_csv(output_dir / "paired_contrasts_holm.csv", index=False)
    comparisons.to_csv(output_dir / "manuscript_comparisons.csv", index=False)
    selected.rename("value").reset_index().to_csv(output_dir / "absolute_endpoints.csv", index=False)
    if dense_auc is not None:
        horizons, scores, times, _ = load_tcga_dense_scores(workbook, order, dense_auc)
        dense_rows = [{"domain": domain, "method": method, "quantile_percent": q,
                       "time_days": times[q], **value}
                      for (domain, method, q), value in scores.items()]
        pd.DataFrame(dense_rows).to_csv(output_dir / "time_dependent_auc_summary.csv", index=False)
        hashes["dense_auc"] = sha256(dense_auc)
    manifest = {"status": "complete", "source_sha256": hashes, "backbones": order,
                "statistical_unit": "backbone", "comparison_rows": len(comparisons),
                "holm_family_size": 4,
                "holm_family": "endpoint, domain and DELTA input branch",
                "difference_direction": "target_minus_control", "ibs_lower_is_better": True}
    (output_dir / "summary_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dense-auc", type=Path)
    args = parser.parse_args()
    print(json.dumps(export_summary(args.workbook, args.output_dir, args.dense_auc), sort_keys=True))


if __name__ == "__main__":
    main()
