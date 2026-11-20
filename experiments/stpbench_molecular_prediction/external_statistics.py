"""Evaluate supplied molecular predictions without model or correction code."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .report import atomic_csv, atomic_text
from .statistics import holm


def approved_index(names, reference):
    """Exact approved symbols or version-stripped, unambiguous Ensembl IDs."""
    clean = reference.fillna("").copy()
    clean["ensembl_gene_id"] = clean["ensembl_gene_id"].str.split(".").str[0].str.upper()
    clean = clean.loc[clean["ensembl_gene_id"].str.match(r"^ENSG\d+$")]
    by_id = clean.drop_duplicates("ensembl_gene_id", keep=False).set_index("ensembl_gene_id")["symbol"].to_dict()
    by_symbol = clean.assign(key=clean["symbol"].str.upper()).drop_duplicates("key", keep=False).set_index("key")["symbol"].to_dict()
    mapped = {}
    for column, name in enumerate(names):
        token = str(name).strip()
        stable = token.split(".", 1)[0].upper()
        symbol = by_id.get(stable) if stable.startswith("ENSG") else by_symbol.get(token.upper())
        if symbol:
            if symbol in mapped:
                raise ValueError(f"Ambiguous target gene after exact mapping: {symbol}")
            mapped[symbol] = column
    return mapped


def cluster_statistics(frame, metric, repetitions, seed):
    """Retain the section-weighted effect; resample whole patient clusters."""
    effect = (frame[f"{metric}_delta"] - frame[f"{metric}_raw"]) * (1 if metric == "pcc" else -1)
    clustered = pd.DataFrame({"patient": frame["patient"], "effect": effect}).groupby("patient", sort=True).effect.agg(["sum", "count"])
    sums, counts = clustered["sum"].to_numpy(), clustered["count"].to_numpy()
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(clustered), size=(repetitions, len(clustered)))
    draws = sums[indices].sum(axis=1)/counts[indices].sum(axis=1)
    observed = abs(float(sums.sum()))
    # Largest target cohort has 23 patients: exhaustively enumerate bounded chunks.
    exceed = total = 0
    for begin in range(0, 2**len(sums), 65536):
        ids = np.arange(begin, min(begin+65536, 2**len(sums)), dtype=np.uint64)
        signs = ((ids[:, None] >> np.arange(len(sums), dtype=np.uint64)) & 1).astype(np.float64)*2-1
        exceed += int(np.count_nonzero(np.abs(signs@sums) >= observed-1e-12))
        total += len(ids)
    return {"raw": float(frame[f"{metric}_raw"].mean()),
            "delta": float(frame[f"{metric}_delta"].mean()),
            "effect": float(effect.mean()), "ci_low": float(np.quantile(draws, .025)),
            "ci_high": float(np.quantile(draws, .975)), "p": exceed/total,
            "patients": len(clustered), "sections": len(frame),
            "ci_method": "patient_cluster_bootstrap_section_weighted_mean",
            "p_method": "exact_patient_cluster_sign_flip"}


def summarize_predictions(frame, repetitions=20_000, seed=42):
    """Keep cohort/panel families separate and resample complete patients."""
    required = {"cohort", "panel", "patient", "pcc_raw", "pcc_delta", "mae_raw", "mae_delta"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("Nonempty cohort/panel/patient paired prediction metrics are required")
    if frame[["cohort", "panel", "patient"]].isna().any().any():
        raise ValueError("Cohort, panel and patient identities must be present")
    values = frame[["pcc_raw", "pcc_delta", "mae_raw", "mae_delta"]].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Paired prediction metrics must be finite")
    if repetitions < 1:
        raise ValueError("Positive bootstrap repetitions are required")
    rows = []
    for (cohort, panel), group in frame.groupby(["cohort", "panel"], sort=True):
        if group.patient.nunique() > 23:
            raise ValueError("Exact patient-cluster sign-flip inference is bounded to 23 patients per task")
        for metric in ("pcc", "mae"):
            rows.append({"cohort": cohort, "panel": panel, "metric": metric,
                         **cluster_statistics(group, metric, repetitions, seed)})
    result = pd.DataFrame(rows)
    for (panel, metric), indices in result.groupby(["panel", "metric"], sort=True).groups.items():
        result.loc[indices, "holm_p"] = holm(result.loc[indices, "p"])
        result.loc[indices, "holm_family"] = f"{panel}:{metric}:external_cohorts"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Supplied section-metric CSV with cohort, panel, patient and paired PCC/MAE")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        raise FileExistsError("Choose a new or empty output directory")
    frame = pd.read_csv(args.input, dtype={"cohort": str, "panel": str, "patient": str})
    result = summarize_predictions(frame, args.bootstrap_repetitions, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(args.output_dir / "external_patient_statistics.csv", result)
    metadata = {"schema": "molecular-external-patient-statistics-v1",
                "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "bootstrap_repetitions": args.bootstrap_repetitions, "seed": args.seed,
                "unit": "patient cluster with section-weighted point estimates",
                "multiplicity": "Holm across cohorts separately per panel and metric"}
    atomic_text(args.output_dir / "audit.json", json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
