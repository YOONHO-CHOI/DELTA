"""Validate and export the frozen HUMC manuscript inference tables.

This report reads previously computed four-arm case/backbone hierarchical
inference. It preserves the five-comparison Holm family within each endpoint
and never substitutes backbone-only variation for the primary uncertainty.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from .scoring import BACKBONES

HUMC_METHODS = ('raw', 'reinhard_raw', 'delta_v1_3_p2', 'reinhard_delta_v1_3_p2')
METRICS_HUMC = (('auroc', 'AUROC'), ('auprc', 'AUPRC'), ('sensitivity', 'Sensitivity'), ('specificity', 'Specificity'))
HUMC_EFFECTS = (('delta_minus_raw', 'DELTA − RAW'), ('reinhard_delta_minus_reinhard_raw', 'R-DELTA − R-RAW'))
HUMC_COMPARISONS = ((('delta_minus_raw', 'raw'), ('delta_minus_reinhard_raw', 'reinhard_raw')), (('reinhard_delta_minus_raw', 'raw'), ('reinhard_delta_minus_reinhard_raw', 'reinhard_raw')))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_humc_inputs(humc_summary: Path, humc_inference: Path, order: list[str]):
    hashes = {}
    humc = pd.read_csv(humc_summary)
    inference = pd.read_csv(humc_inference)
    if (len(humc) != 80 or humc.duplicated(["backbone", "arm"]).any() or
            set(humc["backbone"]) != set(order) or
            set(humc["arm"]) != set(HUMC_METHODS) or
            not humc["status"].eq("complete").all() or
            not np.isfinite(humc[[m for m, _ in METRICS_HUMC]].to_numpy(float)).all() or
            set(humc["n_cancer_cases"]) != {32} or set(humc["n_nt_cases"]) != {100}):
        raise ValueError("Incomplete/non-finite HUMC four-arm inventory")
    if inference.duplicated(["comparison", "metric"]).any():
        raise ValueError("Duplicate HUMC paired inference rows")
    humc = humc.set_index(["backbone", "arm"])
    inference = inference.set_index(["comparison", "metric"])
    for comparison, arm_a, arm_b in (
        ("delta_minus_raw", "raw", "delta_v1_3_p2"),
        ("reinhard_delta_minus_reinhard_raw", "reinhard_raw", "reinhard_delta_v1_3_p2"),
    ):
        for metric, _ in METRICS_HUMC:
            row = inference.loc[(comparison, metric)]
            diff = np.mean([humc.loc[(b, arm_b), metric] - humc.loc[(b, arm_a), metric]
                            for b in order])
            if not math.isclose(float(row["effect_b_minus_a"]), float(diff), abs_tol=1e-10):
                raise ValueError(f"HUMC inference/long-table mismatch: {comparison}/{metric}")
    hashes.update({"humc_summary": sha256(humc_summary),
                   "humc_inference": sha256(humc_inference)})
    return humc, inference, hashes


def humc_comparison_rows(inference: pd.DataFrame, humc: pd.DataFrame,
                         order: list[str]):
    """Validate frozen HUMC hierarchical contrasts against the four-arm means."""
    frame = inference
    if list(frame.index.names) != ["comparison", "metric"] or frame.index.has_duplicates:
        raise ValueError("Duplicate HUMC comparison/metric row")
    rows = {}
    for branch, comparisons in enumerate(HUMC_COMPARISONS):
        target = "delta_v1_3_p2" if branch == 0 else "reinhard_delta_v1_3_p2"
        rows[branch] = []
        for comparison, control in comparisons:
            effects = {}
            for metric, _ in METRICS_HUMC:
                if (comparison, metric) not in frame.index:
                    raise ValueError(f"Missing HUMC paired comparison: {comparison}/{metric}")
                row = frame.loc[(comparison, metric)]
                observed = float(np.mean([
                    humc.loc[(model, target), metric] - humc.loc[(model, control), metric]
                    for model in order
                ]))
                effect, low, high, p = (float(row[field]) for field in (
                    "effect_b_minus_a", "ci95_lower", "ci95_upper",
                    "p_holm_within_metric"))
                if (row["arm_a"] != control or row["arm_b"] != target or
                        not all(math.isfinite(v) for v in (effect, low, high, p)) or
                        low > effect or effect > high or not 0 <= p <= 1 or
                        not math.isclose(observed, effect, abs_tol=1e-10)):
                    raise ValueError(f"Invalid HUMC paired comparison: {comparison}/{metric}")
                effects[metric] = {
                    "mean_pp": 100 * effect,
                    "ci95_low_pp": 100 * low,
                    "ci95_high_pp": 100 * high,
                    "p_holm_within_metric": p,
                }
            rows[branch].append({"comparison": comparison, "target": target,
                                 "control": control, "effects": effects})
    return rows


def humc_effects(inference: pd.DataFrame):
    return {(branch, metric):
            tuple(float(inference.loc[(comparison, metric), field])
                  for field in ("effect_b_minus_a", "ci95_lower", "ci95_upper",
                                "p_holm_within_metric"))
            for branch, (comparison, _) in enumerate(HUMC_EFFECTS)
            for metric, _ in METRICS_HUMC}


def export_summary(summary_path: Path, inference_path: Path,
                   output_dir: Path, order: list[str]) -> dict:
    if len(order) != 20 or len(set(order)) != 20:
        raise ValueError("The manuscript report requires 20 distinct frozen backbones")
    humc, inference, hashes = load_humc_inputs(summary_path, inference_path, order)
    rows = humc_comparison_rows(inference, humc, order)
    humc_effects(inference)
    comparisons = pd.DataFrame([
        {"branch": branch, "comparison": row["comparison"], "target": row["target"],
         "control": row["control"], "metric": metric, **effect}
        for branch, entries in rows.items() for row in entries
        for metric, effect in row["effects"].items()
    ])
    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons.to_csv(output_dir / "manuscript_comparisons.csv", index=False)
    humc.reset_index().to_csv(output_dir / "absolute_endpoints.csv", index=False)
    inference.reset_index().to_csv(output_dir / "hierarchical_pairwise_inference.csv", index=False)
    manifest = {"status": "complete", "source_sha256": hashes, "backbones": order,
                "comparison_rows": len(comparisons), "statistical_units": ["case", "backbone"],
                "inference": "paired case/backbone hierarchical bootstrap",
                "holm_family_size": 5, "holm_family": "within endpoint",
                "difference_direction": "target_minus_control"}
    (output_dir / "summary_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backbones", nargs="+", choices=BACKBONES, default=list(BACKBONES))
    args = parser.parse_args()
    print(json.dumps(export_summary(args.summary, args.inference, args.output_dir,
                                    list(args.backbones)), sort_keys=True))


if __name__ == "__main__":
    main()

