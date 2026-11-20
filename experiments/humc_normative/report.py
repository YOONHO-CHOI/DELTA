"""Create numeric HUMC four-arm tables from completed case-level OOF artifacts.

The report preserves paired case and backbone inference, calibration thresholds,
colour audits and consolidated CSV/Excel outputs. It never extracts embeddings
or fits DELTA operators.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .scoring import (
    BACKBONES, SEED, V13_ARMS, bootstrap_backbone_mean,
    hierarchical_four_arm_inference,
)
from .report_io import (
    DEFAULT_NT_PATCHES, DEFAULT_CANCER_PATCHES, DEFAULT_OUTPUT_ROOT,
    PUBLICATION_OPTIONS, SCHEMA_VERSION,
    atomic_csv, atomic_json, atomic_parquet,
    evaluation_paths, script_identity, v13_prediction_paths,
    write_consolidated_results_workbook,
)


def command_v13_report(args: argparse.Namespace) -> None:
    """Create the locked multi-backbone four-arm result and minimal colour sanity audit."""

    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    output_root = Path(args.output_root)
    baseline_root = Path(args.baseline_root) if getattr(args, "baseline_root", None) else output_root
    backbones = tuple(args.backbones)
    if len(backbones) != len(set(backbones)):
        raise ValueError("duplicate backbones in v1.3 report")
    statistics = output_root / "statistics"
    summaries = []
    case_by_key: dict[tuple[str, str], pd.DataFrame] = {}
    for backbone in backbones:
        summary_path = statistics / f"{backbone}_humc_only_4arm_summary.parquet"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = pd.read_parquet(summary_path)
        if set(summary["arm"].astype(str)) != set(V13_ARMS):
            raise ValueError(f"four-arm coverage mismatch: {backbone}")
        summaries.append(summary.assign(backbone=backbone))
        reference = None
        for arm in V13_ARMS:
            path = (
                evaluation_paths(baseline_root, backbone)["case"]
                if arm == "raw"
                else v13_prediction_paths(
                    baseline_root if arm == "reinhard_raw" else output_root,
                    backbone,
                    arm,
                )["case"]
            )
            frame = pd.read_parquet(path).sort_values("case_identifier").reset_index(drop=True)
            keys = frame[["case_identifier", "label", "fold"]]
            if reference is None:
                reference = keys
            else:
                pd.testing.assert_frame_equal(reference, keys, check_dtype=False)
            if len(frame) != 132 or int(frame["label"].sum()) != 32:
                raise ValueError(f"OOF case coverage mismatch: {backbone}:{arm}")
            case_by_key[(backbone, arm)] = frame
    summary_long = pd.concat(summaries, ignore_index=True)
    atomic_csv(summary_long, statistics / "humc_20backbone_4arm_summary_long.csv")
    atomic_parquet(summary_long, statistics / "humc_20backbone_4arm_summary_long.parquet")

    contrast_rows = []
    for backbone in backbones:
        indexed = summary_long.loc[summary_long["backbone"].eq(backbone)].set_index("arm")
        for metric in ("auroc", "auprc", "sensitivity", "specificity"):
            values = indexed[metric].astype(float)
            definitions = {
                "reinhard_minus_raw": values["reinhard_raw"] - values["raw"],
                "delta_minus_raw": values["delta_v1_3_p2"] - values["raw"],
                "combined_minus_raw": values["reinhard_delta_v1_3_p2"] - values["raw"],
                "combined_minus_reinhard": values["reinhard_delta_v1_3_p2"] - values["reinhard_raw"],
                "combined_minus_delta": values["reinhard_delta_v1_3_p2"] - values["delta_v1_3_p2"],
                "factorial_interaction": values["reinhard_delta_v1_3_p2"]
                - values["reinhard_raw"] - values["delta_v1_3_p2"] + values["raw"],
            }
            for contrast, value in definitions.items():
                contrast_rows.append(
                    {"backbone": backbone, "metric": metric, "contrast": contrast, "value": float(value)}
                )
    contrasts = pd.DataFrame(contrast_rows)
    atomic_csv(contrasts, statistics / "humc_20backbone_backbone_contrasts.csv")
    atomic_parquet(contrasts, statistics / "humc_20backbone_backbone_contrasts.parquet")
    aggregate_rows = []
    for (metric, contrast), group in contrasts.groupby(["metric", "contrast"], sort=True):
        values = group["value"].to_numpy(dtype=np.float64)
        lower, upper = bootstrap_backbone_mean(values, int(args.bootstrap_reps), SEED)
        aggregate_rows.append(
            {
                "metric": metric,
                "contrast": contrast,
                "n_backbones": int(len(values)),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "ci95_lower": lower,
                "ci95_upper": upper,
                "positive_backbones": int(np.sum(values > 0)),
                "negative_backbones": int(np.sum(values < 0)),
                "zero_backbones": int(np.sum(values == 0)),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    atomic_csv(aggregate, statistics / "humc_20backbone_contrast_summary.csv")
    atomic_parquet(aggregate, statistics / "humc_20backbone_contrast_summary.parquet")

    arm_inference, pairwise_inference, backbone_pairwise_effects = hierarchical_four_arm_inference(
        summary_long,
        case_by_key,
        backbones,
        int(args.bootstrap_reps),
        SEED,
    )
    atomic_csv(arm_inference, statistics / "humc_multibackbone_arm_performance_inference.csv")
    atomic_parquet(arm_inference, statistics / "humc_multibackbone_arm_performance_inference.parquet")
    atomic_csv(pairwise_inference, statistics / "humc_multibackbone_pairwise_arm_inference.csv")
    atomic_parquet(pairwise_inference, statistics / "humc_multibackbone_pairwise_arm_inference.parquet")
    atomic_csv(backbone_pairwise_effects, statistics / "humc_multibackbone_pairwise_effects_by_backbone.csv")
    atomic_parquet(backbone_pairwise_effects, statistics / "humc_multibackbone_pairwise_effects_by_backbone.parquet")

    color_path = Path(args.color_audit)
    color = pd.read_parquet(color_path).sort_values("case_identifier").reset_index(drop=True)
    required_color = {"case_identifier", "cohort", "label", "lab_distance_to_nt_center"}
    if not required_color.issubset(color.columns) or color.case_identifier.duplicated().any():
        raise ValueError("A complete case-level color-audit table is required")
    color_label_auroc = float(roc_auc_score(color["label"], color["lab_distance_to_nt_center"]))
    color_rows = []
    for backbone in backbones:
        for arm in V13_ARMS:
            case = case_by_key[(backbone, arm)]
            if not case["case_identifier"].equals(color["case_identifier"]):
                raise ValueError(f"colour audit case order mismatch: {backbone}:{arm}")
            all_rho = float(spearmanr(case["anomaly_score"], color["lab_distance_to_nt_center"]).statistic)
            nt_mask = color["cohort"].eq("NT").to_numpy()
            nt_rho = float(
                spearmanr(
                    case.loc[nt_mask, "anomaly_score"],
                    color.loc[nt_mask, "lab_distance_to_nt_center"],
                ).statistic
            )
            color_rows.append(
                {
                    "backbone": backbone,
                    "arm": arm,
                    "spearman_all_cases": all_rho,
                    "spearman_nt_cases": nt_rho,
                    "color_distance_label_auroc": color_label_auroc,
                }
            )
    color_audit = pd.DataFrame(color_rows)
    atomic_csv(color_audit, statistics / "humc_20backbone_color_sanity.csv")
    atomic_parquet(color_audit, statistics / "humc_20backbone_color_sanity.parquet")

    arm_labels = {
        "raw": "Raw",
        "reinhard_raw": "Reinhard",
        "delta_v1_3_p2": "DELTA",
        "reinhard_delta_v1_3_p2": "Reinhard-DELTA",
    }
    # Canonical publication names: the keys of BACKBONE_IDS in the locked
    # DELTA WSI-downstream reporting code, indexed here by the runtime IDs.
    backbone_display_names = {
        "uni": "Original UNI",
        "phikonv2": "Phikon-v2",
        "virchow2": "Virchow2",
        "conch": "CONCH",
        "conchv15": "CONCH v1.5",
        "uni2h": "UNI2-h",
        "hoptimus0": "H-optimus-0",
        "h0mini": "H0-mini",
        "exaonepath25": "EXAONE-Path-2.5",
        "gpfm": "GPFM",
        "moozy_patch": "MOOZY (Lunit DINO)",
        "gigapath": "Prov-GigaPath",
        "musk": "MUSK",
        "ciga": "Ciga",
        "hipt": "HIPT",
        "retccl": "RetCCL",
        "ctranspath": "CTransPath",
        "kaiko_vitb8": "Kaiko ViT-B/8",
        "virchow": "Virchow",
        "beph": "BEPH",
    }
    missing_display_names = set(backbones).difference(backbone_display_names)
    if missing_display_names:
        raise ValueError(f"missing canonical backbone display names: {sorted(missing_display_names)}")

    workbook_summary = summary_long.copy()
    workbook_summary.insert(
        0,
        "model",
        workbook_summary["backbone"].map(backbone_display_names),
    )
    workbook_summary.insert(
        2,
        "method",
        workbook_summary["arm"].map(arm_labels),
    )
    workbook_summary["fn"] = np.rint(
        32 * (1.0 - workbook_summary["sensitivity"].astype(float))
    ).astype(int)
    workbook_summary["fp"] = np.rint(
        100 * (1.0 - workbook_summary["specificity"].astype(float))
    ).astype(int)
    workbook_priority_columns = [
        "model", "method", "auroc", "auprc", "sensitivity", "specificity",
        "fn", "fp", "backbone", "arm",
    ]
    workbook_summary = workbook_summary[
        workbook_priority_columns
        + [column for column in workbook_summary.columns if column not in workbook_priority_columns]
    ]
    workbook_sheets: dict[str, pd.DataFrame] = {
        "All Results": workbook_summary,
        "Arm Inference": arm_inference.assign(
            method=arm_inference["arm"].map(arm_labels)
        ),
        "Pairwise Inference": pairwise_inference,
        "Backbone Effects": backbone_pairwise_effects.assign(
            model=backbone_pairwise_effects["backbone"].map(backbone_display_names)
        ),
        "Contrast Summary": aggregate,
        "Color Audit": color_audit.assign(
            model=color_audit["backbone"].map(backbone_display_names),
            method=color_audit["arm"].map(arm_labels),
        ),
        "Definitions": pd.DataFrame(
            [
                ("Cohort", "100 NT and 32 Cancer cases; Cancer is excluded from fitting."),
                ("Prediction", "Case-level pooled 5-fold OOF anomaly scores; higher indicates Cancer."),
                ("Operating point", "Fold-specific 95th percentile of NT training scores."),
                ("Primary metrics", "AUROC, AUPRC, sensitivity, specificity, FN, and FP."),
                ("DELTA", "DELTA v1.3 p2; no TCGA data used."),
                ("Inference", "Paired HUMC-case plus resampled-backbone hierarchical bootstrap."),
                ("Multiple testing", "Holm correction across five requested comparisons within each metric."),
                ("Workbook policy", "One final workbook; CSV and Parquet remain the machine-readable source of truth."),
            ],
            columns=["item", "definition"],
        ),
    }
    for backbone in backbones:
        model_name = backbone_display_names[backbone]
        excel_model_sheet = "".join(
            "-" if character in "[]:*?/\\" else character
            for character in model_name
        )[:31]
        if excel_model_sheet in workbook_sheets:
            raise ValueError(f"duplicate normalized Excel sheet name: {excel_model_sheet}")
        workbook_sheets[excel_model_sheet] = (
            workbook_summary.loc[workbook_summary["backbone"].eq(backbone)]
            .reset_index(drop=True)
        )
    workbook_path = statistics / str(PUBLICATION_OPTIONS["excel"]["filename"])
    write_consolidated_results_workbook(workbook_path, workbook_sheets)

    completion = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "backbones": list(backbones),
        "arms": list(V13_ARMS),
        "delta_version": "1.3",
        "bank_fraction": "p2",
        "rank_rule": "permutation-null-adjusted 90%-energy",
        "tcga_used": False,
        "color_distance_label_auroc": color_label_auroc,
        "workbook": str(workbook_path),
        "workbook_policy": "one consolidated workbook with integrated and per-backbone sheets",
        "pairwise_inference": {
            "comparisons": [
                "DELTA - Raw",
                "DELTA - Reinhard",
                "Reinhard-DELTA - Raw",
                "Reinhard-DELTA - Reinhard",
                "Reinhard-DELTA - DELTA",
            ],
            "composite_main_effect_used": False,
            "confidence_interval": "stratified paired HUMC-case plus backbone-panel hierarchical bootstrap",
            "p_value": "two-sided centered hierarchical paired-bootstrap test",
            "consistency_test": "paired backbone sign-flip randomization and exact sign test (secondary)",
            "multiple_testing": "Holm across five requested direct arm comparisons within each metric; global 20-test Holm retained",
        },
        "script": script_identity(),
    }
    atomic_json(statistics / "HUMC_20BACKBONE_V13_REPORT.json", completion)
    print(aggregate.to_string(index=False))
    print(json.dumps(completion, ensure_ascii=False, sort_keys=True))
    print("HUMC_20BACKBONE_V13_REPORT_COMPLETE", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    report = subparsers.add_parser("report", help="create the locked four-arm numeric report")
    report.add_argument("--nt-patches", type=Path, default=DEFAULT_NT_PATCHES)
    report.add_argument("--cancer-patches", type=Path, default=DEFAULT_CANCER_PATCHES)
    report.add_argument("--output-root", type=Path, required=True)
    report.add_argument("--color-audit", type=Path, required=True, help="External completed case-level color measurement table")
    report.add_argument("--baseline-root", type=Path)
    report.add_argument("--backbones", nargs="+", choices=BACKBONES, default=BACKBONES)
    report.add_argument("--bootstrap-reps", type=int, default=20_000)
    report.add_argument("--io-workers", type=int, default=8)
    report.set_defaults(func=command_v13_report)
    return parser


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

