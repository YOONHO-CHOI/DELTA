"""Collect manifest-validated PLISM numerical results without image outputs.

The condition evaluator verifies feature payloads. This collector rechecks
numerical CSVs, index files, operator files and feature-completion receipts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from .provenance import bind_release_output
from .report import BACKBONES, collect_marker_identity

STAINS = ("MY", "GIV", "GIVH", "GM", "GMH", "GV", "GVH", "HR", "HRH", "KR", "KRH", "LM", "LMH")
SCANNERS = ("S60", "S210", "S360", "SQ", "AT2", "GT450", "P")

def json_load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def aggregate_multistain_results(analysis_root: Path) -> dict[str, Any] | None:
    """Collect complete 13-stain results; never mix a partial panel into a table."""
    backbones = json_load(analysis_root / 'protocol.yaml')['backbones']
    folders = [analysis_root / 'results' / 'scanners' / backbone for backbone in backbones]
    markers = [folder / 'CONDITIONS_COMPLETE.json' for folder in folders]
    if not any(path.is_file() for path in markers):
        return None
    if not all(path.is_file() for path in markers):
        raise RuntimeError('Multistain panel is partial; publication bundle was not updated')
    physical_frames, retrieval_frames = [], []
    for backbone, folder, marker_path in zip(backbones, folders, markers):
        marker = json_load(marker_path)
        physical_path = folder / 'conditions_summary.csv'
        retrieval_path = folder / 'condition_retrieval_summary.csv'
        if (marker.get('status') != 'complete' or marker.get('backbone') != backbone
                or marker.get('pair_count') != 4095 or marker.get('retrieval_comparisons') != 234
                or sha256(physical_path) != marker.get('summary_sha256')
                or sha256(retrieval_path) != marker.get('retrieval_sha256')):
            raise RuntimeError(f'Invalid multistain result manifest: {backbone}')
        physical = pd.read_csv(physical_path)
        retrieval = pd.read_csv(retrieval_path)
        if len(physical) != 4095 * 9 or len(retrieval) != 234 * 9:
            raise RuntimeError(f'Incomplete condition results: {backbone}')
        physical_frames.append(physical)
        retrieval_frames.append(retrieval)
    physical = pd.concat(physical_frames, ignore_index=True)
    retrieval = pd.concat(retrieval_frames, ignore_index=True)
    physical['method'] = physical.method.map(normalise_method)
    retrieval['method'] = retrieval.method.map(normalise_method)
    if physical.method.isna().any() or retrieval.method.isna().any():
        raise RuntimeError('Unknown multistain correction method')
    for frame in (physical, retrieval):
        frame['group'] = np.select(
            [frame.pair_type.eq('scanner_within_stain'),
             frame.pair_type.eq('stain_within_scanner'),
             frame.pair_type.eq('stain_and_scanner')],
            [frame.stain_a if 'stain_a' in frame else frame.query_stain,
             frame.scanner_a if 'scanner_a' in frame else frame.query_scanner,
             'All combined changes'], default='INVALID')
        if frame.group.eq('INVALID').any():
            raise RuntimeError('Unknown PLISM condition comparison class')
    distance_col = 'tissue_macro_distance_reduction'
    top1_col = 'tissue_macro_top1_retrieval'
    tissue_col = 'tissue_macro_tissue_concordance'
    if not np.isfinite(physical[distance_col].to_numpy()).all() or not np.isfinite(
            retrieval[[top1_col, tissue_col]].to_numpy()).all():
        raise RuntimeError('Non-finite multistain summary')
    grouping = ['backbone', 'pair_type', 'group', 'method']
    physical = physical.groupby(grouping, as_index=False)[distance_col].mean()
    retrieval = retrieval.groupby(grouping, as_index=False)[[top1_col, tissue_col]].mean()
    merged = physical.merge(retrieval, on=grouping, validate='one_to_one')
    random = merged.method.str.startswith('random_')
    random_mean = (merged.loc[random].groupby(grouping[:-1], as_index=False)[
        [distance_col, top1_col, tissue_col]].mean().assign(method='Random subspace (five-basis mean)'))
    grouped = pd.concat([merged.loc[~random], random_mean], ignore_index=True)
    if grouped.groupby(['pair_type', 'group', 'method']).backbone.nunique().ne(20).any():
        raise RuntimeError('Not all 20 encoders contribute to each multistain result')
    macro = grouped.groupby(['pair_type', 'group', 'method'], as_index=False)[
        [distance_col, top1_col, tissue_col]].mean()
    counts = macro.groupby('pair_type').size().to_dict()
    if counts != {'scanner_within_stain': 65, 'stain_within_scanner': 35,
                  'stain_and_scanner': 5}:
        raise RuntimeError(f'Unexpected multistain table dimensions: {counts}')
    return {
        'scope': 'Frozen TCGA operators; descriptive PLISM external transfer',
        'warning': 'Different stains are serial sections, so contrasts are not pure stain effects.',
        'distance_pairs': {'scanner_within_stain': 273, 'stain_within_scanner': 546,
                           'stain_and_scanner': 3276},
        'retrieval_references': {'scanner_within_stain': 'S60 within each stain',
                                 'stain_within_scanner': 'MY within each scanner',
                                 'stain_and_scanner': 'MY/S60'},
        'table': macro.to_dict('records'),
        'manifest_sha256': {backbone: sha256(marker)
                            for backbone, marker in zip(backbones, markers)},
    }

def normalise_method(value: str) -> str:
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    if key in {"delta", "delta_v1_3_p2", "delta_v1.3_p2"}:
        return "DELTA-v1.3 p2"
    if key in {"original", "raw", "original_embedding"}:
        return "Original embedding"
    if "fitting" in key and "pca" in key:
        return "Fitting-bank PCA"
    if "unpaired" in key:
        return "Unpaired displacement"
    if key.startswith("random"):
        return key
    raise ValueError(f"Unexpected method label: {value!r}")

def aggregate_results(analysis_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scanner_root = analysis_root / "results" / "scanners"
    physical_paths = sorted(scanner_root.glob("*/physical_capture_by_tissue.csv"))
    retrieval_paths = sorted(scanner_root.glob("*/retrieval_by_tissue.csv"))
    if not physical_paths or len(physical_paths) != len(retrieval_paths):
        raise RuntimeError(
            f"Incomplete result inventory: physical={len(physical_paths)}, retrieval={len(retrieval_paths)}"
        )

    physical = pd.concat((pd.read_csv(path) for path in physical_paths), ignore_index=True)
    retrieval = pd.concat((pd.read_csv(path) for path in retrieval_paths), ignore_index=True)
    physical["method_normalised"] = physical["method"].map(normalise_method)
    retrieval["method_normalised"] = retrieval["method"].map(normalise_method)

    join_keys = ["backbone", "method", "tissue_type", "method_normalised"]
    merged = physical.merge(retrieval, on=join_keys, how="inner", validate="one_to_one")
    metrics = [
        "removed_fraction",
        "top1",
        "within_tissue_top1",
        "reciprocal_rank",
        "nonidentical_field_tissue_consistency",
    ]
    if not np.isfinite(merged[metrics].to_numpy(dtype=float)).all():
        raise RuntimeError("PLISM result table contains non-finite values")

    # First average the 46 tissue categories within each encoder and basis.
    by_basis = (
        merged.groupby(["backbone", "method_normalised"], as_index=False)[metrics]
        .mean()
        .sort_values(["backbone", "method_normalised"])
    )
    random_mask = by_basis["method_normalised"].str.startswith("random")
    random_mean = (
        by_basis.loc[random_mask]
        .groupby("backbone", as_index=False)[metrics]
        .mean()
        .assign(method="Random subspace (five-basis mean)")
    )
    fixed = by_basis.loc[~random_mask].rename(columns={"method_normalised": "method"})
    by_backbone = pd.concat([fixed, random_mean], ignore_index=True)

    expected_methods = {
        "DELTA-v1.3 p2",
        "Original embedding",
        "Fitting-bank PCA",
        "Unpaired displacement",
        "Random subspace (five-basis mean)",
    }
    actual_methods = set(by_backbone["method"])
    if actual_methods != expected_methods:
        raise RuntimeError(f"Unexpected method inventory: {sorted(actual_methods)}")
    backbone_count = by_backbone["backbone"].nunique()
    if backbone_count != 20:
        raise RuntimeError(f"Expected 20 encoders, found {backbone_count}")

    macro = by_backbone.groupby("method", as_index=False)[metrics].mean()
    original = macro.set_index("method").loc["Original embedding"]
    for metric in metrics:
        macro[f"delta_from_original_{metric}"] = macro[metric] - float(original[metric])

    method_order = [
        "Original embedding",
        "DELTA-v1.3 p2",
        "Fitting-bank PCA",
        "Unpaired displacement",
        "Random subspace (five-basis mean)",
    ]
    macro["_order"] = macro["method"].map({value: i for i, value in enumerate(method_order)})
    by_backbone["_order"] = by_backbone["method"].map(
        {value: i for i, value in enumerate(method_order)}
    )
    macro = macro.sort_values("_order").drop(columns="_order")
    by_backbone = by_backbone.sort_values(["backbone", "_order"]).drop(columns="_order")
    return macro.to_dict("records"), by_backbone.to_dict("records")


def collect_source_receipts(analysis_root, input_root):
    protocol = json_load(analysis_root / "protocol.yaml")
    if len(protocol["backbones"]) != 20 or set(protocol["backbones"]) != set(BACKBONES):
        raise ValueError("Expected the frozen 20-encoder panel")
    indices = {}
    for stain in STAINS:
        suffix = "" if stain == "MY" else "_" + stain
        ready = json_load(input_root / "plism" / f"READY{suffix}.json")
        index = input_root / "plism" / f"index{suffix}.parquet"
        indices[stain] = sha256(index)
        if ready.get("status") != "complete" or ready.get("index_sha256") != indices[stain]:
            raise ValueError("PLISM index does not match its frozen receipt")
    receipts = {}
    for backbone in BACKBONES:
        folder = analysis_root / "results" / "scanners" / backbone
        path = folder / "CONDITIONS_COMPLETE.json"
        marker = json_load(path)
        identity = collect_marker_identity(marker)
        if marker.get("status") != "complete" or marker.get("backbone") != backbone or identity["index_sha256"] != indices:
            raise ValueError("Inconsistent PLISM condition receipt")
        feature_markers = {}
        for stain in STAINS:
            filename = "FEATURES_COMPLETE.json" if stain == "MY" else f"FEATURES_{stain}_COMPLETE.json"
            feature_path = folder / filename
            feature = json_load(feature_path)
            if feature.get("status") != "complete" or feature.get("index_sha256") != indices[stain]:
                raise ValueError("Invalid feature-completion receipt")
            for scanner in SCANNERS:
                if feature["feature_sha256"][scanner] != identity["feature_sha256"][f"{stain}_{scanner}"]:
                    raise ValueError("Feature identity differs across result receipts")
            feature_markers[stain] = sha256(feature_path)
        for method, expected in identity["operator_sha256"].items():
            operator = (input_root / "tcga" / backbone / "delta_operator.npz" if method == "DELTA-v1.3-p2"
                        else analysis_root / "results" / "specificity" / backbone / f"{method}.npz")
            if sha256(operator) != expected:
                raise ValueError("Frozen operator changed")
        physical, retrieval = folder / "conditions_summary.csv", folder / "condition_retrieval_summary.csv"
        for file, key in ((physical, "summary_sha256"), (retrieval, "retrieval_sha256")):
            if sha256(file) != marker[key]:
                raise ValueError("Numerical CSV differs from its frozen receipt")
        receipts[backbone] = {**identity, "manifest_sha256": sha256(path),
                              "summary_sha256": marker["summary_sha256"],
                              "retrieval_sha256": marker["retrieval_sha256"],
                              "feature_marker_sha256": feature_markers}
        print("PLISM_COLLECTION_SOURCE_VERIFIED", backbone, flush=True)
    return receipts


def build_bundle(analysis_root, input_root):
    receipts = collect_source_receipts(analysis_root, input_root)
    index = pd.read_parquet(input_root / "plism" / "index.parquet")
    required = {"scanner", "index", "coordinate", "tissue_type"}
    if not required.issubset(index.columns) or len(index) != 23919 or index["index"].nunique() != 3417:
        raise ValueError("Invalid complete MY acquisition inventory")
    if set(index["scanner"]) != set(SCANNERS) or index["tissue_type"].nunique() != 46:
        raise ValueError("Incomplete scanner/tissue inventory")
    multistain = aggregate_multistain_results(analysis_root)
    if multistain is None:
        raise ValueError("Complete 13-condition numerical results are required")
    macro, backbones = aggregate_results(analysis_root)
    result = {"schema": "plism-numeric-bundle-v1", "status": "complete",
              "dataset": {"staining_conditions": 13, "scanners": 7, "matched_fields": 3417,
                          "tissue_categories": 46, "encoders": 20,
                          "operator_source": "Frozen TCGA-BRCA operators; no PLISM refitting"},
              "source_receipts": receipts, "multistain": multistain,
              "macro_results": macro, "backbone_results": backbones}
    return bind_release_output(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True, help="Frozen input directory containing plism/ and tcga/")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(args.output_json)
    bundle = build_bundle(args.analysis_root, args.input_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output_json.with_name(args.output_json.name + ".partial")
    if temp.exists():
        raise FileExistsError(temp)
    temp.write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, args.output_json)
    print("PLISM_NUMERIC_COLLECTION_COMPLETE", args.output_json)


if __name__ == "__main__":
    main()

