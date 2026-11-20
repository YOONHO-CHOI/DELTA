"""Distance and retrieval endpoints from supplied raw and adjusted features.

Rows must identify spatially corresponding tissue regions in both acquisitions.
This module accepts no operators and performs no embedding adjustment.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def evaluate_corresponding_features(raw_reference, raw_query, adjusted_reference,
                                    adjusted_query, tissue_labels) -> pd.DataFrame:
    raw_a, raw_b, adjusted_a, adjusted_b = [np.asarray(value) for value in (
        raw_reference, raw_query, adjusted_reference, adjusted_query)]
    tissues = np.asarray(tissue_labels)
    if raw_a.ndim != 2 or len(raw_a) < 2 or any(value.shape != raw_a.shape
            or not np.isfinite(value).all() for value in (raw_a, raw_b, adjusted_a, adjusted_b)):
        raise ValueError("Four finite, corresponding feature matrices with at least two rows are required")
    if tissues.ndim != 1 or len(tissues) != len(raw_a):
        raise ValueError("Tissue labels must align with corresponding feature rows")
    original = np.sum((raw_a - raw_b).astype(np.float64) ** 2, axis=1)
    corrected = np.sum((adjusted_a - adjusted_b).astype(np.float64) ** 2, axis=1)
    fraction = np.divide(original - corrected, original, out=np.full(len(tissues), np.nan), where=original > 1e-12)
    gallery, query = adjusted_a, adjusted_b
    distances = np.maximum(np.sum(query ** 2, axis=1)[:, None]
                           + np.sum(gallery ** 2, axis=1)[None, :] - 2 * (query @ gallery.T), 0)
    truth = np.diag(distances).copy()
    ranks = 1 + np.sum(distances < truth[:, None], axis=1)
    within = distances.copy()
    within[tissues[:, None] != tissues[None, :]] = np.inf
    within_ranks = 1 + np.sum(within < truth[:, None], axis=1)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argmin(distances, axis=1)
    return pd.DataFrame(dict(tissue_type=tissues, raw_squared_distance=original,
        corrected_squared_distance=corrected, removed_fraction=fraction, matched_field_rank=ranks,
        reciprocal_rank=1 / ranks, top1=ranks == 1, within_tissue_top1=within_ranks == 1,
        nonidentical_field_tissue_consistency=tissues[neighbors] == tissues))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="External NPZ with four feature matrices and tissue_labels")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.input, allow_pickle=False) as values:
        frame = evaluate_corresponding_features(*(values[key] for key in (
            "raw_reference", "raw_query", "adjusted_reference", "adjusted_query", "tissue_labels")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    frame.to_csv(temporary, index=False)
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
