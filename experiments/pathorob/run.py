"""Run PathoROB representation or supervised ROI-prediction evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.pathorob.evaluation import (
    average_performance_drop,
    evaluate_representation,
    evaluate_roi_prediction,
    evaluate_roi_prediction_inventory,
    thread_count_from_environment,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", type=Path, required=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    representation = sub.add_parser("representation")
    common(representation)
    representation.add_argument("--features", type=Path, required=True)
    representation.add_argument("--pathorob-root", type=Path, required=True)
    representation.add_argument("--pathorob-metadata-dir", type=Path, required=True)
    representation.add_argument("--dry-run", action="store_true")

    classification = sub.add_parser("classification")
    common(classification)
    feature_source = classification.add_mutually_exclusive_group(required=True)
    feature_source.add_argument("--features", type=Path)
    feature_source.add_argument("--feature-inventory", type=Path)
    classification.add_argument("--repetitions", type=int, default=20)
    classification.add_argument(
        "--workers", type=int, default=thread_count_from_environment(1)
    )
    classification.add_argument("--tolkach-split-resource", type=Path)
    classification.add_argument(
        '--raw-reference-scores', type=Path,
        help='Existing same-encoder RAW roi_balanced_accuracy.csv for common-reference APD',
    )
    args = parser.parse_args()
    if args.command == 'classification' and args.method != 'RAW' and args.raw_reference_scores is None:
        parser.error('Non-RAW classification requires --raw-reference-scores; APD never uses a method-specific baseline')

    args.output.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.metadata)
    if args.command == "representation":
        features = np.load(args.features, mmap_mode="r", allow_pickle=False)
        result = evaluate_representation(
            features,
            metadata,
            dataset=args.dataset,
            feature_id=args.method,
            output_dir=args.output,
            metadata_dir=args.pathorob_metadata_dir,
            pathorob_root=args.pathorob_root,
            dry_run=args.dry_run,
        )
        pd.DataFrame(
            [
                {"dataset": args.dataset, "method": args.method, "endpoint": key, "value": value}
                for key, value in asdict(result).items()
                if key not in {"selected_k", "robustness_applicable", "clustering_applicable"}
            ]
        ).to_csv(args.output / "representation_metrics.csv", index=False)
    else:
        parameters = {
            "dataset": args.dataset,
            "method": args.method,
            "repetitions": args.repetitions,
            "workers": args.workers,
            "tolkach_split_resource": args.tolkach_split_resource,
        }
        if args.feature_inventory:
            inventory = pd.read_csv(args.feature_inventory)
            required = {"split_id", "features"}
            missing = required.difference(inventory.columns)
            if missing or inventory["split_id"].duplicated().any():
                raise ValueError(
                    f"invalid split-local feature inventory; missing={sorted(missing)}"
                )
            parent = args.feature_inventory.resolve().parent
            paths = {
                str(row.split_id): (
                    Path(str(row.features))
                    if Path(str(row.features)).is_absolute()
                    else parent / str(row.features)
                )
                for row in inventory.itertuples(index=False)
            }
            scores = evaluate_roi_prediction_inventory(paths, metadata, **parameters)
        else:
            features = np.load(args.features, mmap_mode="r", allow_pickle=False)
            scores = evaluate_roi_prediction(features, metadata, **parameters)
        scores.to_csv(args.output / "roi_balanced_accuracy.csv", index=False)
        raw_reference_scores = (
            pd.read_csv(args.raw_reference_scores) if args.raw_reference_scores else None
        )
        average_performance_drop(scores, reference_scores=raw_reference_scores).to_csv(
            args.output / "roi_average_performance_drop.csv", index=False
        )

    manifest = {
        "schema": "delta-pathorob-evaluation-v1",
        "command": args.command,
        "dataset": args.dataset,
        "method": args.method,
        "metadata_sha256": sha256_file(args.metadata),
    }
    if args.command == 'classification':
        manifest['apd_reference_rule'] = 'common_RAW_balanced_split_mean_over_repetitions'
        if args.raw_reference_scores:
            manifest['raw_reference_scores_sha256'] = sha256_file(args.raw_reference_scores)
    if args.command == "representation" or args.features:
        manifest["features_sha256"] = sha256_file(args.features)
    else:
        manifest["feature_inventory_sha256"] = sha256_file(args.feature_inventory)
    (args.output / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
