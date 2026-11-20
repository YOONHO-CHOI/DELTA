"""Generate and validate split contracts from explicitly supplied metadata.

The package does not bundle experiment cohorts, splits or reference images.
Input metadata and any frozen comparison contracts remain external inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from .splits import (
    APD_SPECS,
    CLASSIFICATION_TASKS,
    REPETITION_SEEDS,
    build_classification_splits,
    build_official_apd_split_family,
    build_official_apd_splits,
    prepare_apd_metadata,
    validate_patient_splits,
)


def _canonical(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "task_id", "split_id", "repeat", "seed", "condition",
        "correlation_strength", "patient_id", "partition", "medical_center",
        "stratification_label", "event", "time_days",
    ]
    return frame[columns].sort_values(["split_id", "patient_id"]).reset_index(drop=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _membership_sha256(frame: pd.DataFrame, column: str) -> str:
    values = sorted(frame[column].dropna().astype(str).unique().tolist())
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def freeze_apd_family(
    metadata_dir: Path,
    output_dir: Path,
    tolkach_resource: Path,
) -> None:
    """Freeze compact membership hashes for the complete pinned APD sequence."""

    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("freeze-apd-family requires PYTHONHASHSEED=0")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    metadata_hashes: dict[str, str] = {}
    for dataset in ("camelyon", "tcga_4x4", "tolkach_esca"):
        metadata_path = metadata_dir / f"{dataset}.csv"
        metadata_hashes[dataset] = _sha256(metadata_path)
        metadata = prepare_apd_metadata(pd.read_csv(metadata_path))
        resource = tolkach_resource if dataset == "tolkach_esca" else None
        splits = build_official_apd_split_family(
            metadata, dataset, REPETITION_SEEDS, resource
        )
        expected = int(APD_SPECS[dataset]["n_splits"]) * len(REPETITION_SEEDS)
        if len(splits) != expected:
            raise ValueError(f"wrong APD split count: {dataset}/{len(splits)}/{expected}")
        for split_index, split in enumerate(splits):
            for partition in ("train", "validation", "id_test", "ood_test"):
                frame = getattr(split, partition)
                rows.append(
                    {
                        "dataset": dataset,
                        "split_index": split_index,
                        "split_id": split.split_id,
                        "repetition_seed": int(split.repetition_seed),
                        "split_order": int(split.split_order),
                        "partition": partition,
                        "patch_count": int(len(frame)),
                        "patch_uid_sha256": _membership_sha256(frame, "patch_uid"),
                        "slide_count": int(frame["slide_id"].astype(str).nunique()),
                        "slide_id_sha256": _membership_sha256(frame, "slide_id"),
                    }
                )
    hashes = pd.DataFrame(rows)
    target = output_dir / "membership_hashes.csv"
    hashes.to_csv(target, index=False, lineterminator="\n")
    manifest = {
        "schema": "pathorob-official-apd-split-freeze-v1",
        "status": "complete",
        "upstream_repository": "https://github.com/bifold-pathomics/PathoROB",
        "upstream_commit": "6583cf0b0d902c8cc032308262fa3a3befdc0687",
        "pythonhashseed": 0,
        "repetition_seeds": list(REPETITION_SEEDS),
        "split_count": int(hashes[["dataset", "split_index"]].drop_duplicates().shape[0]),
        "partition_hash_rows": int(len(hashes)),
        "metadata_sha256": metadata_hashes,
        "tolkach_resource_sha256": _sha256(tolkach_resource),
        "split_implementation_sha256": _sha256(Path(__file__).with_name("splits.py")),
        "membership_hashes_sha256": _sha256(target),
        "note": (
            "Pinned upstream sequential resampling; PYTHONHASHSEED=0 freezes "
            "the upstream Tolkach set-iteration order."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def validate_reference(root: Path) -> None:
    """Validate an externally supplied, complete scientific contract tree."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"External contract directory does not exist: {root}")
    if not (root / "manifest.json").is_file():
        raise FileNotFoundError(f"External contract manifest is required: {root / 'manifest.json'}")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8-sig"))
    for item in manifest["files"]:
        path = root / item["path"]
        if not path.is_file() or path.stat().st_size != int(item["size_bytes"]):
            raise ValueError(f"missing or wrong-size public file: {item['path']}")
        if _sha256(path) != item["sha256"]:
            raise ValueError(f"SHA256 mismatch: {item['path']}")
        if path.suffix.lower() in {".csv", ".json", ".md", ".yaml", ".yml"}:
            text = path.read_text(encoding="utf-8-sig", errors="strict")
            if "/storage01/" in text or "yh_choi" in text or "C:\\Users\\" in text:
                raise ValueError(f"machine-local path leaked into {item['path']}")

    downstream = root / "pathorob" / "aggregation"
    for task_id in CLASSIFICATION_TASKS:
        task = downstream / task_id
        cohort = pd.read_csv(task / "cohort.csv")
        committed = pd.read_csv(task / "splits.csv")
        validate_patient_splits(cohort, committed)
        regenerated = build_classification_splits(cohort, task_id)
        pd.testing.assert_frame_equal(
            _canonical(regenerated), _canonical(committed), check_dtype=False
        )

    current = root / "tcga_brca"
    cohort = pd.read_csv(current / "cohort.csv")
    split = pd.read_csv(current / "split.csv")
    validate_patient_splits(cohort, split)
    if split["patient_id"].nunique() != 996:
        raise ValueError("TCGA-BRCA p2 exact split must contain 996 patients")

    pathorob = root / "pathorob"
    for dataset in ("camelyon", "tcga_4x4", "tolkach_esca"):
        metadata = prepare_apd_metadata(pd.read_csv(pathorob / f"{dataset}.csv"))
        resource = pathorob / "tolkach_splits.json" if dataset == "tolkach_esca" else None
        splits = build_official_apd_split_family(
            metadata, dataset, REPETITION_SEEDS, resource
        )
        total = len(splits)
        for split in splits:
            partitions = {
                "train": split.train,
                "validation": split.validation,
                "id_test": split.id_test,
                "ood_test": split.ood_test,
            }
            names = list(partitions)
            for left_index, left in enumerate(names):
                for right in names[left_index + 1 :]:
                    patch_overlap = set(partitions[left]["patch_uid"]) & set(
                        partitions[right]["patch_uid"]
                    )
                    if patch_overlap:
                        raise ValueError(
                            f"APD patch leakage in {dataset}/{split.split_id}: "
                            f"{left}/{right}"
                        )
                    slide_overlap = set(partitions[left]["slide_id"]) & set(
                        partitions[right]["slide_id"]
                    )
                    allowed_tolkach_overlap = (
                        dataset == "tolkach_esca"
                        and {left, right} == {"train", "validation"}
                    )
                    if slide_overlap and not allowed_tolkach_overlap:
                        raise ValueError(
                            f"unexpected APD slide overlap in "
                            f"{dataset}/{split.split_id}: {left}/{right}"
                        )
        expected = {"camelyon": 160, "tcga_4x4": 140, "tolkach_esca": 80}[dataset]
        if total != expected:
            raise ValueError(f"{dataset}: expected {expected} APD splits, got {total}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate an external scientific contract tree")
    validate.add_argument("--root", type=Path, required=True,
                          help="External metadata root containing manifest.json and task contracts")
    generate = subparsers.add_parser("generate-downstream", help="regenerate one split family")
    generate.add_argument("--task", choices=sorted(CLASSIFICATION_TASKS), required=True)
    generate.add_argument("--cohort", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    apd = subparsers.add_parser("materialize-apd", help="materialize one APD repetition")
    apd.add_argument("--dataset", choices=["camelyon", "tcga_4x4", "tolkach_esca"], required=True)
    apd.add_argument("--metadata", type=Path, required=True)
    apd.add_argument("--seed", type=int, default=REPETITION_SEEDS[0])
    apd.add_argument("--tolkach-resource", type=Path)
    apd.add_argument("--output-dir", type=Path, required=True)
    freeze_apd = subparsers.add_parser(
        "freeze-apd-family", help="freeze all pinned APD membership hashes"
    )
    freeze_apd.add_argument("--metadata-dir", type=Path, required=True)
    freeze_apd.add_argument("--tolkach-resource", type=Path, required=True)
    freeze_apd.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "validate":
        validate_reference(args.root)
        print("public-data contracts: OK")
    elif args.command == "generate-downstream":
        cohort = pd.read_csv(args.cohort)
        result = build_classification_splits(cohort, args.task)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False, lineterminator="\n")
        print(f"wrote {len(result)} rows to {args.output}")
    elif args.command == "materialize-apd":
        metadata = prepare_apd_metadata(pd.read_csv(args.metadata))
        splits = build_official_apd_splits(
            metadata,
            args.dataset,
            int(args.seed),
            args.tolkach_resource,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        columns = [
            "patch_uid", "slide_id", "biological_class", "medical_center",
            "subset", "row_position", "partition", "split_id",
        ]
        for split in splits:
            result = pd.concat(
                [split.train, split.validation, split.id_test, split.ood_test],
                ignore_index=True,
            )[columns]
            result.to_csv(
                args.output_dir / f"{split.split_id}.csv",
                index=False,
                lineterminator="\n",
            )
        print(f"wrote {len(splits)} APD splits to {args.output_dir}")
    else:
        freeze_apd_family(
            args.metadata_dir,
            args.output_dir,
            args.tolkach_resource,
        )
        print(f"froze complete APD family at {args.output_dir}")


if __name__ == "__main__":
    main()
