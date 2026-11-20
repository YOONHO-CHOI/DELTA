"""Deterministic public-data split contracts used by DELTA experiments.

This module contains no dataset download logic or operator estimation.
It makes the published patient and
PathoROB APD partitions reproducible from explicitly supplied compact metadata.
Experiment cohort and partition files are not bundled with the package.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


REPETITION_SEEDS = tuple(random.Random(1000).sample(range(10_000), 20))
RHO_LEVELS = (0.0, 0.25, 0.50, 0.75, 1.0)

CLASSIFICATION_TASKS = {
    "camelyon_slide_tumor": {
        "label_column": "patient_label",
        "preferred_by_center": {"RUMC": "tumor", "UMCU": "normal"},
        "train_per_center": 14,
    },
    "tcga_nsclc_subtype": {
        "label_column": "patient_label",
        "preferred_by_center": {
            "Asterand": "lusc",
            "Christiana Healthcare": "luad",
            "Roswell Park": "luad",
            "University of Pittsburgh": "lusc",
        },
        "train_per_center": 10,
    },
    "tcga4_four_class": {
        "label_column": "patient_label",
        "preferred_by_center": {
            "Asterand": "brca",
            "Christiana Healthcare": "coad",
            "Roswell Park": "luad",
            "University of Pittsburgh": "lusc",
        },
        "train_per_center": 8,
    },
}

APD_SPECS = {
    "camelyon": {
        "id_centers": ["RUMC", "UMCU"],
        "classes": ["normal", "tumor"],
        "n_splits": 8,
        "max_train_slides": 14,
        "patches_per_slide": 300,
    },
    "tcga_4x4": {
        "id_centers": [
            "Asterand",
            "Christiana Healthcare",
            "Roswell Park",
            "University of Pittsburgh",
        ],
        "classes": [
            "Breast_invasive_carcinoma",
            "Colon_adenocarcinoma",
            "Lung_adenocarcinoma",
            "Lung_squamous_cell_carcinoma",
        ],
        "n_splits": 7,
        "max_train_slides": 8,
        "patches_per_slide": 30,
    },
    "tolkach_esca": {
        "id_centers": ["VALSET2_WNS", "VALSET4_CHA_FULL"],
        "classes": ["TUMOR", "MUSC_PROP", "SH_OES", "SH_MAG", "REGR_TU", "ADVENT"],
        "n_splits": 4,
        "max_train_slides": 6,
        "patches_per_slide": 100,
    },
}


@dataclass(frozen=True)
class APDSplit:
    """One exact PathoROB patch-disjoint APD partition."""

    split_id: str
    train: pd.DataFrame
    validation: pd.DataFrame
    id_test: pd.DataFrame
    ood_test: pd.DataFrame
    repetition_seed: int
    split_order: int


def _patient_table(cohort: pd.DataFrame, label_column: str) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for patient_id, frame in cohort.groupby("patient_id", sort=True):
        record = {"patient_id": str(patient_id)}
        for column in ("medical_center", "source_subset", label_column):
            values = sorted(frame[column].dropna().astype(str).unique().tolist())
            if len(values) != 1:
                raise ValueError(f"non-unique patient field: {patient_id}/{column}/{values}")
            record[column] = values[0]
        rows.append(record)
    return pd.DataFrame(rows)


def _target_counts(
    labels: list[str], preferred: str, train_total: int, rho: float
) -> dict[str, int]:
    preferred_count = int(round(train_total * (1.0 + (len(labels) - 1) * rho) / len(labels)))
    preferred_count = min(train_total, max(0, preferred_count))
    result = {label: 0 for label in labels}
    result[preferred] = preferred_count
    other = [label for label in labels if label != preferred]
    for position in range(train_total - preferred_count):
        result[other[position % len(other)]] += 1
    return result


def build_classification_splits(
    cohort: pd.DataFrame,
    task_id: str,
) -> pd.DataFrame:
    """Recreate all 20 x 5 patient-atomic classification split manifests."""

    if task_id not in CLASSIFICATION_TASKS:
        raise ValueError(f"unknown public classification task: {task_id}")
    contract = CLASSIFICATION_TASKS[task_id]
    label_column = str(contract["label_column"])
    preferred_by_center = dict(contract["preferred_by_center"])
    patients = _patient_table(cohort, label_column)
    id_patients = patients[patients["source_subset"].eq("ID")].copy()
    ood_patients = patients[patients["source_subset"].eq("OOD")].copy()
    centers = sorted(preferred_by_center)
    labels = sorted(id_patients[label_column].unique().tolist())
    if set(id_patients["medical_center"]) != set(centers):
        raise ValueError("ID centers do not match the frozen task contract")

    cell_sizes = id_patients.groupby(["medical_center", label_column]).size().to_dict()
    maximum = min(
        cell_sizes[(center, preferred_by_center[center])] - 2 for center in centers
    )
    train_per_center = min(int(contract["train_per_center"]), int(maximum))
    rows: list[dict[str, object]] = []

    for repeat, seed in enumerate(REPETITION_SEEDS):
        shuffled: dict[tuple[str, str], list[str]] = {}
        cells = [(center, label) for center in centers for label in labels]
        for cell_index, (center, label) in enumerate(cells):
            values = id_patients[
                id_patients["medical_center"].eq(center)
                & id_patients[label_column].eq(label)
            ]["patient_id"].tolist()
            random.Random(int(seed) + 1009 * cell_index).shuffle(values)
            shuffled[(center, label)] = values

        for rho in RHO_LEVELS:
            condition = f"rho-{int(round(100 * rho)):03d}"
            split_id = f"{task_id}__rep-{repeat:02d}__{condition}"
            assignments: dict[str, str] = {}
            for center in centers:
                targets = _target_counts(
                    labels, preferred_by_center[center], train_per_center, rho
                )
                for label in labels:
                    values = shuffled[(center, label)]
                    validation = values[:1]
                    candidates = values[1:]
                    count = targets[label]
                    if count > len(candidates) - 1:
                        raise ValueError(f"insufficient patients in {center}/{label}")
                    for patient in candidates[:count]:
                        assignments[patient] = "train"
                    for patient in validation:
                        assignments[patient] = "validation"
                    for patient in candidates[count:]:
                        assignments[patient] = "id_test"
            for patient in ood_patients["patient_id"]:
                assignments[str(patient)] = "ood_test"
            if len(assignments) != len(patients):
                raise ValueError(f"incomplete patient coverage for {split_id}")

            lookup = patients.set_index("patient_id")
            for patient_id, partition in sorted(assignments.items()):
                record = lookup.loc[patient_id]
                rows.append(
                    {
                        "task_id": task_id,
                        "split_id": split_id,
                        "repeat": repeat,
                        "seed": int(seed),
                        "condition": condition,
                        "correlation_strength": float(rho),
                        "patient_id": patient_id,
                        "partition": partition,
                        "medical_center": record["medical_center"],
                        "stratification_label": record[label_column],
                        "event": np.nan,
                        "time_days": np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _split_survival_id(frame: pd.DataFrame, seed: int) -> dict[str, str]:
    rng = np.random.default_rng(int(seed))
    values = frame.copy()
    quartile = pd.qcut(
        values["time_days"].rank(method="first"), q=4, labels=False, duplicates="drop"
    )
    values["stratum"] = values["event"].astype(str) + "__q" + quartile.astype(str)
    assignments: dict[str, str] = {}
    for _, group in values.groupby("stratum", sort=True):
        patients = group["patient_id"].astype(str).to_numpy()
        rng.shuffle(patients)
        n = len(patients)
        n_train = max(1, int(round(0.70 * n)))
        n_validation = max(1, int(round(0.15 * n))) if n >= 3 else 0
        if n_train + n_validation >= n and n >= 3:
            n_train = n - n_validation - 1
        for patient in patients[:n_train]:
            assignments[str(patient)] = "train"
        for patient in patients[n_train : n_train + n_validation]:
            assignments[str(patient)] = "validation"
        for patient in patients[n_train + n_validation :]:
            assignments[str(patient)] = "id_test"
    return assignments


def build_survival_splits(
    cohort: pd.DataFrame, minimum_ood_center_size: int = 5
) -> pd.DataFrame:
    """Recreate the historical 20 x 2 TCGA-BRCA patient split family."""

    records: list[dict[str, object]] = []
    for patient_id, frame in cohort.groupby("patient_id", sort=True):
        record: dict[str, object] = {"patient_id": str(patient_id)}
        for column in ("medical_center", "event", "time_days"):
            values = sorted(frame[column].dropna().unique().tolist())
            if len(values) != 1:
                raise ValueError(f"non-unique survival field: {patient_id}/{column}")
            record[column] = values[0]
        records.append(record)
    patients = pd.DataFrame(records)
    patients["event"] = patients["event"].astype(int)
    patients["time_days"] = patients["time_days"].astype(float)
    center_counts = patients.groupby("medical_center").size().sort_values(ascending=False)
    top_six = center_counts.head(6).index.tolist()
    ood_centers = [
        center
        for center, count in center_counts.items()
        if center not in top_six and int(count) >= minimum_ood_center_size
    ]
    analysis = patients[patients["medical_center"].isin(top_six + ood_centers)].copy()
    rows: list[dict[str, object]] = []
    conditions = {"train-centers-3": top_six[:3], "train-centers-6": top_six}
    for repeat, seed in enumerate(REPETITION_SEEDS):
        for condition, centers in conditions.items():
            assignments = _split_survival_id(
                analysis[analysis["medical_center"].isin(centers)], int(seed)
            )
            for patient in analysis[analysis["medical_center"].isin(ood_centers)]["patient_id"]:
                assignments[str(patient)] = "ood_test"
            split_id = f"tcga_brca_overall_survival__rep-{repeat:02d}__{condition}"
            lookup = analysis.set_index("patient_id")
            for patient_id, partition in sorted(assignments.items()):
                record = lookup.loc[patient_id]
                rows.append(
                    {
                        "task_id": "tcga_brca_overall_survival",
                        "split_id": split_id,
                        "repeat": repeat,
                        "seed": int(seed),
                        "condition": condition,
                        "correlation_strength": np.nan,
                        "patient_id": patient_id,
                        "partition": partition,
                        "medical_center": record["medical_center"],
                        "stratification_label": "",
                        "event": int(record["event"]),
                        "time_days": float(record["time_days"]),
                    }
                )
    return pd.DataFrame(rows)


def _split_map(dataset: str, split: int, n: int) -> list[tuple[int, int, int]]:
    if dataset == "camelyon":
        return sorted(
            [
                (0, 0, (7 - split) * n),
                (0, 1, (7 + split) * n),
                (1, 0, (7 + split) * n),
                (1, 1, (7 - split) * n),
            ]
        )
    if dataset == "tcga_4x4":
        diagonal = [(i, j, (split + 2) * n) for i in range(4) for j in range(4) if i == j]
        inverse = [
            (i, j, (1 if split % 2 == 1 else (2 if split < 3 else 0)) * n)
            for i in range(4)
            for j in range(4)
            if i + j == 3
        ]
        rest = [
            (i, j, (2 if split < 2 else (1 if split < 5 else 0)) * n)
            for i in range(4)
            for j in range(4)
            if i != j and i + j != 3
        ]
        return sorted(diagonal + inverse + rest)
    if dataset == "tolkach_esca":
        return sorted(
            [(0, j, (3 - split) * n) for j in range(3)]
            + [(0, j, (3 + split) * n) for j in range(3, 6)]
            + [(1, j, (3 + split) * n) for j in range(3)]
            + [(1, j, (3 - split) * n) for j in range(3, 6)]
        )
    raise ValueError(f"unsupported APD dataset: {dataset}")


def _concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
    selected = [part for part in parts if len(part)]
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def prepare_apd_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic row and patch identifiers to compact PathoROB metadata."""

    required = {"subset", "slide_id", "patch_id", "biological_class", "medical_center"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"missing PathoROB metadata columns: {sorted(missing)}")
    frame = metadata.copy().reset_index(drop=True)
    frame["patch_uid"] = frame["slide_id"].astype(str) + "-" + frame["patch_id"].astype(str)
    frame["row_position"] = np.arange(len(frame), dtype=np.int64)
    if frame["patch_uid"].duplicated().any():
        raise ValueError("PathoROB patch_uid is not unique")
    return frame


def build_official_apd_splits(
    metadata: pd.DataFrame,
    dataset: str,
    repetition_seed: int,
    tolkach_split_resource: Path | None = None,
) -> list[APDSplit]:
    """Return one repetition from the pinned PathoROB APD split sequence.

    PathoROB constructs each centre/class list once, then shuffles the list in
    place for every repetition.  Consequently, repetition ``k`` depends on all
    earlier repetitions even though each repetition has its own random seed.
    For a seed in :data:`REPETITION_SEEDS`, this wrapper reconstructs the
    complete prefix and returns only the requested repetition.  An arbitrary
    non-canonical seed is treated as a one-repetition exploratory sequence.
    """

    try:
        stop = REPETITION_SEEDS.index(int(repetition_seed)) + 1
    except ValueError:
        seeds = (int(repetition_seed),)
    else:
        seeds = REPETITION_SEEDS[:stop]
    family = build_official_apd_split_family(
        metadata,
        dataset,
        seeds,
        tolkach_split_resource=tolkach_split_resource,
    )
    count = int(APD_SPECS[dataset]["n_splits"])
    return family[-count:]


def build_official_apd_split_family(
    metadata: pd.DataFrame,
    dataset: str,
    repetition_seeds: tuple[int, ...] = REPETITION_SEEDS,
    tolkach_split_resource: Path | None = None,
) -> list[APDSplit]:
    """Recreate the exact sequential APD resampling used by PathoROB.

    The implementation is pinned to upstream commit
    ``6583cf0b0d902c8cc032308262fa3a3befdc0687``.  In particular, the
    centre/class patch lists persist between repetitions, and Tolkach draws
    both centre-specific feasible-split indices before shuffling any class.
    """

    if dataset not in APD_SPECS:
        raise ValueError(f"unsupported APD dataset: {dataset}")
    required = {
        "patch_uid", "slide_id", "biological_class", "medical_center",
        "subset", "row_position",
    }
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"missing APD columns: {sorted(missing)}")
    spec = APD_SPECS[dataset]
    id_meta = metadata[metadata["subset"].astype(str).eq("ID")].copy()
    ood = metadata[metadata["subset"].astype(str).eq("OOD")].copy().reset_index(drop=True)
    feasible: Mapping[str, object] | None = None
    if dataset == "tolkach_esca":
        if tolkach_split_resource is None:
            raise ValueError("Tolkach requires the frozen train/test resource")
        feasible = json.loads(Path(tolkach_split_resource).read_text(encoding="utf-8"))

    grid: dict[tuple[int, int], pd.DataFrame] = {}
    for i, center in enumerate(spec["id_centers"]):
        for j, biological_class in enumerate(spec["classes"]):
            grid[(i, j)] = id_meta[
                id_meta["medical_center"].astype(str).eq(center)
                & id_meta["biological_class"].astype(str).eq(biological_class)
            ].copy().reset_index(drop=True)

    results: list[APDSplit] = []
    n = int(spec["patches_per_slide"])
    id_test_start = (int(spec["max_train_slides"]) + 1) * n
    max_train = int(spec["max_train_slides"])
    for repetition_seed in tuple(map(int, repetition_seeds)):
        rng = random.Random(repetition_seed)
        split_choices: list[int] | None = None
        if feasible is not None:
            split_choices = [
                rng.randint(0, len(feasible["train"][center]) - 1)
                for center in spec["id_centers"]
            ]

        for i, center in enumerate(spec["id_centers"]):
            test_cases: set[str] = set()
            if feasible is not None and split_choices is not None:
                test_cases = set(map(str, feasible["test"][center][split_choices[i]]))
            for j, _biological_class in enumerate(spec["classes"]):
                frame = grid[(i, j)]
                blocks = [
                    frame.iloc[start : start + n].copy()
                    for start in range(0, len(frame), n)
                ]
                rng.shuffle(blocks)
                if test_cases:
                    case_order = [str(block.iloc[0]["slide_id"]) for block in blocks]
                    # Keep the pinned upstream set construction exactly: while
                    # the relative order of moved test slides is immaterial to
                    # the current repetition, it becomes part of the mutable
                    # list presented to later repetitions.
                    for test_case in list(set(test_cases) & set(case_order)):
                        position = [
                            str(block.iloc[0]["slide_id"]) for block in blocks
                        ].index(test_case)
                        blocks.append(blocks.pop(position))
                grid[(i, j)] = _concat(blocks).reset_index(drop=True)

        for split_order in range(int(spec["n_splits"])):
            train: list[pd.DataFrame] = []
            validation: list[pd.DataFrame] = []
            id_test: list[pd.DataFrame] = []
            for i, j, count in _split_map(dataset, split_order, n):
                frame = grid[(i, j)]
                train.append(frame.iloc[:count])
                validation.append(frame.iloc[count : count + int(count / max_train)])
                id_test.append(frame.iloc[id_test_start:])
            split_id = f"rep-{repetition_seed:04d}__split-{split_order + 1:02d}"
            partitions = {
                "train": _concat(train),
                "validation": _concat(validation),
                "id_test": _concat(id_test),
                "ood_test": ood.copy(),
            }
            for partition, frame in partitions.items():
                frame["partition"] = partition
                frame["split_id"] = split_id
            results.append(
                APDSplit(
                    split_id=split_id,
                    train=partitions["train"],
                    validation=partitions["validation"],
                    id_test=partitions["id_test"],
                    ood_test=partitions["ood_test"],
                    repetition_seed=repetition_seed,
                    split_order=split_order,
                )
            )
    return results


def validate_patient_splits(cohort: pd.DataFrame, splits: pd.DataFrame) -> None:
    """Fail closed on patient overlap, incomplete partitions, or center leakage."""

    cohort_patients = set(cohort["patient_id"].astype(str))
    required = {"train", "validation", "id_test", "ood_test"}
    for split_id, frame in splits.groupby("split_id", sort=True):
        if frame["patient_id"].duplicated().any():
            raise ValueError(f"duplicate patient assignment in {split_id}")
        if set(frame["partition"]) != required:
            raise ValueError(f"incomplete partitions in {split_id}")
        if not set(frame["patient_id"].astype(str)).issubset(cohort_patients):
            raise ValueError(f"unknown patient in {split_id}")
        id_centers = set(
            frame[frame["partition"].isin(["train", "validation", "id_test"])][
                "medical_center"
            ].astype(str)
        )
        ood_centers = set(
            frame[frame["partition"].eq("ood_test")]["medical_center"].astype(str)
        )
        if id_centers & ood_centers:
            raise ValueError(f"ID/OOD center leakage in {split_id}")
