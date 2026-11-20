"""Split-contract checks with generated metadata, never experiment cohorts."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.common.splits import (
    APD_SPECS, CLASSIFICATION_TASKS, REPETITION_SEEDS,
    build_classification_splits, build_official_apd_split_family,
    build_official_apd_splits, prepare_apd_metadata, validate_patient_splits,
)
from experiments.common.split_cli import validate_reference


REPOSITORY = Path(__file__).resolve().parents[1]


def _normalized(frame):
    columns = ["task_id", "split_id", "repeat", "seed", "condition",
               "correlation_strength", "patient_id", "partition", "medical_center",
               "stratification_label", "event", "time_days"]
    return frame[columns].sort_values(["split_id", "patient_id"]).reset_index(drop=True)


def synthetic_classification_cohort(task_id):
    contract = CLASSIFICATION_TASKS[task_id]
    labels = sorted(set(contract["preferred_by_center"].values()))
    rows = []
    for subset, centers, count in (
        ("ID", sorted(contract["preferred_by_center"]), 8),
        ("OOD", ["synthetic-external-center"], 4),
    ):
        for center_index, center in enumerate(centers):
            for label_index, label in enumerate(labels):
                for index in range(count):
                    patient = f"synthetic-{task_id}-{subset}-c{center_index}-y{label_index}-{index}"
                    for slide in range(2):
                        rows.append(dict(patient_id=patient, slide_id=f"{patient}-slide-{slide}",
                                         patient_label=label, label=label, label_code=label_index,
                                         medical_center=center, source_subset=subset,
                                         analysis_unit="patient"))
    return pd.DataFrame(rows)


@pytest.fixture
def apd_inputs(tmp_path, monkeypatch):
    """Shrink only patch counts; retain the actual split maps and seed sequence."""
    inputs = {}
    for dataset, original in list(APD_SPECS.items()):
        # Keep every validation partition nonempty under the pinned integer
        # split arithmetic (the four-class task needs at least four patches).
        patch_count = 4 if dataset == "tcga_4x4" else 2
        spec = dict(original, patches_per_slide=patch_count)
        monkeypatch.setitem(APD_SPECS, dataset, spec)
        rows = []
        slide_count = int(spec["max_train_slides"]) + 3
        for center_index, center in enumerate(spec["id_centers"]):
            for class_index, label in enumerate(spec["classes"]):
                for slide in range(slide_count):
                    suffix = f"s{slide}" if dataset == "tolkach_esca" else f"y{class_index}-s{slide}"
                    slide_id = f"synthetic-{dataset}-c{center_index}-{suffix}"
                    for patch in range(patch_count):
                        rows.append(dict(subset="ID", slide_id=slide_id,
                                         patch_id=f"y{class_index}-p{patch}",
                                         biological_class=label, medical_center=center))
        for class_index, label in enumerate(spec["classes"]):
            for patch in range(patch_count):
                rows.append(dict(subset="OOD", slide_id=f"synthetic-{dataset}-ood-y{class_index}",
                                 patch_id=f"p{patch}", biological_class=label,
                                 medical_center="synthetic-external-center"))
        metadata = prepare_apd_metadata(pd.DataFrame(rows))
        resource = None
        if dataset == "tolkach_esca":
            feasible = {"train": {}, "test": {}}
            for center_index, center in enumerate(spec["id_centers"]):
                slides = [f"synthetic-{dataset}-c{center_index}-s{slide}" for slide in range(slide_count)]
                tests = [slides[-2:], slides[:2]]
                feasible["test"][center] = tests
                feasible["train"][center] = [[slide for slide in slides if slide not in test] for test in tests]
            resource = tmp_path / "synthetic_tolkach_splits.json"
            resource.write_text(json.dumps(feasible), encoding="utf-8")
        inputs[dataset] = metadata, resource
    return inputs


@pytest.mark.parametrize("task_id", sorted(CLASSIFICATION_TASKS))
def test_synthetic_classification_splits_are_deterministic_and_patient_atomic(task_id):
    cohort = synthetic_classification_cohort(task_id)
    first = build_classification_splits(cohort, task_id)
    second = build_classification_splits(cohort.sample(frac=1, random_state=17), task_id)
    pd.testing.assert_frame_equal(_normalized(first), _normalized(second), check_dtype=False)
    validate_patient_splits(cohort, first)
    assert first.split_id.nunique() == 20 * 5
    assert not first.duplicated(["split_id", "patient_id"]).any()
    for _, split in first.groupby("split_id"):
        assert set(split.patient_id) == set(cohort.patient_id)
        assert set(split.partition) == {"train", "validation", "id_test", "ood_test"}
        expanded = cohort.merge(split[["patient_id", "partition"]], on="patient_id", validate="many_to_one")
        assert expanded.groupby("patient_id").partition.nunique().eq(1).all()


@pytest.fixture
def synthetic_patient_split():
    rows, split = [], []
    for part_index, partition in enumerate(("train", "validation", "id_test", "ood_test")):
        for index in range(4):
            patient = f"synthetic-patient-{part_index}-{index}"
            center = "synthetic-ood" if partition == "ood_test" else "synthetic-id"
            split.append(dict(split_id="synthetic-fixed", patient_id=patient,
                              partition=partition, medical_center=center))
            for slide in range(2):
                rows.append(dict(patient_id=patient, slide_id=f"{patient}-slide-{slide}"))
    return pd.DataFrame(rows), pd.DataFrame(split)


def test_patient_split_validation_accepts_multiple_slides_per_patient(synthetic_patient_split):
    cohort, split = synthetic_patient_split
    validate_patient_splits(cohort, split)
    assert len(cohort) == 2 * split.patient_id.nunique()


@pytest.mark.parametrize("problem", ["duplicate", "partition", "unknown", "center"])
def test_patient_split_validation_rejects_leakage_or_invalid_membership(synthetic_patient_split, problem):
    cohort, split = synthetic_patient_split
    if problem == "duplicate":
        split = pd.concat([split, split.iloc[[0]]], ignore_index=True)
    elif problem == "partition":
        split = split[split.partition.ne("validation")]
    elif problem == "unknown":
        split.loc[0, "patient_id"] = "synthetic-unknown"
    else:
        split.loc[split.partition.eq("ood_test"), "medical_center"] = "synthetic-id"
    with pytest.raises(ValueError):
        validate_patient_splits(cohort, split)


@pytest.fixture
def external_contract_root(tmp_path, apd_inputs):
    root = tmp_path / "external-contracts"
    for task_id in CLASSIFICATION_TASKS:
        folder = root / "pathorob" / "aggregation" / task_id
        folder.mkdir(parents=True)
        cohort = synthetic_classification_cohort(task_id)
        cohort.to_csv(folder / "cohort.csv", index=False)
        build_classification_splits(cohort, task_id).to_csv(folder / "splits.csv", index=False)
    # Canonical panel size is an algorithm contract. All identities here are invented.
    cohort_rows, split_rows = [], []
    for index in range(996):
        patient = f"synthetic-survival-{index:04d}"
        partition = "train" if index < 381 else "validation" if index < 465 else "id_test" if index < 545 else "ood_test"
        center = "synthetic-ood" if partition == "ood_test" else "synthetic-id"
        cohort_rows.append(dict(patient_id=patient, slide_id=patient + "-slide-0"))
        split_rows.append(dict(split_id="synthetic-fixed", patient_id=patient,
                               partition=partition, medical_center=center))
    folder = root / "tcga_brca"
    folder.mkdir()
    pd.DataFrame(cohort_rows).to_csv(folder / "cohort.csv", index=False)
    pd.DataFrame(split_rows).to_csv(folder / "split.csv", index=False)
    for dataset, (metadata, resource) in apd_inputs.items():
        metadata.to_csv(root / "pathorob" / f"{dataset}.csv", index=False)
        if resource is not None:
            (root / "pathorob" / "tolkach_splits.json").write_bytes(resource.read_bytes())
    target = root / "stain_reference"
    target.mkdir()
    parameters = {"target": {"mean_l": 45., "mean_a": 2., "mean_b": -3.,
                             "std_l": 8., "std_a": 4., "std_b": 5.}}
    (target / "target_lab.json").write_text(json.dumps(parameters), encoding="utf-8")
    files = [{"path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size,
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
             for path in sorted(root.rglob("*")) if path.is_file()]
    (root / "manifest.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    return root


def test_explicit_external_manifest_hashes_and_split_contracts_validate(external_contract_root):
    validate_reference(external_contract_root)


def test_external_manifest_rejects_modified_input(external_contract_root):
    path = external_contract_root / "stain_reference" / "target_lab.json"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="wrong-size"):
        validate_reference(external_contract_root)


def test_external_validation_requires_an_explicit_existing_root(tmp_path, monkeypatch, capsys):
    from experiments.common.split_cli import main
    monkeypatch.setattr("sys.argv", ["delta-public-splits", "validate"])
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == 2
    assert "--root" in capsys.readouterr().err
    with pytest.raises(FileNotFoundError, match="External contract directory"):
        validate_reference(tmp_path / "not-supplied")
    with pytest.raises(FileNotFoundError, match="manifest is required"):
        validate_reference(tmp_path)




@pytest.mark.parametrize("dataset", sorted(APD_SPECS))
def test_pathorob_first_repetition_matches_partition_contract(dataset, apd_inputs):
    metadata, resource = apd_inputs[dataset]
    splits = build_official_apd_splits(metadata, dataset, REPETITION_SEEDS[0], resource)
    assert len(splits) == APD_SPECS[dataset]["n_splits"]
    for split in splits:
        partitions = {name: getattr(split, name) for name in ("train", "validation", "id_test", "ood_test")}
        names = list(partitions)
        for left_index, left in enumerate(names):
            for right in names[left_index + 1:]:
                assert not set(partitions[left].patch_uid) & set(partitions[right].patch_uid)
                allowed = dataset == "tolkach_esca" and {left, right} == {"train", "validation"}
                if not allowed:
                    assert not set(partitions[left].slide_id) & set(partitions[right].slide_id)


@pytest.mark.parametrize("dataset", sorted(APD_SPECS))
def test_pathorob_all_repetitions_follow_sequential_upstream_contract(dataset, apd_inputs):
    metadata, resource = apd_inputs[dataset]
    family = build_official_apd_split_family(metadata, dataset, REPETITION_SEEDS, resource)
    count = int(APD_SPECS[dataset]["n_splits"])
    assert len(family) == count * len(REPETITION_SEEDS)
    for repeat, seed in enumerate(REPETITION_SEEDS):
        requested = build_official_apd_splits(metadata, dataset, seed, resource)
        expected = family[repeat * count:(repeat + 1) * count]
        assert [item.split_id for item in requested] == [item.split_id for item in expected]
        for left, right in zip(requested, expected):
            for partition in ("train", "validation", "id_test", "ood_test"):
                assert set(getattr(left, partition).patch_uid) == set(getattr(right, partition).patch_uid)


def test_pathorob_second_repetition_is_not_an_independent_fresh_shuffle(apd_inputs):
    metadata, _ = apd_inputs["camelyon"]
    sequential = build_official_apd_split_family(metadata, "camelyon", REPETITION_SEEDS[:2])[int(APD_SPECS["camelyon"]["n_splits"])]
    independent = build_official_apd_split_family(metadata, "camelyon", (REPETITION_SEEDS[1],))[0]
    assert set(sequential.train.patch_uid) != set(independent.train.patch_uid)
