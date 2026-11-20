from pathlib import Path

import numpy as np
import pandas as pd

from experiments.pathorob.evaluation import (
    average_performance_drop,
    evaluate_roi_split,
    evaluate_roi_prediction_inventory,
    export_official_features,
)
from experiments.common.splits import APDSplit, APD_SPECS


def test_official_feature_export_is_center_wise_and_aligned(tmp_path: Path):
    metadata = pd.DataFrame(
        {
            "slide_id": ["s1", "s2", "s3"],
            "patch_id": ["p1", "p2", "p3"],
            "medical_center": ["A", "A", "B"],
        }
    )
    features = np.arange(12, dtype=np.float32).reshape(3, 4)
    output = export_official_features(features, metadata, "toy", "camelyon", tmp_path)
    assert sorted(path.name for path in output.glob("*.npz")) == ["A.npz", "B.npz"]
    with np.load(output / "A.npz", allow_pickle=False) as payload:
        np.testing.assert_array_equal(payload["s1-p1"], features[0])
        np.testing.assert_array_equal(payload["s2-p2"], features[1])


def _partition(rows, labels):
    return pd.DataFrame(
        {
            "row_position": rows,
            "biological_class": labels,
            "patch_uid": [f"p{row}" for row in rows],
            "slide_id": [f"s{row}" for row in rows],
        }
    )


def test_roi_classifier_uses_frozen_train_validation_and_test_partitions():
    features = np.array(
        [[-3.0], [-2.0], [2.0], [3.0], [-2.5], [2.5], [-4.0], [4.0], [-5.0], [5.0]],
        dtype=np.float32,
    )
    split = APDSplit(
        split_id="toy",
        train=_partition([0, 1, 2, 3], [0, 0, 1, 1]),
        validation=_partition([4, 5], [0, 1]),
        id_test=_partition([6, 7], [0, 1]),
        ood_test=_partition([8, 9], [0, 1]),
        repetition_seed=7028,
        split_order=0,
    )
    result = evaluate_roi_split(features, split)
    assert result["id_balanced_accuracy"] == 1.0
    assert result["ood_balanced_accuracy"] == 1.0
    assert result["best_c"] > 0


def test_average_performance_drop_matches_public_formula():
    rows = []
    for split_order in range(APD_SPECS["camelyon"]["n_splits"]):
        rows.append(
            {
                "dataset": "camelyon",
                "method": "DELTA",
                "domain": "OOD",
                "repetition_seed": 7028,
                "split_order": split_order,
                "balanced_accuracy": 0.8 if split_order == 0 else 0.72,
            }
        )
    raw = pd.DataFrame(rows).assign(method='RAW',balanced_accuracy=0.9)
    result = average_performance_drop(pd.DataFrame(rows),reference_scores=raw)
    assert np.isclose(result.loc[0, "average_performance_drop"], -0.20)


def test_split_local_inventory_fails_closed_when_a_split_is_missing():
    metadata = pd.DataFrame(
        {
            "subset": ["ID"],
            "slide_id": ["s1"],
            "patch_id": ["p1"],
            "biological_class": ["normal"],
            "medical_center": ["RUMC"],
        }
    )
    try:
        evaluate_roi_prediction_inventory(
            {"wrong-split": Path("missing.npy")},
            metadata,
            dataset="camelyon",
            method="DELTA",
            repetitions=1,
        )
    except (ValueError, KeyError):
        pass
    else:
        raise AssertionError("split-local DELTA inventory must fail closed")
