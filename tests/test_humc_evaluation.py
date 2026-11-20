from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.humc_normative import scoring as humc


def test_locked_case_aggregation_retains_the_minimum_patch_count() -> None:
    scores = np.arange(19)
    assert humc.top_fraction_mean(scores) == np.mean(scores[-5:])
    with pytest.raises(ValueError, match="empty"):
        humc.top_fraction_mean([])


def test_normative_fold_excludes_cancer_and_heldout_cases_from_calibration(monkeypatch) -> None:
    rows = []
    for case, cohort, fold in [("a", "NT", 1), ("b", "NT", 1), ("c", "Cancer", 1), ("d", "NT", 0), ("e", "Cancer", 0)]:
        for patch in range(2):
            rows.append(dict(row_idx=len(rows), patch_uid=f"{case}-{patch}", patch_path="unused",
                             cohort=cohort, label=int(cohort == "Cancer"), case_identifier=case,
                             slide_id=case, fold=fold, x_px_l0=patch, y_px_l0=patch,
                             tissue_fraction=.9, is_pure=True, grand_qc_oof_score=.2))
    metadata = pd.DataFrame(rows)
    features = np.arange(len(rows) * 2, dtype=np.float32).reshape(-1, 2)
    calls = []

    def knn(query, reference, k, device, batch_size, query_cases=None, reference_cases=None):
        calls.append((query.copy(), reference.copy(), query_cases, reference_cases))
        return np.arange(len(query), dtype=np.float32) / 10

    monkeypatch.setattr(humc, "exact_knn_anomaly", knn)
    result = humc.score_normative_fold(metadata, features, "fixture", 0, "raw", device="cpu", neighbors=2)
    np.testing.assert_array_equal(calls[0][0], features[:4])
    np.testing.assert_array_equal(calls[0][1], features[:4])
    np.testing.assert_array_equal(calls[0][2], [0, 0, 1, 1])
    np.testing.assert_array_equal(calls[0][2], calls[0][3])
    np.testing.assert_array_equal(calls[1][0], features[6:])
    assert calls[1][2] is calls[1][3] is None
    assert result.training_nt_cases == 2
    assert result.training_nt_patches == 4
    assert set(result.case_scores.case_identifier) == {"d", "e"}
    assert result.threshold == np.quantile([np.mean(np.asarray([0, .1], dtype=np.float32).astype(np.float64)),
                                           np.mean(np.asarray([.2, .3], dtype=np.float32).astype(np.float64))], .95, method="linear")
    assert result.case_scores["prediction"].dtype == np.int8


def test_exact_knn_case_exclusion_and_insufficient_neighbors() -> None:
    pytest.importorskip("torch")
    vectors = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
    codes = np.asarray([0, 0, 1, 1])
    scores = humc.exact_knn_anomaly(vectors, vectors, 2, "cpu", 2, codes, codes)
    np.testing.assert_array_equal(scores, np.ones(4, dtype=np.float32))
    with pytest.raises(ValueError, match="fewer than k valid"):
        humc.exact_knn_anomaly(vectors, vectors, 3, "cpu", 2, codes, codes)

