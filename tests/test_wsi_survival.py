"""Survival endpoints on supplied predictions, independent of DELTA fitting."""
from __future__ import annotations

import math
import numpy as np
import pytest

from experiments.wsi_survival import metrics


def test_harrell_concordance_retains_risk_direction_and_half_credit_for_ties():
    times = np.asarray([1., 2., 3.])
    events = np.ones(3, dtype=int)
    assert metrics.harrell_c(times, events, np.asarray([3., 2., 1.])) == 1.
    assert metrics.harrell_c(times, events, np.asarray([1., 2., 3.])) == 0.
    assert metrics.harrell_c(times, events, np.asarray([1., 1., 0.])) == pytest.approx(5 / 6)


def test_censoring_km_preserves_left_limits_at_censoring_times():
    km = metrics.censoring_km(np.asarray([1., 2., 3.]), np.asarray([0, 1, 1]))
    query = np.asarray([1., 1.5, 2., 2.5])
    np.testing.assert_allclose(km(query, left=True), [1., 2/3, 2/3, 2/3])
    np.testing.assert_allclose(km(query), [2/3] * 4)


def test_breslow_survival_predictions_are_batch_invariant():
    risks = np.log(np.asarray([1., 2.]))
    times, hazard = metrics.breslow_baseline(np.asarray([1., 2.]), np.asarray([1, 1]), risks)
    np.testing.assert_array_equal(times, [1., 2.])
    np.testing.assert_allclose(hazard, [1/3, 5/6], atol=1e-14)
    alone = metrics.survival_probability(times, hazard, np.asarray([0.]), np.asarray([1.]))
    together = metrics.survival_probability(times, hazard, np.log(np.asarray([1., 4.])), np.asarray([1.]))
    assert alone[0, 0] == pytest.approx(math.exp(-1/3), abs=1e-14)
    np.testing.assert_array_equal(alone[0], together[0])


def test_uno_and_time_dependent_auc_retain_training_reference_and_ties():
    train_times = np.asarray([1., 2., 3., 4., 5.])
    train_events = np.ones(5, dtype=int)
    times = np.asarray([1., 2., 3., 4.])
    events = np.ones(4, dtype=int)
    for risks, expected in [(np.asarray([4., 3., 2., 1.]), 1.),
                            (np.asarray([1., 2., 3., 4.]), 0.),
                            (np.zeros(4), .5)]:
        assert metrics.uno_c(train_times, train_events, times, events, risks) == expected
        assert metrics.time_dependent_auc(train_times, train_events, times, events, risks, 2.5) == expected


def test_ipcw_brier_matches_uncensored_hand_calculation():
    times = np.asarray([1., 2.])
    events = np.asarray([1, 1])
    predictions = np.asarray([[.2], [.8]])
    actual = metrics.brier_score_ipcw(times, events, times, events, predictions, np.asarray([1.5]))
    np.testing.assert_allclose(actual, [.04], rtol=0, atol=1e-14)


@pytest.mark.models
def test_cox_loss_matches_a_hand_calculated_two_event_example():
    import torch
    from experiments.wsi_survival import model
    value = model.cox_efron_loss(torch.log(torch.tensor([1., 2.])),
                                torch.tensor([1., 2.]), torch.tensor([1., 1.]))
    assert float(value) == pytest.approx(math.log(3) / 2, abs=1e-7)
    model.self_test()


@pytest.mark.models
def test_patch_aggregation_probability_metrics_on_supplied_predictions():
    from experiments.patch_aggregation import evaluation
    labels = np.asarray([0, 1, 0, 1])
    probabilities = np.asarray([[.8, .2], [.3, .7], [.65, .35], [.25, .75]])
    assert evaluation.metrics(labels, probabilities) == {
        "balanced_accuracy": 1., "macro_f1": 1., "ece_10bin": .27499999999999997,
        "auroc": 1., "auprc": 1., "brier": .07875,
    }
