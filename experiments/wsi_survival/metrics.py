"""Patient-level survival endpoints evaluated against the training reference.

These functions consume supplied times, events, risks and survival predictions.
They perform no encoder inference, operator estimation or embedding correction.
"""
from __future__ import annotations

import math
import numpy as np

def censoring_km(train_time: np.ndarray, train_event: np.ndarray):
    order = np.argsort(train_time, kind="stable")
    times = np.asarray(train_time[order], dtype=np.float64)
    censored = 1 - np.asarray(train_event[order], dtype=np.int64)
    unique = np.unique(times)
    survival = 1.0
    after = []
    for value in unique:
        at_risk = int(np.count_nonzero(times >= value))
        count = int(censored[times == value].sum())
        if at_risk > 0:
            survival *= 1.0 - count / at_risk
        after.append(survival)
    unique = np.asarray(unique, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)

    def evaluate(values: np.ndarray | float, left: bool = False) -> np.ndarray:
        scalar = np.asarray(values).ndim == 0
        query = np.atleast_1d(np.asarray(values, dtype=np.float64))
        side = "left" if left else "right"
        indices = np.searchsorted(unique, query, side=side) - 1
        result = np.ones(query.shape, dtype=np.float64)
        valid = indices >= 0
        # searchsorted(left)-1 already selects the last time strictly before
        # the query. Use its AFTER value: using BEFORE would skip twice.
        result[valid] = after[indices[valid]]
        result = np.maximum(result, 1e-6)
        return result[0] if scalar else result

    return evaluate

def pairwise_concordance(
    time_values: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    weights: np.ndarray | None = None,
    tau: float | None = None,
) -> float:
    numerator = 0.0
    denominator = 0.0
    for index in np.flatnonzero(event.astype(bool)):
        if tau is not None and time_values[index] > tau:
            continue
        comparable = time_values > time_values[index]
        if not comparable.any():
            continue
        weight = 1.0 if weights is None else float(weights[index])
        differences = risk[index] - risk[comparable]
        numerator += weight * float(
            np.count_nonzero(differences > 0) + 0.5 * np.count_nonzero(differences == 0)
        )
        denominator += weight * int(np.count_nonzero(comparable))
    return float(numerator / denominator) if denominator > 0 else math.nan

def harrell_c(time_values: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    return pairwise_concordance(time_values, event, risk)

def uno_c(
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    test_risk: np.ndarray,
) -> float:
    km = censoring_km(train_time, train_event)
    tau = float(np.max(train_time[train_event.astype(bool)]))
    weights = 1.0 / np.square(km(test_time, left=True))
    return pairwise_concordance(test_time, test_event, test_risk, weights, tau)

def breslow_baseline(
    time_values: np.ndarray, event: np.ndarray, risk: np.ndarray,
    *, risk_reference: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Breslow hazard on the explicit, shared training-risk reference scale."""
    event_times = np.unique(time_values[event.astype(bool)])
    increments = []
    shifted = np.asarray(risk, dtype=np.float64) - float(risk_reference)
    exp_risk = np.exp(shifted)
    for current in event_times:
        deaths = int(np.count_nonzero((time_values == current) & event.astype(bool)))
        denominator = float(exp_risk[time_values >= current].sum())
        increments.append(deaths / max(denominator, 1e-12))
    return event_times.astype(np.float64), np.cumsum(increments)

def survival_probability(
    baseline_times: np.ndarray,
    baseline_hazard: np.ndarray,
    risk: np.ndarray,
    evaluation_times: np.ndarray,
    *, risk_reference: float = 0.0,
) -> np.ndarray:
    """Predict independently of test-batch composition; never recenter on test."""
    indices = np.searchsorted(baseline_times, evaluation_times, side="right") - 1
    hazard = np.zeros(len(evaluation_times), dtype=np.float64)
    valid = indices >= 0
    hazard[valid] = baseline_hazard[indices[valid]]
    shifted = np.asarray(risk, dtype=np.float64) - float(risk_reference)
    return np.exp(-np.exp(shifted[:, None]) * hazard[None, :])

def brier_score_ipcw(
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    survival: np.ndarray,
    evaluation_times: np.ndarray,
) -> np.ndarray:
    km = censoring_km(train_time, train_event)
    result = []
    for column, current in enumerate(evaluation_times):
        observed = (test_time > current).astype(np.float64)
        cases = (test_time <= current) & test_event.astype(bool)
        controls = test_time > current
        weights = np.zeros(len(test_time), dtype=np.float64)
        weights[cases] = 1.0 / km(test_time[cases], left=True)
        weights[controls] = 1.0 / float(km(float(current)))
        result.append(float(np.mean(weights * np.square(observed - survival[:, column]))))
    return np.asarray(result)

def time_dependent_auc(
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    risk: np.ndarray,
    current: float,
) -> float:
    cases = np.flatnonzero((test_time <= current) & test_event.astype(bool))
    controls = np.flatnonzero(test_time > current)
    if len(cases) == 0 or len(controls) == 0:
        return math.nan
    km = censoring_km(train_time, train_event)
    case_weights = 1.0 / km(test_time[cases], left=True)
    control_weight = 1.0 / float(km(float(current)))
    numerator = 0.0
    denominator = 0.0
    for case, weight in zip(cases, case_weights):
        differences = risk[case] - risk[controls]
        numerator += float(weight * control_weight) * float(
            np.count_nonzero(differences > 0) + 0.5 * np.count_nonzero(differences == 0)
        )
        denominator += float(weight * control_weight) * len(controls)
    return float(numerator / denominator) if denominator > 0 else math.nan
