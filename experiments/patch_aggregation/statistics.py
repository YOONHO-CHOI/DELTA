"""Frozen patch-aggregation inference contract (Supplementary Methods S14).

No data are generated here. Effects and intervals use unrounded paired values;
only the exact signed-rank hypothesis test rounds to twelve decimal places.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def exact_signed_rank_p(differences) -> float:
    values = np.asarray(differences, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Signed-rank differences must be finite")
    nonzero = np.round(values, 12)
    nonzero = nonzero[nonzero != 0]
    if not len(nonzero):
        return 1.0
    if len(nonzero) > 30:
        raise ValueError("Exact paired encoder test supports at most 30 units")
    weights = (rankdata(np.abs(nonzero), method="average") * 2).astype(int)
    total = int(weights.sum())
    observed = min(int(weights[nonzero > 0].sum()), int(weights[nonzero < 0].sum()))
    counts = np.zeros(total + 1, dtype=np.int64)
    counts[0] = 1
    for weight in weights:
        counts[weight:] += counts[:-weight].copy()
    return min(1.0, 2 * int(counts[:observed + 1].sum()) / 2 ** len(nonzero))


def holm_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("A complete Holm family requires finite P values in [0, 1]")
    order = np.argsort(values, kind="stable")
    result = np.empty_like(values)
    result[order] = np.minimum(1, np.maximum.accumulate(
        values[order] * (len(values) - np.arange(len(values)))))
    return result
