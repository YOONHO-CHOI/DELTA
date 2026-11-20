"""Exact paired tissue-group statistics for the frozen encoder panel."""
import numpy as np
import math

TEST_DECIMALS = 12  # Fraction units; suppress floating-point pseudo-ties only.


def exact_signed_rank(differences: list[float]) -> tuple[float, float, int]:
    """Conditional exact two-sided Wilcoxon signed-rank, with tied average ranks.

    Sign enumeration is computed by dynamic programming. Exact zeros are
    excluded (Wilcox rule); no asymptotic normal approximation is used.
    """
    rounded = [round(float(x), TEST_DECIMALS) for x in differences]
    values = [x for x in rounded if x != 0.0]
    n = len(values)
    if not n:
        return 0.0, 1.0, 0
    order = sorted(range(n), key=lambda i: abs(values[i]))
    ranks2, start = [0] * n, 0
    while start < n:
        end = start + 1
        while end < n and abs(values[order[end]]) == abs(values[order[start]]):
            end += 1
        for j in range(start, end):
            ranks2[order[j]] = start + 1 + end
        start = end
    total = sum(ranks2)
    positive = sum(rank for rank, v in zip(ranks2, values) if v > 0)
    counts, upto = [0] * (total + 1), 0
    counts[0] = 1
    for rank in ranks2:
        for k in range(upto, -1, -1):
            counts[k + rank] += counts[k]
        upto += rank
    extreme = min(positive, total - positive)
    return extreme / 2, min(1.0, 2 * sum(counts[:extreme + 1]) / (2 ** n)), n


def holm_adjust(p_values):
    order, previous = sorted(range(len(p_values)), key=p_values.__getitem__), 0.0
    result = [0.0] * len(order)
    for position, i in enumerate(order):
        previous = max(previous, min(1.0, (len(order) - position) * p_values[i]))
        result[i] = previous
    return result


def fixed_panel_tissue_values(panel):
    """One value per tissue group after equally averaging the fixed encoders."""
    import numpy as np
    values = np.asarray(panel, dtype=np.float64)
    if values.shape != (20, 46) or not np.isfinite(values).all():
        raise ValueError("Expected a finite, aligned 20-encoder by 46-tissue panel")
    return values.mean(axis=0).tolist()


def tissue_bootstrap_description(values, indices):
    """Pointwise percentile interval from shared paired tissue-group indices."""
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    indices = np.asarray(indices)
    if (values.shape != (46,) or indices.ndim != 2 or indices.shape[1] != 46
            or not np.isfinite(values).all() or indices.min() < 0 or indices.max() >= 46):
        raise ValueError("Expected finite effects and paired indices for the 46 groups")
    return {"mean": float(values.mean()),
            "ci95": np.quantile(values[indices].mean(axis=1), [0.025, 0.975]).tolist()}

