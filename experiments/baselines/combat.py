"""PathoROB paper-condition InMoose ComBat split correction."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd


def load_inmoose_pycombat() -> Callable:
    """Load InMoose 0.7.8, optionally from an isolated dependency path."""

    configured = os.environ.get("DELTA_INMOOSE_PATH")
    if configured:
        dependency_path = Path(configured).expanduser().resolve()
        if not dependency_path.is_dir():
            raise FileNotFoundError(
                f"DELTA_INMOOSE_PATH is absent: {dependency_path}"
            )
        if str(dependency_path) not in sys.path:
            sys.path.insert(0, str(dependency_path))
    try:
        from inmoose.pycombat import pycombat_norm
    except ImportError as exc:
        raise ImportError(
            "InMoose 0.7.8 is required for ComBat. Install the 'benchmark' "
            "extra or set DELTA_INMOOSE_PATH."
        ) from exc

    return pycombat_norm


def combat_correct_official_split(
    features: np.ndarray,
    partitions: Mapping[str, pd.DataFrame],
    fit_patch_uids: set[str] | None = None,
    pycombat: Callable | None = None,
) -> np.ndarray:
    """Fit train ComBat and correct query partitions against its reference batch.

    This function preserves the existing PathoROB replication behavior.  New
    strict-inductive downstream studies use the separately named
    :mod:`experiments.baselines.combat_inductive` implementation.
    """

    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("ComBat features must be a finite two-dimensional array")
    required = {"patch_uid", "row_position", "medical_center"}
    for name, frame in partitions.items():
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"ComBat {name} columns are missing: {sorted(missing)}"
            )

    train = partitions["train"]
    if fit_patch_uids is None:
        fit = train
    else:
        fit = train[
            train["patch_uid"].astype(str).isin(set(map(str, fit_patch_uids)))
        ]
    if fit.empty or fit["medical_center"].nunique() < 2:
        raise ValueError("ComBat fitting requires at least two medical centers")

    combat = pycombat or load_inmoose_pycombat()
    fit_positions = fit["row_position"].to_numpy(dtype=np.int64)
    fit_values = values[fit_positions]
    # InMoose estimates one empirical-Bayes prior across features. A feature
    # that is exactly invariant in the fitting partition has undefined
    # standardized variance and can poison every corrected dimension with
    # NaN. Such a feature contains no batch information, so exclude it from
    # ComBat and carry it through unchanged.
    varying = np.ptp(fit_values, axis=0) > 0
    if not varying.any():
        raise ValueError("ComBat fitting features are all invariant")
    corrected_varying = np.asarray(
        combat(
            fit_values[:, varying].T,
            fit["medical_center"].astype(str).tolist(),
        ),
        dtype=np.float64,
    ).T
    if corrected_varying.shape != (len(fit), int(varying.sum())):
        raise ValueError("InMoose ComBat fitting output shape is invalid")
    if not np.isfinite(corrected_varying).all():
        raise ValueError("InMoose ComBat fitting output contains NaN or Inf")
    corrected_reference = fit_values.copy()
    corrected_reference[:, varying] = corrected_varying

    corrected = values.copy()
    corrected[fit_positions] = corrected_reference
    fit_position_set = set(map(int, fit_positions))
    for name in ("train", "validation", "id_test", "ood_test"):
        frame = partitions[name]
        query = (
            frame[~frame["row_position"].astype(int).isin(fit_position_set)]
            if name == "train"
            else frame
        )
        if query.empty:
            continue
        query_positions = query["row_position"].to_numpy(dtype=np.int64)
        joined = np.concatenate(
            [corrected_reference[:, varying], values[query_positions][:, varying]],
            axis=0,
        )
        batches = ["__corrected_train_reference__"] * len(corrected_reference)
        batches.extend(query["medical_center"].astype(str).tolist())
        adjusted = np.asarray(
            combat(
                joined.T,
                batches,
                ref_batch="__corrected_train_reference__",
            ),
            dtype=np.float64,
        ).T
        adjusted_query = adjusted[len(corrected_reference) :]
        if not np.isfinite(adjusted_query).all():
            raise ValueError("InMoose ComBat query output contains NaN or Inf")
        corrected[np.ix_(query_positions, np.flatnonzero(varying))] = adjusted_query
    if not np.isfinite(corrected).all():
        raise ValueError("ComBat output contains NaN or Inf")
    return corrected.astype(np.float32)
