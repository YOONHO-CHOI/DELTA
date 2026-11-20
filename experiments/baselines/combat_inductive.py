"""Strict-inductive ComBat utilities for the PathoROB downstream program."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from .combat import load_inmoose_pycombat


def _validate_inputs(
    features: np.ndarray,
    partitions: Mapping[str, pd.DataFrame],
) -> np.ndarray:
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
    return values


def _fit_subset(
    train: pd.DataFrame,
    fit_patch_uids: set[str] | None,
) -> pd.DataFrame:
    if fit_patch_uids is None:
        fit = train.copy()
    else:
        fit = train[
            train["patch_uid"].astype(str).isin(set(map(str, fit_patch_uids)))
        ].copy()
    if fit.empty or fit["medical_center"].nunique() < 2:
        raise ValueError("ComBat fitting requires at least two training centers")
    return fit


@dataclass(frozen=True)
class FrozenComBatAffine:
    """Per-training-center affine maps recovered from one train-only ComBat fit."""

    centers: tuple[str, ...]
    slopes: Mapping[str, np.ndarray]
    intercepts: Mapping[str, np.ndarray]
    fit_patch_uids: tuple[str, ...]


def _fit_inmoose_affine(
    raw_fit: np.ndarray,
    batch_labels: list[str],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Fit InMoose once and expose its exact per-batch affine maps.

    ComBat is feature-wise affine after its empirical-Bayes parameters have
    been fitted.  Recovering those maps with a second least-squares fit is
    numerically unstable for nearly invariant embedding dimensions.  This
    helper mirrors InMoose 0.7.8's public ``pycombat_norm`` execution path and
    derives the frozen maps directly from ``gamma_star`` and ``delta_star``.
    """

    # Ensure the configured bundled InMoose path has been added to sys.path.
    load_inmoose_pycombat()
    module = importlib.import_module("inmoose.pycombat.pycombat_norm")

    vci = module.make_design_matrix(
        np.asarray(raw_fit, dtype=np.float64).T,
        batch_labels,
        covar_mod=None,
        ref_batch=None,
        na_cov_action="raise",
    )
    dat = np.asarray(vci.counts, dtype=np.float64)
    design = np.asarray(vci.design, dtype=np.float64).T
    categories = tuple(map(str, vci.batch.categories))
    batches_ind = [vci.batch_composition[item] for item in vci.batch.categories]
    batch_sizes = [len(indices) for indices in batches_ind]
    n_sample = int(dat.shape[1])

    b_hat, grand_mean, var_pooled = module.calculate_mean_var(
        design,
        batches_ind,
        vci.ref_batch_idx,
        dat,
        batch_sizes,
        int(vci.n_batch),
        n_sample,
    )
    stand_mean = np.asarray(
        module.calculate_stand_mean(
            grand_mean,
            n_sample,
            design,
            int(vci.n_batch),
            b_hat,
        ),
        dtype=np.float64,
    )
    # No biological covariates are passed in this protocol, so the preserved
    # mean is one feature-wise vector and can be applied to future samples.
    if not np.allclose(
        stand_mean,
        stand_mean[:, [0]],
        rtol=1e-12,
        atol=1e-12,
        equal_nan=False,
    ):
        raise RuntimeError("train-only ComBat stand_mean is sample-dependent")

    s_data = module.standardise_data(dat, stand_mean, var_pooled, n_sample)
    gamma_star, delta_star, batch_design = module.fit_model(
        design,
        int(vci.n_batch),
        s_data,
        batches_ind,
        False,
        True,
        None,
        vci.ref_batch_idx,
    )
    corrected_fit = np.asarray(
        module.adjust_data(
            s_data,
            gamma_star,
            delta_star,
            batch_design,
            batch_sizes,
            var_pooled,
            stand_mean,
            n_sample,
            vci.ref_batch_idx,
            batches_ind,
            dat,
        ),
        dtype=np.float64,
    ).T

    pooled_scale = np.sqrt(np.asarray(var_pooled, dtype=np.float64))
    preserved_mean = stand_mean[:, 0]
    slopes: dict[str, np.ndarray] = {}
    intercepts: dict[str, np.ndarray] = {}
    for index, center in enumerate(categories):
        delta = np.asarray(delta_star[index], dtype=np.float64)
        gamma = np.asarray(gamma_star[index], dtype=np.float64)
        if (
            not np.isfinite(delta).all()
            or not np.isfinite(gamma).all()
            or np.any(delta <= 0)
        ):
            raise ValueError(f"invalid fitted ComBat parameters for {center!r}")
        slope = 1.0 / np.sqrt(delta)
        intercept = (
            preserved_mean * (1.0 - slope)
            - gamma * pooled_scale * slope
        )
        slopes[center] = slope
        intercepts[center] = intercept

    if not np.isfinite(corrected_fit).all():
        raise ValueError("InMoose ComBat fitting output contains NaN or Inf")
    return corrected_fit, slopes, intercepts


def fit_combat_inductive(
    features: np.ndarray,
    train: pd.DataFrame,
    fit_patch_uids: set[str] | None = None,
    pycombat: Callable | None = None,
) -> FrozenComBatAffine:
    """Fit ComBat once on train and freeze its feature-wise affine maps."""

    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("ComBat features must be finite and two-dimensional")
    fit = _fit_subset(train, fit_patch_uids)
    positions = fit["row_position"].to_numpy(dtype=np.int64)
    raw_fit = values[positions]
    batch_labels = fit["medical_center"].astype(str).tolist()
    exact_slopes: dict[str, np.ndarray] | None = None
    exact_intercepts: dict[str, np.ndarray] | None = None
    if pycombat is None:
        # Exactly invariant training dimensions contain no batch information
        # and make ComBat's standardized variance undefined.  Preserve them
        # unchanged, matching the established PathoROB ComBat implementation,
        # and fit the empirical-Bayes model only on varying dimensions.
        varying = np.ptp(raw_fit, axis=0) > 0
        if not varying.any():
            raise ValueError("ComBat fitting features are all invariant")
        corrected_varying, slopes_varying, intercepts_varying = _fit_inmoose_affine(
            raw_fit[:, varying],
            batch_labels,
        )
        corrected_fit = raw_fit.copy()
        corrected_fit[:, varying] = corrected_varying
        exact_slopes = {}
        exact_intercepts = {}
        for center in slopes_varying:
            slope = np.ones(raw_fit.shape[1], dtype=np.float64)
            intercept = np.zeros(raw_fit.shape[1], dtype=np.float64)
            slope[varying] = slopes_varying[center]
            intercept[varying] = intercepts_varying[center]
            exact_slopes[center] = slope
            exact_intercepts[center] = intercept
    else:
        corrected_fit = np.asarray(
            pycombat(raw_fit.T, batch_labels),
            dtype=np.float64,
        ).T
    if corrected_fit.shape != raw_fit.shape:
        raise ValueError("InMoose ComBat train output shape is invalid")

    slopes: dict[str, np.ndarray] = {}
    intercepts: dict[str, np.ndarray] = {}
    for center in sorted(fit["medical_center"].astype(str).unique()):
        mask = fit["medical_center"].astype(str).eq(center).to_numpy()
        x = raw_fit[mask]
        y = corrected_fit[mask]
        if exact_slopes is not None and exact_intercepts is not None:
            slope = exact_slopes[center]
            intercept = exact_intercepts[center]
        else:
            # Custom test backends do not expose fitted parameters.  Retain
            # the generic regression fallback for those injected callables.
            x_mean = x.mean(axis=0)
            y_mean = y.mean(axis=0)
            centered = x - x_mean
            denominator = np.sum(centered * centered, axis=0)
            numerator = np.sum(centered * (y - y_mean), axis=0)
            slope = np.divide(
                numerator,
                denominator,
                out=np.ones_like(numerator),
                where=denominator > 1e-12,
            )
            intercept = y_mean - slope * x_mean
        reconstructed = x * slope + intercept
        if not np.allclose(
            reconstructed, y, rtol=2e-5, atol=2e-5, equal_nan=False
        ):
            raise RuntimeError(
                f"ComBat mapping for center {center!r} is not feature-wise affine"
            )
        slopes[center] = slope.astype(np.float64)
        intercepts[center] = intercept.astype(np.float64)
    return FrozenComBatAffine(
        centers=tuple(sorted(slopes)),
        slopes=slopes,
        intercepts=intercepts,
        fit_patch_uids=tuple(sorted(fit["patch_uid"].astype(str))),
    )


def transform_combat_inductive(
    features: np.ndarray,
    frame: pd.DataFrame,
    model: FrozenComBatAffine,
    *,
    unseen_batch_policy: str = "error",
) -> np.ndarray:
    """Apply frozen train maps sample-by-sample without query-batch statistics."""

    if unseen_batch_policy not in {"error", "identity_explicit"}:
        raise ValueError("unseen_batch_policy must be error or identity_explicit")
    values = np.asarray(features, dtype=np.float64)
    positions = frame["row_position"].to_numpy(dtype=np.int64)
    output = np.asarray(values[positions], dtype=np.float64).copy()
    centers = frame["medical_center"].astype(str).to_numpy()
    unknown = sorted(set(centers).difference(model.centers))
    if unknown and unseen_batch_policy == "error":
        raise ValueError(
            "strict-inductive ComBat has no train-fitted parameters for "
            f"unseen centers: {unknown}"
        )
    for center in sorted(set(centers).intersection(model.centers)):
        mask = centers == center
        output[mask] = (
            output[mask] * model.slopes[center] + model.intercepts[center]
        )
    if not np.isfinite(output).all():
        raise ValueError("strict-inductive ComBat output contains NaN or Inf")
    return output.astype(np.float32)


def fit_transform_combat_inductive_split(
    features: np.ndarray,
    partitions: Mapping[str, pd.DataFrame],
    *,
    fit_patch_uids: set[str] | None = None,
    unseen_batch_policy: str = "error",
    pycombat: Callable | None = None,
) -> tuple[np.ndarray, FrozenComBatAffine]:
    """Strict train-only ComBat with no validation/test batch aggregation."""

    values = _validate_inputs(features, partitions)
    model = fit_combat_inductive(
        values,
        partitions["train"],
        fit_patch_uids=fit_patch_uids,
        pycombat=pycombat,
    )
    corrected = values.astype(np.float32, copy=True)
    for name in ("train", "validation", "id_test", "ood_test"):
        frame = partitions[name]
        positions = frame["row_position"].to_numpy(dtype=np.int64)
        corrected[positions] = transform_combat_inductive(
            values,
            frame,
            model,
            unseen_batch_policy=unseen_batch_policy,
        )
    return corrected, model
