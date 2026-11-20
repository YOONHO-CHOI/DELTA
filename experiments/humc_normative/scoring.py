"""Maintained HUMC normative evaluation and paired clinical inference.

The screening profile averages the top 1% with a minimum of five patches
and uses case-excluded non-tumor calibration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


SEED = 42

N_FOLDS = 5

K_NEIGHBORS = 20

TOP_FRACTION = 0.01

TOP_MINIMUM = 5

V13_ARMS = (
    "raw",
    "reinhard_raw",
    "delta_v1_3_p2",
    "reinhard_delta_v1_3_p2",
)

BACKBONES = (
    "ciga",
    "hipt",
    "retccl",
    "ctranspath",
    "kaiko_vitb8",
    "virchow",
    "beph",
    "uni",
    "phikonv2",
    "virchow2",
    "conch",
    "conchv15",
    "uni2h",
    "hoptimus0",
    "h0mini",
    "exaonepath25",
    "gpfm",
    "moozy_patch",
    "gigapath",
    "musk",
)

DEFAULT_BATCH_SIZES = {
    "ciga": 1024,
    "hipt": 256,
    "retccl": 256,
    "ctranspath": 128,
    "kaiko_vitb8": 128,
    "virchow": 32,
    "beph": 128,
    "uni": 128,
    "phikonv2": 128,
    "virchow2": 32,
    "conch": 128,
    "conchv15": 32,
    "uni2h": 32,
    "hoptimus0": 16,
    "h0mini": 256,
    "exaonepath25": 256,
    "gpfm": 128,
    "moozy_patch": 512,
    "gigapath": 128,
    "musk": 64,
}


def fold_training_indices(metadata: pd.DataFrame, fold: int) -> np.ndarray:
    return np.flatnonzero(
        (metadata["cohort"].eq("NT") & metadata["fold"].ne(int(fold))).to_numpy()
    )


@dataclass(frozen=True)
class NormativeFoldResult:
    patch_scores: pd.DataFrame
    slide_scores: pd.DataFrame
    case_scores: pd.DataFrame
    threshold: float
    training_nt_cases: int
    training_nt_patches: int


def score_normative_fold(
    metadata: pd.DataFrame,
    features: np.ndarray,
    backbone: str,
    fold: int,
    arm: str,
    *,
    device: str = "cuda",
    query_batch_size: int = 512,
    neighbors: int = K_NEIGHBORS,
) -> NormativeFoldResult:
    """Score one held-out fold; calibrate only on other-fold NT cases.

    Calibration excludes every patch from the query's own case, then applies
    the locked top-1%-minimum-five aggregate and linear 95th percentile. No
    file I/O, model fitting, feature transformation, or cancer calibration is
    performed here.
    """
    train_indices = fold_training_indices(metadata, fold)
    test_indices = np.flatnonzero(metadata["fold"].eq(fold).to_numpy())
    all_case_names = sorted(metadata["case_identifier"].astype(str).unique())
    case_to_code = {case: index for index, case in enumerate(all_case_names)}
    case_codes = metadata["case_identifier"].astype(str).map(case_to_code).to_numpy(dtype=np.int64)
    reference = np.asarray(features[train_indices], dtype=np.float32)
    train_scores = exact_knn_anomaly(
        reference, reference, neighbors, device, query_batch_size,
        case_codes[train_indices], case_codes[train_indices],
    )
    training = metadata.iloc[train_indices][["case_identifier"]].copy()
    training["patch_anomaly_score"] = train_scores
    training_cases = aggregate_scores(training, ["case_identifier"])
    threshold = float(np.quantile(training_cases["anomaly_score"], 0.95, method="linear"))
    test_scores = exact_knn_anomaly(
        np.asarray(features[test_indices], dtype=np.float32),
        reference, neighbors, device, query_batch_size,
    )
    columns = [
        "row_idx", "patch_uid", "patch_path", "cohort", "label", "case_identifier",
        "slide_id", "fold", "x_px_l0", "y_px_l0", "tissue_fraction", "is_pure",
        "grand_qc_oof_score",
    ]
    scored = metadata.iloc[test_indices][columns].copy()
    scored["patch_anomaly_score"] = test_scores
    scored["backbone"] = backbone
    scored["arm"] = arm
    slide = aggregate_scores(scored, ["cohort", "label", "case_identifier", "slide_id", "fold"])
    slide["backbone"] = backbone
    slide["arm"] = arm
    case = aggregate_scores(scored, ["cohort", "label", "case_identifier", "fold"])
    case["threshold"] = threshold
    case["prediction"] = case["anomaly_score"].ge(threshold).astype(np.int8)
    case["backbone"] = backbone
    case["arm"] = arm
    return NormativeFoldResult(
        scored, slide, case, threshold,
        int(training_cases["case_identifier"].nunique()), int(len(train_indices)),
    )


def exact_binomial_interval(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Clopper-Pearson interval for a binomial proportion."""

    from scipy.stats import beta

    successes = int(successes)
    total = int(total)
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("invalid binomial counts")
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    return lower, upper


def clinical_operating_points(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """Return the two extreme screening operating points for higher-is-abnormal scores."""

    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError("clinical operating points require finite scores and both classes")
    positive = scores[labels == 1]
    negative = scores[labels == 0]

    fn0_threshold = float(np.min(positive))
    fn0_prediction = scores >= fn0_threshold
    fp0_threshold = float(np.nextafter(np.max(negative), np.inf))
    fp0_prediction = scores >= fp0_threshold

    def counts(prediction: np.ndarray) -> dict[str, int]:
        return {
            "tp": int(np.sum(prediction & (labels == 1))),
            "fn": int(np.sum(~prediction & (labels == 1))),
            "tn": int(np.sum(~prediction & (labels == 0))),
            "fp": int(np.sum(prediction & (labels == 0))),
        }

    fn0 = counts(fn0_prediction)
    fp0 = counts(fp0_prediction)
    fn0_sens_ci = exact_binomial_interval(fn0["tp"], fn0["tp"] + fn0["fn"])
    fn0_spec_ci = exact_binomial_interval(fn0["tn"], fn0["tn"] + fn0["fp"])
    fp0_sens_ci = exact_binomial_interval(fp0["tp"], fp0["tp"] + fp0["fn"])
    fp0_spec_ci = exact_binomial_interval(fp0["tn"], fp0["tn"] + fp0["fp"])
    return {
        "fn0_threshold": fn0_threshold,
        "fn0_prediction": fn0_prediction,
        "fn0_tp": fn0["tp"],
        "fn0_fn": fn0["fn"],
        "fn0_tn": fn0["tn"],
        "fn0_fp": fn0["fp"],
        "fn0_sensitivity": fn0["tp"] / (fn0["tp"] + fn0["fn"]),
        "fn0_sensitivity_ci_lower": fn0_sens_ci[0],
        "fn0_sensitivity_ci_upper": fn0_sens_ci[1],
        "fn0_specificity": fn0["tn"] / (fn0["tn"] + fn0["fp"]),
        "fn0_specificity_exact_ci_lower": fn0_spec_ci[0],
        "fn0_specificity_exact_ci_upper": fn0_spec_ci[1],
        "fn0_screen_negative_fraction": fn0["tn"] / len(labels),
        "fp0_threshold": fp0_threshold,
        "fp0_prediction": fp0_prediction,
        "fp0_tp": fp0["tp"],
        "fp0_fn": fp0["fn"],
        "fp0_tn": fp0["tn"],
        "fp0_fp": fp0["fp"],
        "fp0_sensitivity": fp0["tp"] / (fp0["tp"] + fp0["fn"]),
        "fp0_sensitivity_ci_lower": fp0_sens_ci[0],
        "fp0_sensitivity_ci_upper": fp0_sens_ci[1],
        "fp0_specificity": fp0["tn"] / (fp0["tn"] + fp0["fp"]),
        "fp0_specificity_ci_lower": fp0_spec_ci[0],
        "fp0_specificity_ci_upper": fp0_spec_ci[1],
    }


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm family-wise-error adjusted p-values."""

    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * float(values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted


def conditional_mcnemar_power(discordant_a_only: int, discordant_b_only: int, alpha: float = 0.05) -> float:
    """Descriptive post-hoc conditional power for the exact McNemar test."""

    from scipy.stats import binom, binomtest

    total = int(discordant_a_only + discordant_b_only)
    if total == 0:
        return float("nan")
    alternative_probability = float(discordant_b_only / total)
    rejection = [
        k for k in range(total + 1)
        if binomtest(k, total, 0.5, alternative="two-sided").pvalue < alpha
    ]
    return float(np.sum(binom.pmf(rejection, total, alternative_probability)))


def clinical_bootstrap(
    labels: np.ndarray,
    scores_by_arm: dict[str, np.ndarray],
    repetitions: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared stratified case bootstrap for FN0 specificity and AUROC."""

    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels, dtype=np.int8)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    arms = list(V13_ARMS)
    fn0_specificity = np.empty((int(repetitions), len(arms)), dtype=np.float64)
    auroc = np.empty_like(fn0_specificity)
    rng = np.random.default_rng(seed)
    for repetition in range(int(repetitions)):
        sampled_positive = rng.choice(positive, size=len(positive), replace=True)
        sampled_negative = rng.choice(negative, size=len(negative), replace=True)
        sampled = np.concatenate([sampled_negative, sampled_positive])
        sampled_labels = labels[sampled]
        for arm_index, arm in enumerate(arms):
            sampled_scores = scores_by_arm[arm][sampled]
            threshold = float(np.min(sampled_scores[sampled_labels == 1]))
            fn0_specificity[repetition, arm_index] = float(
                np.mean(sampled_scores[sampled_labels == 0] < threshold)
            )
            auroc[repetition, arm_index] = roc_auc_score(sampled_labels, sampled_scores)
    return fn0_specificity, auroc


def bootstrap_two_sided_p(differences: np.ndarray) -> float:
    """Add-one two-sided bootstrap sign p-value."""

    differences = np.asarray(differences, dtype=np.float64)
    repetitions = len(differences)
    lower = (np.sum(differences <= 0) + 1) / (repetitions + 1)
    upper = (np.sum(differences >= 0) + 1) / (repetitions + 1)
    return float(min(1.0, 2 * min(lower, upper)))


def bootstrap_backbone_mean(values: np.ndarray, repetitions: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("backbone bootstrap requires at least two finite values")
    rng = np.random.default_rng(seed)
    means = np.empty(int(repetitions), dtype=np.float64)
    for start in range(0, int(repetitions), 4096):
        stop = min(start + 4096, int(repetitions))
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def hierarchical_four_arm_inference(
    summary_long: pd.DataFrame,
    case_by_key: dict[tuple[str, str], pd.DataFrame],
    backbones: Sequence[str],
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Infer the five requested direct paired contrasts among the four locked arms.

    The same stratified HUMC case weights are applied to every backbone and arm
    in each replicate, preserving their paired dependence. Backbones are then
    resampled as a second level so intervals reflect both case and backbone-panel
    variability. No composite DELTA main effect is formed: every direct arm pair
    is retained separately. Primary significance uses the centered hierarchical
    paired-bootstrap null with Holm correction across the five comparisons within
    each metric. A global 20-test Holm value and backbone directional-consistency
    tests are retained as secondary audit fields.
    """

    from scipy.stats import binomtest

    metric_names = ("auroc", "auprc", "sensitivity", "specificity")
    arm_count = len(V13_ARMS)
    backbone_count = len(backbones)
    repetitions = int(repetitions)
    if repetitions < 200:
        raise ValueError("hierarchical inference requires at least 200 bootstrap repetitions")

    reference = case_by_key[(backbones[0], V13_ARMS[0])].sort_values("case_identifier")
    labels = reference["label"].to_numpy(dtype=np.int8)
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    if (len(negative), len(positive)) != (100, 32):
        raise ValueError("hierarchical inference expects 100 NT and 32 Cancer cases")

    scores = np.empty((backbone_count, arm_count, len(labels)), dtype=np.float64)
    predictions = np.empty_like(scores, dtype=np.int8)
    for backbone_index, backbone in enumerate(backbones):
        for arm_index, arm in enumerate(V13_ARMS):
            frame = case_by_key[(backbone, arm)].sort_values("case_identifier").reset_index(drop=True)
            pd.testing.assert_frame_equal(
                reference[["case_identifier", "label", "fold"]].reset_index(drop=True),
                frame[["case_identifier", "label", "fold"]].reset_index(drop=True),
                check_dtype=False,
                obj=f"hierarchical-inference:{backbone}:{arm}",
            )
            scores[backbone_index, arm_index] = frame["anomaly_score"].to_numpy(dtype=np.float64)
            predictions[backbone_index, arm_index] = frame["prediction"].to_numpy(dtype=np.int8)

    rng = np.random.default_rng(seed)
    weights = np.zeros((repetitions, len(labels)), dtype=np.int16)
    weights[:, negative] = rng.multinomial(
        len(negative), np.full(len(negative), 1.0 / len(negative)), size=repetitions
    )
    weights[:, positive] = rng.multinomial(
        len(positive), np.full(len(positive), 1.0 / len(positive)), size=repetitions
    )
    bootstrap_metrics = np.empty(
        (repetitions, backbone_count, arm_count, len(metric_names)), dtype=np.float64
    )

    # Vectorized weighted AUROC/AUPRC exactly handles any tied score groups.
    for backbone_index in range(backbone_count):
        for arm_index in range(arm_count):
            arm_scores = scores[backbone_index, arm_index]
            ascending = np.argsort(arm_scores, kind="mergesort")
            ascending_scores = arm_scores[ascending]
            starts = np.r_[0, np.flatnonzero(np.diff(ascending_scores) != 0) + 1]
            ordered_weights = weights[:, ascending]
            ordered_labels = labels[ascending]
            positive_group = np.add.reduceat(
                ordered_weights * ordered_labels[None, :], starts, axis=1
            )
            negative_group = np.add.reduceat(
                ordered_weights * (1 - ordered_labels)[None, :], starts, axis=1
            )
            negative_before = np.cumsum(negative_group, axis=1) - negative_group
            bootstrap_metrics[:, backbone_index, arm_index, 0] = np.sum(
                positive_group * (negative_before + 0.5 * negative_group), axis=1
            ) / float(len(positive) * len(negative))

            descending = ascending[::-1]
            descending_scores = arm_scores[descending]
            starts = np.r_[0, np.flatnonzero(np.diff(descending_scores) != 0) + 1]
            ordered_weights = weights[:, descending]
            ordered_labels = labels[descending]
            positive_group = np.add.reduceat(
                ordered_weights * ordered_labels[None, :], starts, axis=1
            )
            total_group = np.add.reduceat(ordered_weights, starts, axis=1)
            cumulative_positive = np.cumsum(positive_group, axis=1)
            cumulative_total = np.cumsum(total_group, axis=1)
            precision = np.divide(
                cumulative_positive,
                cumulative_total,
                out=np.zeros_like(cumulative_positive, dtype=np.float64),
                where=cumulative_total > 0,
            )
            bootstrap_metrics[:, backbone_index, arm_index, 1] = np.sum(
                positive_group * precision, axis=1
            ) / float(len(positive))

            arm_prediction = predictions[backbone_index, arm_index]
            bootstrap_metrics[:, backbone_index, arm_index, 2] = (
                weights[:, positive] @ arm_prediction[positive]
            ) / float(len(positive))
            bootstrap_metrics[:, backbone_index, arm_index, 3] = (
                weights[:, negative] @ (1 - arm_prediction[negative])
            ) / float(len(negative))

    comparisons = (
        ("delta_minus_raw", "raw", "delta_v1_3_p2", "DELTA - Raw"),
        ("delta_minus_reinhard_raw", "reinhard_raw", "delta_v1_3_p2", "DELTA - Reinhard"),
        ("reinhard_delta_minus_raw", "raw", "reinhard_delta_v1_3_p2", "Reinhard-DELTA - Raw"),
        (
            "reinhard_delta_minus_reinhard_raw",
            "reinhard_raw",
            "reinhard_delta_v1_3_p2",
            "Reinhard-DELTA - Reinhard",
        ),
        (
            "reinhard_delta_minus_delta",
            "delta_v1_3_p2",
            "reinhard_delta_v1_3_p2",
            "Reinhard-DELTA - DELTA",
        ),
    )
    arm_index_by_name = {arm: index for index, arm in enumerate(V13_ARMS)}
    backbone_draws = rng.integers(0, backbone_count, size=(repetitions, backbone_count))
    arm_rows = []
    effect_rows = []
    per_backbone_rows = []
    for metric_index, metric in enumerate(metric_names):
        point = (
            summary_long.pivot(index="backbone", columns="arm", values=metric)
            .loc[list(backbones), list(V13_ARMS)]
            .to_numpy(dtype=np.float64)
        )
        for arm_index, arm in enumerate(V13_ARMS):
            sampled = np.take_along_axis(
                bootstrap_metrics[:, :, arm_index, metric_index], backbone_draws, axis=1
            ).mean(axis=1)
            lower, upper = np.quantile(sampled, [0.025, 0.975])
            arm_rows.append(
                {
                    "metric": metric,
                    "arm": arm,
                    "mean": float(point[:, arm_index].mean()),
                    "ci95_lower": float(lower),
                    "ci95_upper": float(upper),
                    "n_backbones": backbone_count,
                    "n_cases": len(labels),
                    "bootstrap_repetitions": repetitions,
                }
            )

        for comparison_index, (comparison, arm_a, arm_b, display_label) in enumerate(comparisons):
            index_a = arm_index_by_name[arm_a]
            index_b = arm_index_by_name[arm_b]
            backbone_effect = point[:, index_b] - point[:, index_a]
            boot_effect_by_backbone = (
                bootstrap_metrics[:, :, index_b, metric_index]
                - bootstrap_metrics[:, :, index_a, metric_index]
            )
            sampled_effect = np.take_along_axis(
                boot_effect_by_backbone, backbone_draws, axis=1
            ).mean(axis=1)
            lower, upper = np.quantile(sampled_effect, [0.025, 0.975])
            for backbone, value in zip(backbones, backbone_effect):
                per_backbone_rows.append(
                    {
                        "backbone": backbone,
                        "metric": metric,
                        "comparison": comparison,
                        "arm_a": arm_a,
                        "arm_b": arm_b,
                        "comparison_label": display_label,
                        "effect_b_minus_a": float(value),
                    }
                )

            nonzero = backbone_effect[~np.isclose(backbone_effect, 0.0, atol=1e-12)]
            improved = int(np.sum(backbone_effect > 1e-12))
            worsened = int(np.sum(backbone_effect < -1e-12))
            sign_p = (
                1.0
                if len(nonzero) == 0
                else float(binomtest(improved, len(nonzero), 0.5, alternative="two-sided").pvalue)
            )
            observed = float(backbone_effect.mean())
            centered_bootstrap = sampled_effect - float(sampled_effect.mean())
            hierarchical_p = float(
                (np.sum(np.abs(centered_bootstrap) >= abs(observed) - 1e-15) + 1)
                / (len(centered_bootstrap) + 1)
            )
            permutation_repetitions = max(100_000, min(500_000, repetitions * 10))
            extreme = 0
            generated = 0
            permutation_rng = np.random.default_rng(
                seed + 10_000 + metric_index * len(comparisons) + comparison_index
            )
            while generated < permutation_repetitions:
                chunk = min(20_000, permutation_repetitions - generated)
                signs = permutation_rng.choice((-1.0, 1.0), size=(chunk, backbone_count))
                randomized = np.mean(signs * backbone_effect[None, :], axis=1)
                extreme += int(np.sum(np.abs(randomized) >= abs(observed) - 1e-15))
                generated += chunk
            effect_rows.append(
                {
                    "metric": metric,
                    "comparison": comparison,
                    "arm_a": arm_a,
                    "arm_b": arm_b,
                    "comparison_label": display_label,
                    "effect_b_minus_a": observed,
                    "ci95_lower": float(lower),
                    "ci95_upper": float(upper),
                    "improved_backbones": improved,
                    "worsened_backbones": worsened,
                    "tied_backbones": int(backbone_count - improved - worsened),
                    "consistency_fraction": float(improved / backbone_count),
                    "p_sign_exact": sign_p,
                    "p_hierarchical_bootstrap": hierarchical_p,
                    "p_sign_flip": float((extreme + 1) / (permutation_repetitions + 1)),
                    "permutation_repetitions": permutation_repetitions,
                    "bootstrap_repetitions": repetitions,
                    "inference_unit": "paired HUMC cases and resampled backbone panel",
                    "contrast": f"{arm_b} - {arm_a}",
                }
            )

    effects = pd.DataFrame(effect_rows)
    effects["p_holm_within_metric"] = effects.groupby("metric")["p_hierarchical_bootstrap"].transform(
        lambda values: holm_adjust(values.to_numpy(dtype=np.float64))
    )
    effects["p_sign_exact_holm_within_metric"] = effects.groupby("metric")["p_sign_exact"].transform(
        lambda values: holm_adjust(values.to_numpy(dtype=np.float64))
    )
    effects["p_holm_global_24"] = holm_adjust(effects["p_hierarchical_bootstrap"])
    effects["significant_holm_within_metric_0_05"] = effects["p_holm_within_metric"] < 0.05
    return pd.DataFrame(arm_rows), effects, pd.DataFrame(per_backbone_rows)


def l2_normalize(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float32)
    norms = np.linalg.norm(frame, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("zero or non-finite embedding norm")
    return frame / norms


def exact_knn_anomaly(
    query: np.ndarray,
    reference: np.ndarray,
    k: int,
    device: str,
    query_batch_size: int,
    query_case_codes: np.ndarray | None = None,
    reference_case_codes: np.ndarray | None = None,
    *,
    check=None,
    progress=None,
) -> np.ndarray:
    """Exact cosine anomaly with optional per-batch check and progress callbacks.

    progress(completed_query_rows, total_query_rows, reference_rows) runs after
    each batch; callbacks never change normalization, masking or scores.
    """
    import torch

    if len(reference) < k:
        raise ValueError(f"reference bank has {len(reference)} rows, fewer than k={k}")
    reference_np = l2_normalize(reference)
    query_np = l2_normalize(query)
    ref_tensor = torch.from_numpy(reference_np).to(device=device, dtype=torch.float32)
    ref_cases = None
    if reference_case_codes is not None:
        ref_cases = torch.from_numpy(np.asarray(reference_case_codes, dtype=np.int64)).to(device)
    output = np.empty(len(query_np), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(query_np), query_batch_size):
            if check is not None:
                check()
            stop = min(start + query_batch_size, len(query_np))
            batch = torch.from_numpy(query_np[start:stop]).to(device=device, dtype=torch.float32)
            similarities = batch @ ref_tensor.T
            if query_case_codes is not None:
                if ref_cases is None:
                    raise ValueError("reference case codes are required for LOO masking")
                batch_cases = torch.from_numpy(
                    np.asarray(query_case_codes[start:stop], dtype=np.int64)
                ).to(device)
                similarities.masked_fill_(batch_cases[:, None].eq(ref_cases[None, :]), -torch.inf)
            nearest = torch.topk(similarities, k=k, dim=1, largest=True, sorted=False).values
            if not torch.isfinite(nearest).all():
                raise ValueError("LOO masking left fewer than k valid neighbors")
            output[start:stop] = (1.0 - nearest.mean(dim=1)).cpu().numpy()
            del batch, similarities, nearest
            if progress is not None:
                progress(stop, len(query_np), len(reference_np))
    del ref_tensor
    if not np.isfinite(output).all():
        raise ValueError("non-finite anomaly scores")
    return output


def top_fraction_mean(values: Sequence[float]) -> float:
    scores = np.asarray(values, dtype=np.float64)
    if len(scores) == 0:
        raise ValueError("cannot aggregate an empty score set")
    count = min(len(scores), max(TOP_MINIMUM, int(math.ceil(TOP_FRACTION * len(scores)))))
    partition = np.partition(scores, len(scores) - count)
    return float(np.mean(partition[-count:]))


def aggregate_scores(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    records = []
    for keys, group in frame.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_columns, keys))
        record["anomaly_score"] = top_fraction_mean(group["patch_anomaly_score"])
        record["n_patches"] = int(len(group))
        record["top_patch_count"] = min(
            len(group), max(TOP_MINIMUM, int(math.ceil(TOP_FRACTION * len(group))))
        )
        records.append(record)
    return pd.DataFrame(records)


def binary_metrics(labels: np.ndarray, scores: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        roc_auc_score,
    )

    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions)),
        "sensitivity": float(np.mean(predictions[labels == 1] == 1)),
        "specificity": float(np.mean(predictions[labels == 0] == 0)),
    }


def stratified_auc_bootstrap(
    labels: np.ndarray,
    scores: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    negatives = np.flatnonzero(labels == 0)
    positives = np.flatnonzero(labels == 1)
    values = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        sample = np.concatenate(
            [
                rng.choice(negatives, size=len(negatives), replace=True),
                rng.choice(positives, size=len(positives), replace=True),
            ]
        )
        values[repetition] = roc_auc_score(labels[sample], scores[sample])
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))

