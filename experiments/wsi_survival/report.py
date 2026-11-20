# -*- coding: utf-8 -*-
"""Build the frozen TCGA-BRCA survival tables for one DELTA input branch.

Select raw or Reinhard-normalized input with --branch. Source data and
canonical result artifacts are read-only inputs. Report calculations and
workbook names retain the frozen study contract.
"""

# %% 1. IMPORTS
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


# %% 1A. WORKBOOK AND CSV OUTPUT SETTINGS
SAVE_OUTPUTS = True
SAVE_INDIVIDUAL_CSV = True
WORKBOOK_FILENAME = "survival_manuscript_results_delta_v1_3_p2.xlsx"

# %% 2. SELF-CONTAINED ANALYSIS FUNCTIONS
REQUIRED_COLUMNS = (
    "backbone", "split_id", "method", "domain", "metric", "value", "status"
)
RESULT_FILENAMES = (
    "survival_metrics_long.csv",
)
V13_RESULT_FILENAME = "ratio_results_delta_v1_3_p2.csv"
V13_COMBINED_RESULT_FILENAME = "delta_v1_3_p2.csv"
V13_PROMOTION_AUDIT_FILENAME = "final_promotion.json"
V13_POST_PROMOTION_AUDIT_FILENAME = "final_validation.json"
V13_METHOD_CONTRACT_SHA256 = (
    "ff4ce2b550ba8ab9f9019bc5bfdb7d777bcaec577ab747e07a81e2c3ff7edc2f"
)
V13_PROJECTION = "centered_euclidean_hard_projection"
V13_CANONICAL_METHODS = {
    "raw": ("DELTA-v1.3-p2", "DELTA-v1.3"),
    "reinhard": ("Reinhard-DELTA-v1.3-p2", "Reinhard-DELTA-v1.3"),
}
V13_REQUIRED_COLUMNS = set(REQUIRED_COLUMNS) | {
    "branch",
    "bank_fraction_percent",
}
FORBIDDEN_LEGACY_METHODS = {
    "DELTA",
    "Reinhard-DELTA",
    "DELTA-v1-final",
    "Reinhard-DELTA-v1-final",
}
# Final-only reporting uses immutable canonical method names. Legacy DELTA
# files are intentionally excluded above, so no alias may silently relabel one.
OFFICIAL_METHOD_ALIASES_BY_FILENAME: dict[str, dict[str, str]] = {}
DOMAIN_LABELS = {"in_center": "ID", "held_out_center": "OOD"}
METRIC_LABELS = {
    "uno_c_index": "Uno C-index",
    "harrell_c_index": "Harrell C-index",
    "integrated_brier_score": "IBS",
    "td_auc_mean": "Mean time-dependent AUC",
}
HIGHER_IS_BETTER = {
    "uno_c_index": True,
    "harrell_c_index": True,
    "integrated_brier_score": False,
    "td_auc_mean": True,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def discover_result_files(results_roots: Sequence[Path]) -> list[Path]:
    """Read the frozen control results for each backbone."""
    files = []
    for results_root in results_roots:
        root = Path(results_root)
        if not root.is_dir():
            raise FileNotFoundError(f"Results directory not found: {root}")
        for backbone_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            for filename in RESULT_FILENAMES:
                candidate = backbone_dir / filename
                if candidate.is_file():
                    files.append(candidate)
    if not files:
        raise FileNotFoundError(f"No canonical survival CSVs under {results_roots}")
    return files


def load_survival_results(
    results_roots: Sequence[Path],
    include_backbones: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load canonical CSVs, fail on conflicts, and derive mean tdAUC."""
    include = None if include_backbones is None else set(map(str, include_backbones))
    frames = []
    provenance_rows = []
    for path in discover_result_files(results_roots):
        frame = pd.read_csv(path)
        missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        frame = frame.loc[:, REQUIRED_COLUMNS].copy()
        source_methods = sorted(frame["method"].astype(str).unique().tolist())
        forbidden = sorted(set(source_methods).intersection(FORBIDDEN_LEGACY_METHODS))
        if forbidden:
            raise ValueError(
                f"Legacy DELTA methods are forbidden in the final-only report: "
                f"file={path}, methods={forbidden}"
            )
        method_aliases = OFFICIAL_METHOD_ALIASES_BY_FILENAME.get(path.name, {})
        frame["method"] = frame["method"].astype(str).replace(method_aliases)
        canonical_methods = sorted(frame["method"].unique().tolist())
        observed = set(frame["backbone"].astype(str).unique())
        if len(observed) != 1:
            raise ValueError(f"{path} must contain exactly one backbone")
        backbone = next(iter(observed))
        if backbone != path.parent.name:
            raise ValueError(
                f"Backbone/path mismatch: row={backbone}, directory={path.parent.name}"
            )
        if include is not None and backbone not in include:
            continue
        frame["source_file"] = str(path.resolve())
        frames.append(frame)
        provenance_rows.append(
            {
                "backbone": backbone,
                "source_file": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "source_methods": ";".join(source_methods),
                "canonical_methods": ";".join(canonical_methods),
                "method_aliases_applied": ";".join(
                    f"{source}->{target}"
                    for source, target in sorted(method_aliases.items())
                    if source in source_methods
                ),
            }
        )
    if not frames:
        raise ValueError("Selected backbones produced no result rows")

    combined = pd.concat(frames, ignore_index=True)
    combined["value"] = pd.to_numeric(combined["value"], errors="coerce")
    combined = combined[
        combined["status"].astype(str).eq("complete")
        & np.isfinite(combined["value"].to_numpy(dtype=float))
    ].copy()
    split_ids = sorted(combined["split_id"].astype(str).unique().tolist())
    if len(split_ids) != 1:
        raise ValueError(f"Exactly one frozen split is required; found {split_ids}")

    key = ["backbone", "split_id", "method", "domain", "metric"]
    conflicts = (
        combined.groupby(key, dropna=False)["value"]
        .agg(lambda values: np.ptp(np.asarray(values, dtype=float)))
        .gt(1e-12)
    )
    if conflicts.any():
        raise ValueError(
            "Conflicting duplicate rows: "
            + str(conflicts[conflicts].index.tolist()[:5])
        )
    combined = combined.sort_values(key + ["source_file"]).drop_duplicates(
        key, keep="last"
    )

    auc_metrics = {"td_auc_q25", "td_auc_q50", "td_auc_q75"}
    derived = []
    group_key = ["backbone", "split_id", "method", "domain"]
    auc_rows = combined[combined["metric"].isin(auc_metrics)]
    for keys, group in auc_rows.groupby(group_key, sort=True):
        if set(group["metric"].astype(str)) != auc_metrics:
            continue
        record = dict(zip(group_key, keys))
        record.update(
            {
                "metric": "td_auc_mean",
                "value": float(group["value"].mean()),
                "status": "complete",
                "source_file": "derived:mean(td_auc_q25,td_auc_q50,td_auc_q75)",
            }
        )
        derived.append(record)
    if derived:
        combined = pd.concat([combined, pd.DataFrame(derived)], ignore_index=True)
    return combined.sort_values(key).reset_index(drop=True), pd.DataFrame(provenance_rows)


def load_v13_p2_results(
    experiment_root: Path,
    include_backbones: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the promoted cap-free v1.3 p2 artifacts with fail-closed checks."""

    def scalar(operator: np.lib.npyio.NpzFile, key: str) -> object:
        if key not in operator.files:
            raise ValueError(f"Operator is missing {key}")
        value = np.asarray(operator[key])
        if value.size != 1:
            raise ValueError(f"Operator field must be scalar: {key} shape={value.shape}")
        return value.reshape(-1)[0].item()

    root = Path(experiment_root)
    if not root.is_dir():
        raise FileNotFoundError(f"DELTA-v1.3 experiment not found: {root}")

    promotion_path = root / "audit" / V13_PROMOTION_AUDIT_FILENAME
    post_promotion_path = root / "audit" / V13_POST_PROMOTION_AUDIT_FILENAME
    combined_result_path = root / "results" / V13_COMBINED_RESULT_FILENAME
    for required in (promotion_path, post_promotion_path, combined_result_path):
        if not required.is_file():
            raise FileNotFoundError(f"Corrected canonical artifact missing: {required}")
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    post_promotion = json.loads(post_promotion_path.read_text(encoding="utf-8"))
    combined_sha256 = sha256_file(combined_result_path)
    promotion_checks = {
        "status": promotion.get("status") == "complete",
        "backbone_count": int(promotion.get("backbone_count", -1)) == 20,
        "endpoint_count": int(promotion.get("endpoint_count", -1)) == 480,
        "pair_count": int(promotion.get("pair_count", -1)) == 55_988,
        "rank_input_rows": int(promotion.get("rank_input_rows", -1)) == 55_988,
        "fixed_rank_row_cap": promotion.get("fixed_rank_row_cap") is None,
        "projection": promotion.get("projection") == V13_PROJECTION,
        "universal_euclidean": promotion.get("universal_euclidean") is True,
        "results_sha256": promotion.get("results_sha256") == combined_sha256,
    }
    failed = [name for name, passed in promotion_checks.items() if not passed]
    if failed:
        raise ValueError(f"Invalid final promotion audit {promotion_path}: {failed}")
    post_checks = {
        "status": post_promotion.get("status") == "complete",
        "backbone_count": int(post_promotion.get("backbone_count", -1)) == 20,
        "branch_count": int(post_promotion.get("branch_count", -1)) == 40,
        "endpoint_count": int(post_promotion.get("endpoint_count", -1)) == 480,
        "pair_count": int(post_promotion.get("pair_count", -1)) == 55_988,
        "rank_input_rows": int(post_promotion.get("rank_input_rows", -1)) == 55_988,
        "fixed_rank_row_cap": post_promotion.get("fixed_rank_row_cap") is None,
        "projection": post_promotion.get("projection") == V13_PROJECTION,
        "universal_euclidean": post_promotion.get("universal_euclidean") is True,
        "results_sha256": post_promotion.get("results_sha256") == combined_sha256,
    }
    failed = [name for name, passed in post_checks.items() if not passed]
    if failed:
        raise ValueError(
            f"Invalid final validation audit {post_promotion_path}: {failed}"
        )

    post_records = {
        (str(record["backbone"]), str(record["branch"])): record
        for record in post_promotion.get("records", [])
    }
    frames: list[pd.DataFrame] = []
    provenance_rows: list[dict[str, object]] = []
    for backbone in stable_unique(include_backbones):
        source = root / "artifacts" / backbone / V13_RESULT_FILENAME
        if not source.is_file():
            raise FileNotFoundError(f"Official DELTA-v1.3 p2 result missing: {source}")
        frame = pd.read_csv(source)
        missing = sorted(V13_REQUIRED_COLUMNS.difference(frame.columns))
        if missing:
            raise ValueError(f"{source} is missing columns: {missing}")
        observed_backbones = set(frame["backbone"].astype(str))
        if observed_backbones != {backbone}:
            raise ValueError(
                f"Backbone mismatch in {source}: expected={backbone}, "
                f"observed={sorted(observed_backbones)}"
            )
        fractions = set(frame["bank_fraction_percent"].astype(int))
        if fractions != {2}:
            raise ValueError(f"Non-p2 rows in {source}: {sorted(fractions)}")
        branches = set(frame["branch"].astype(str))
        if branches != set(V13_CANONICAL_METHODS):
            raise ValueError(f"Both frozen branches are required in {source}: {branches}")
        if not frame["status"].astype(str).eq("complete").all():
            raise ValueError(f"Non-complete DELTA-v1.3 rows in {source}")
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        if not np.isfinite(frame["value"].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite DELTA-v1.3 values in {source}")

        contract_paths: list[Path] = []
        rank_records: list[str] = []
        for branch, (source_method, canonical_method) in V13_CANONICAL_METHODS.items():
            branch_rows = frame[frame["branch"].astype(str).eq(branch)]
            methods = set(branch_rows["method"].astype(str))
            if methods != {source_method}:
                raise ValueError(
                    f"Unexpected v1.3 method in {source}, branch={branch}: {methods}"
                )
            branch_root = root / "artifacts" / backbone / branch / "ratios" / "p02"
            manifest_path = branch_root / "manifest.json"
            operator_path = branch_root / "operator.npz"
            patient_features_path = branch_root / "patient_features.parquet"
            for required in (manifest_path, operator_path, patient_features_path):
                if not required.is_file():
                    raise FileNotFoundError(f"Corrected branch artifact missing: {required}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            post_record = post_records.get((backbone, branch))
            if post_record is None:
                raise ValueError(f"Missing final branch audit: {backbone}/{branch}")
            with np.load(operator_path, allow_pickle=False) as operator:
                pair_count = int(scalar(operator, "pair_count"))
                rank_input_rows = int(scalar(operator, "rank_input_rows"))
                applied_rank = int(scalar(operator, "applied_rank"))
                intrinsic_rank = int(scalar(operator, "intrinsic_rank"))
                operator_contract_sha = str(scalar(operator, "method_contract_sha256"))
                projection = str(scalar(operator, "projection"))
            checks = {
                "schema": manifest.get("schema")
                == "delta-v1.3-p2-universal-euclidean-branch-v1",
                "status": manifest.get("status") == "complete",
                "backbone": manifest.get("backbone") == backbone,
                "branch": manifest.get("branch") == branch,
                "pairs": int(manifest.get("pair_count", -1)) == pair_count == 55_988,
                "rank_input_rows": int(manifest.get("rank_input_rows", -1))
                == rank_input_rows
                == 55_988,
                "all_pairs_used": manifest.get("all_pairs_used_for_rank") is True,
                "rank": int(manifest.get("applied_rank", -1))
                == applied_rank
                == int(post_record.get("rank", -1)),
                "intrinsic_rank": int(manifest.get("intrinsic_rank", -1))
                == intrinsic_rank,
                "projection": manifest.get("projection")
                == projection
                == V13_PROJECTION,
                "method_contract": manifest.get("method_contract_sha256")
                == operator_contract_sha
                == V13_METHOD_CONTRACT_SHA256,
                "operator_sha256": manifest.get("operator_sha256")
                == post_record.get("operator_sha256")
                == sha256_file(operator_path),
                "patient_features_sha256": manifest.get("patient_features_sha256")
                == post_record.get("patient_features_sha256")
                == sha256_file(patient_features_path),
                "outcome_free": manifest.get("outcomes_used_for_operator") is False,
                "transform_before_aggregation": manifest.get(
                    "transform_before_patient_aggregation"
                )
                is True,
            }
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                raise ValueError(f"Invalid corrected contract in {manifest_path}: {failed}")
            contract_paths.extend([manifest_path, operator_path, patient_features_path])
            rank_records.append(
                f"{branch}:projection={projection};intrinsic_rank={intrinsic_rank};"
                f"applied_rank={applied_rank};rule=null_adjusted_90pct;"
                f"cap_free=True;pairs={pair_count};rank_input_rows={rank_input_rows}"
            )
            frame.loc[frame["branch"].astype(str).eq(branch), "method"] = canonical_method

        loaded = frame.loc[:, REQUIRED_COLUMNS].copy()
        loaded["source_file"] = str(source.resolve())
        frames.append(loaded)
        provenance_rows.append(
            {
                "backbone": backbone,
                "source_file": str(source.resolve()),
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
                "source_methods": ";".join(
                    source_method for source_method, _ in V13_CANONICAL_METHODS.values()
                ),
                "canonical_methods": ";".join(
                    canonical for _, canonical in V13_CANONICAL_METHODS.values()
                ),
                "method_aliases_applied": ";".join(
                    f"{source_method}->{canonical}"
                    for source_method, canonical in V13_CANONICAL_METHODS.values()
                ),
                "rank_contract": " | ".join(rank_records),
                "corrected_contract_files": ";".join(
                    str(path.resolve()) for path in contract_paths
                ),
                "corrected_contract_sha256": ";".join(
                    sha256_file(path) for path in contract_paths
                ),
                "promotion_audit": str(promotion_path.resolve()),
                "promotion_audit_sha256": sha256_file(promotion_path),
                "post_promotion_audit": str(post_promotion_path.resolve()),
                "post_promotion_audit_sha256": sha256_file(post_promotion_path),
                "combined_result": str(combined_result_path.resolve()),
                "combined_result_sha256": combined_sha256,
            }
        )

    combined = pd.concat(frames, ignore_index=True)
    split_ids = sorted(combined["split_id"].astype(str).unique().tolist())
    if len(split_ids) != 1:
        raise ValueError(f"Exactly one frozen v1.3 split is required: {split_ids}")
    key = ["backbone", "split_id", "method", "domain", "metric"]
    if combined.duplicated(key).any():
        duplicates = combined.loc[combined.duplicated(key, keep=False), key]
        raise ValueError(f"Duplicate v1.3 endpoint rows: {duplicates.head().to_dict('records')}")

    official = pd.read_csv(combined_result_path)
    official = official.loc[
        official["backbone"].astype(str).isin(stable_unique(include_backbones))
    ].copy()
    official["method"] = official["branch"].astype(str).map(
        {branch: canonical for branch, (_, canonical) in V13_CANONICAL_METHODS.items()}
    )
    official = official.loc[:, REQUIRED_COLUMNS].sort_values(key).reset_index(drop=True)
    observed = combined.loc[:, REQUIRED_COLUMNS].sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(official, observed, check_exact=True)

    auc_metrics = {"td_auc_q25", "td_auc_q50", "td_auc_q75"}
    derived: list[dict[str, object]] = []
    group_key = ["backbone", "split_id", "method", "domain"]
    for keys, group in combined[combined["metric"].isin(auc_metrics)].groupby(
        group_key, sort=True
    ):
        if set(group["metric"].astype(str)) != auc_metrics:
            raise ValueError(f"Incomplete locked tdAUC horizons for {keys}")
        record = dict(zip(group_key, keys))
        record.update(
            {
                "metric": "td_auc_mean",
                "value": float(group["value"].mean()),
                "status": "complete",
                "source_file": "derived:mean(td_auc_q25,td_auc_q50,td_auc_q75)",
            }
        )
        derived.append(record)
    combined = pd.concat([combined, pd.DataFrame(derived)], ignore_index=True)
    return combined.sort_values(key).reset_index(drop=True), pd.DataFrame(provenance_rows)


def bootstrap_mean_ci(
    values: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
    confidence: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1 or repetitions <= 0:
        value = float(values.mean())
        return value, value
    means = np.empty(int(repetitions), dtype=float)
    for start in range(0, int(repetitions), 4096):
        stop = min(start + 4096, int(repetitions))
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high)


def summarize_methods(
    analysis_long: pd.DataFrame,
    bootstrap_replicates: int = 20_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Descriptive summary using every available complete backbone."""
    rng = np.random.default_rng(seed)
    rows = []
    subset = analysis_long[analysis_long["metric"].isin(HIGHER_IS_BETTER)]
    for keys, frame in subset.groupby(["method", "domain", "metric"], sort=True):
        method, domain, metric = keys
        if frame["backbone"].duplicated().any():
            raise ValueError(f"Multiple rows per backbone for {keys}")
        values = frame["value"].to_numpy(dtype=float)
        low, high = bootstrap_mean_ci(values, bootstrap_replicates, rng)
        rows.append(
            {
                "method": method,
                "domain": domain,
                "domain_label": DOMAIN_LABELS.get(domain, domain),
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "higher_is_better": HIGHER_IS_BETTER[metric],
                "n_backbones": len(values),
                "mean": float(values.mean()),
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    return pd.DataFrame(rows)


def summarize_common_backbones(
    analysis_long: pd.DataFrame,
    methods: Sequence[str],
    metrics: Sequence[str] = ("uno_c_index",),
    bootstrap_replicates: int = 20_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Summary on the same complete-case backbone set per endpoint."""
    methods = stable_unique(methods)
    rng = np.random.default_rng(seed)
    rows = []
    subset = analysis_long[
        analysis_long["method"].isin(methods)
        & analysis_long["metric"].isin(metrics)
    ]
    for (domain, metric), frame in subset.groupby(["domain", "metric"], sort=True):
        pivot = frame.pivot(index="backbone", columns="method", values="value")
        if any(method not in pivot.columns for method in methods):
            continue
        common = pivot[methods].dropna()
        if common.empty:
            continue
        backbone_set = ";".join(sorted(common.index.astype(str).tolist()))
        for method in methods:
            values = common[method].to_numpy(dtype=float)
            low, high = bootstrap_mean_ci(values, bootstrap_replicates, rng)
            rows.append(
                {
                    "method": method,
                    "domain": domain,
                    "domain_label": DOMAIN_LABELS.get(domain, domain),
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "higher_is_better": HIGHER_IS_BETTER[metric],
                    "n_backbones": len(values),
                    "mean": float(values.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "common_backbones": backbone_set,
                }
            )
    return pd.DataFrame(rows)


def studentized_from_sums(
    signed_sum: np.ndarray, sum_squares: np.ndarray, n: int
) -> np.ndarray:
    means = signed_sum / n
    variance = np.maximum((sum_squares - n * means * means) / (n - 1), 0.0)
    standard_error = np.sqrt(variance / n)
    result = np.zeros_like(means, dtype=float)
    np.divide(means, standard_error, out=result, where=standard_error > 1e-15)
    result[(standard_error <= 1e-15) & (np.abs(means) > 1e-15)] = np.inf
    return result


def exact_sign_batches(n: int, batch_size: int = 8192) -> Iterator[np.ndarray]:
    total = 1 << n
    bit_positions = np.arange(n, dtype=np.uint64)
    for start in range(0, total, batch_size):
        numbers = np.arange(start, min(start + batch_size, total), dtype=np.uint64)
        bits = ((numbers[:, None] >> bit_positions[None, :]) & 1).astype(np.int8)
        yield bits * 2 - 1


def random_sign_batches(
    n: int, permutations: int, rng: np.random.Generator, batch_size: int = 8192
) -> Iterator[np.ndarray]:
    for start in range(0, permutations, batch_size):
        count = min(batch_size, permutations - start)
        yield rng.integers(0, 2, size=(count, n), dtype=np.int8) * 2 - 1


def max_t_sign_flip(
    differences: np.ndarray,
    permutations: int = 100_000,
    exact_max_n: int = 20,
    seed: int = 42,
) -> dict[str, object]:
    """Paired two-sided sign-flip test with common-control maxT adjustment."""
    differences = np.asarray(differences, dtype=float)
    if differences.ndim != 2 or differences.shape[0] < 3:
        raise ValueError("maxT needs n_backbones x n_controls with n >= 3")
    if not np.isfinite(differences).all():
        raise ValueError("maxT differences must be finite")
    n, control_count = differences.shape
    sum_squares = np.square(differences).sum(axis=0)
    observed_t = studentized_from_sums(differences.sum(axis=0), sum_squares, n)
    raw_counts = np.zeros(control_count, dtype=np.int64)
    adjusted_counts = np.zeros(control_count, dtype=np.int64)
    exact = n <= exact_max_n
    if exact:
        batches = exact_sign_batches(n)
        total = 1 << n
        correction = 0
    else:
        batches = random_sign_batches(n, permutations, np.random.default_rng(seed))
        total = permutations
        correction = 1
    for signs in batches:
        permuted_t = studentized_from_sums(signs @ differences, sum_squares, n)
        absolute = np.abs(permuted_t)
        maximum = absolute.max(axis=1)
        raw_counts += (absolute >= np.abs(observed_t)[None, :]).sum(axis=0)
        adjusted_counts += (
            maximum[:, None] >= np.abs(observed_t)[None, :]
        ).sum(axis=0)
    denominator = total + correction
    return {
        "observed_t": observed_t,
        "p_raw": (raw_counts + correction) / denominator,
        "p_max_t": (adjusted_counts + correction) / denominator,
        "exact": exact,
        "permutations": total,
    }


def compare_target(
    analysis_long: pd.DataFrame,
    target: str,
    controls: Sequence[str],
    bootstrap_replicates: int = 20_000,
    permutations: int = 100_000,
    exact_max_n: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a common-control maxT sensitivity analysis."""
    controls = stable_unique(controls)
    rng = np.random.default_rng(seed)
    rows = []
    subset = analysis_long[analysis_long["metric"].isin(HIGHER_IS_BETTER)]
    for (domain, metric), frame in subset.groupby(["domain", "metric"], sort=True):
        pivot = frame.pivot(index="backbone", columns="method", values="value")
        available = [control for control in controls if control in pivot.columns]
        if target not in pivot.columns or not available:
            continue
        family = pivot[[target, *available]].dropna()
        if len(family) < 3:
            continue
        differences = family[available].rsub(family[target], axis=0)
        if not HIGHER_IS_BETTER[metric]:
            differences = -differences
        test = max_t_sign_flip(
            differences.to_numpy(dtype=float), permutations, exact_max_n, seed
        )
        for index, control in enumerate(available):
            values = differences[control].to_numpy(dtype=float)
            low, high = bootstrap_mean_ci(values, bootstrap_replicates, rng)
            rows.append(
                {
                    "target": target,
                    "control": control,
                    "domain": domain,
                    "domain_label": DOMAIN_LABELS.get(domain, domain),
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "effect_definition": (
                        f"{target} - {control}"
                        if HIGHER_IS_BETTER[metric]
                        else f"{control} - {target} (IBS; positive favors {target})"
                    ),
                    "n_backbones": len(family),
                    "target_mean": float(family[target].mean()),
                    "control_mean": float(family[control].mean()),
                    "favorable_delta_mean": float(values.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "wins": int((values > 1e-12).sum()),
                    "ties": int((np.abs(values) <= 1e-12).sum()),
                    "losses": int((values < -1e-12).sum()),
                    "studentized_t": float(np.asarray(test["observed_t"])[index]),
                    "p_raw": float(np.asarray(test["p_raw"])[index]),
                    "p_max_t": float(np.asarray(test["p_max_t"])[index]),
                    "permutation_mode": "exact" if test["exact"] else "monte_carlo",
                    "permutations": int(test["permutations"]),
                    "family": f"{domain}|{metric}|{target}-focused",
                }
            )
    return pd.DataFrame(rows)


def compare_target_pairwise(
    analysis_long: pd.DataFrame,
    target: str,
    controls: Sequence[str],
    bootstrap_replicates: int = 20_000,
    permutations: int = 100_000,
    exact_max_n: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate each target-versus-control hypothesis as a separate pair.

    Every contrast uses its own complete paired-backbone set.  ``p_comparison``
    is a two-sided exact (or Monte Carlo when n is large) paired sign-flip test
    of the studentized mean difference and is intentionally unadjusted.  The
    Wilcoxon and exact sign-test values are robustness summaries, not a device
    for selecting whichever test happens to be smallest.
    """
    controls = stable_unique(controls)
    rng = np.random.default_rng(seed)
    rows = []
    subset = analysis_long[analysis_long["metric"].isin(HIGHER_IS_BETTER)]
    for (domain, metric), frame in subset.groupby(["domain", "metric"], sort=True):
        pivot = frame.pivot(index="backbone", columns="method", values="value")
        if target not in pivot.columns:
            continue
        for control_index, control in enumerate(controls):
            if control not in pivot.columns:
                continue
            paired = pivot[[target, control]].dropna()
            if len(paired) < 3:
                continue
            values = (paired[target] - paired[control]).to_numpy(dtype=float)
            if not HIGHER_IS_BETTER[metric]:
                values = -values
            test = max_t_sign_flip(
                values[:, None],
                permutations=permutations,
                exact_max_n=exact_max_n,
                seed=seed + control_index,
            )
            low, high = bootstrap_mean_ci(values, bootstrap_replicates, rng)
            wins = int((values > 1e-12).sum())
            ties = int((np.abs(values) <= 1e-12).sum())
            losses = int((values < -1e-12).sum())
            nonzero = values[np.abs(values) > 1e-12]
            sign_p = (
                float(binomtest(wins, wins + losses, 0.5).pvalue)
                if wins + losses
                else 1.0
            )
            wilcoxon_p = (
                float(wilcoxon(nonzero, method="exact").pvalue)
                if len(nonzero)
                else 1.0
            )
            standard_deviation = float(values.std(ddof=1))
            p_comparison = float(np.asarray(test["p_raw"])[0])
            rows.append(
                {
                    "target": target,
                    "control": control,
                    "comparison": f"{target} vs {control}",
                    "domain": domain,
                    "domain_label": DOMAIN_LABELS.get(domain, domain),
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "effect_definition": (
                        f"{target} - {control}"
                        if HIGHER_IS_BETTER[metric]
                        else f"{control} - {target} (IBS; positive favors {target})"
                    ),
                    "n_backbones": len(paired),
                    "target_mean": float(paired[target].mean()),
                    "control_mean": float(paired[control].mean()),
                    "favorable_delta_mean": float(values.mean()),
                    "median_difference": float(np.median(values)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "cohen_dz": (
                        float(values.mean() / standard_deviation)
                        if standard_deviation > 1e-15
                        else np.nan
                    ),
                    "studentized_t": float(np.asarray(test["observed_t"])[0]),
                    "p_comparison": p_comparison,
                    "p_raw": p_comparison,
                    "p_wilcoxon": wilcoxon_p,
                    "p_sign": sign_p,
                    "permutation_mode": "exact" if test["exact"] else "monte_carlo",
                    "permutations": int(test["permutations"]),
                    "multiplicity_adjusted": False,
                    "hypothesis_family": "separate_prespecified_pairwise_contrast",
                    "common_backbones": ";".join(sorted(paired.index.astype(str))),
                }
            )
    return pd.DataFrame(rows)


def build_main_table(
    summary: pd.DataFrame, methods: Sequence[str]
) -> pd.DataFrame:
    frame = summary[summary["metric"].eq("uno_c_index")].copy()
    rows = []
    for method in methods:
        row = {"method": method}
        for domain, label in DOMAIN_LABELS.items():
            match = frame[
                frame["method"].eq(method) & frame["domain"].eq(domain)
            ]
            if match.empty:
                row.update(
                    {
                        f"{label}_n": 0,
                        f"{label}_mean": np.nan,
                        f"{label}_ci95_low": np.nan,
                        f"{label}_ci95_high": np.nan,
                    }
                )
            else:
                value = match.iloc[0]
                row.update(
                    {
                        f"{label}_n": int(value["n_backbones"]),
                        f"{label}_mean": float(value["mean"]),
                        f"{label}_ci95_low": float(value["ci95_low"]),
                        f"{label}_ci95_high": float(value["ci95_high"]),
                    }
                )
        rows.append(row)
    return pd.DataFrame(rows)


def format_estimate(row: pd.Series, prefix: str) -> str:
    if not np.isfinite(float(row[f"{prefix}_mean"])):
        return "NA"
    return (
        f"{float(row[f'{prefix}_mean']):.4f} "
        f"[{float(row[f'{prefix}_ci95_low']):.4f}, "
        f"{float(row[f'{prefix}_ci95_high']):.4f}]"
    )


def write_main_table_markdown(main_table: pd.DataFrame, path: Path) -> None:
    ranks = {}
    for prefix in ("ID", "OOD"):
        ordered = (
            main_table[["method", f"{prefix}_mean"]]
            .dropna()
            .sort_values(f"{prefix}_mean", ascending=False)["method"]
            .tolist()
        )
        ranks[prefix] = {method: rank for rank, method in enumerate(ordered, 1)}
    lines = [
        "| Method | ID Uno C-index, mean [95% CI] | OOD Uno C-index, mean [95% CI] |",
        "|:--|--:|--:|",
    ]
    for _, row in main_table.iterrows():
        cells = []
        for prefix in ("ID", "OOD"):
            text = format_estimate(row, prefix)
            rank = ranks[prefix].get(str(row["method"]))
            if rank == 1:
                text = f"**{text}**"
            elif rank == 2:
                text = f"<u>{text}</u>"
            cells.append(text)
        lines.append(f"| {row['method']} | {cells[0]} | {cells[1]} |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# %% 3. PATH SETTINGS
PROJECT_ROOT = Path(
    os.environ.get(
        "DELTA_PROJECT_ROOT",
        Path(__file__).resolve().parents[2],
    )
).expanduser().resolve()
RESULTS_ROOTS = (
    Path(
        os.environ.get(
            "DELTA_SURVIVAL_CONTROL_RESULTS_ROOT",
            PROJECT_ROOT
            / (
                "experiments/"
                "tcga_brca_full-wsi_survival__20backbone__"
                "delta-v1-3-p2__null-adjusted90__v001/controls/baseline_results"
            ),
        )
    ).expanduser().resolve(),
)
V13_EXPERIMENT_ROOT = Path(
    os.environ.get(
        "DELTA_SURVIVAL_V13_EXPERIMENT_ROOT",
        PROJECT_ROOT
        / (
            "experiments/"
            "tcga_brca_full-wsi_survival__20backbone__"
            "delta-v1-3-p2__null-adjusted90__v001"
        ),
    )
).expanduser().resolve()
OUTPUT_DIR = Path(
    os.environ.get(
        "DELTA_SURVIVAL_REPORT_OUTPUT_DIR",
        PROJECT_ROOT
        / "results/tcga_brca_wsi_survival_report_delta_v1_3_p2/backbones20",
    )
).expanduser().resolve()


# %% 4. CHOOSE BACKBONES
BACKBONE_IDS = {
    "Original UNI": "uni",
    "Phikon-v2": "phikonv2",
    "Virchow2": "virchow2",
    "CONCH": "conch",
    "CONCH v1.5": "conchv15",
    "UNI2-h": "uni2h",
    "H-optimus-0": "hoptimus0",
    "H0-mini": "h0mini",
    "EXAONE-Path-2.5": "exaonepath25",
    "GPFM": "gpfm",
    "MOOZY (Lunit DINO)": "moozy_patch",
    "Prov-GigaPath": "gigapath",
    "MUSK": "musk",
    "Ciga": "ciga",
    "HIPT": "hipt",
    "RetCCL": "retccl",
    "CTransPath": "ctranspath",
    "Kaiko ViT-B/8": "kaiko_vitb8",
    "Virchow": "virchow",
    "BEPH": "beph",
}

# Add/remove names here.  Their written order is preserved in the selection
# table, although paired statistics themselves do not depend on order.
INCLUDE_BACKBONES = [
    "Original UNI",
    "UNI2-h",
    "Virchow",
    "Virchow2",
    "CONCH v1.5",
    "H-optimus-0",
    "H0-mini",
    "GPFM",
    "Prov-GigaPath",
    "Ciga",
    "HIPT",
    "RetCCL",
    "Kaiko ViT-B/8",
    "Phikon-v2",
    "CTransPath",
    "CONCH",
    "BEPH",
    "MOOZY (Lunit DINO)",
    "EXAONE-Path-2.5",
    "MUSK",
]

EXCLUDE_BACKBONES = [
    # "EXAONE-Path-2.5",
    # "H0-mini",
    # "MUSK",  # Uncomment after its canonical result is complete.
]

# "error": stop if any selected backbone has no canonical result file.
# "drop": continue with complete selected backbones and print what was absent.
MISSING_BACKBONE_POLICY = "error"


# %% 5. METHOD AND STATISTICAL SETTINGS
TARGET_METHOD = "DELTA-v1.3"
UNO_CONTRAST_SHEET = "MAIN_02_DELTA_Contrasts"

# Main-text hypotheses.  Each row is interpreted as a separate, prespecified
# one-to-one paired contrast.  Main-text P values are not multiplicity adjusted.
MAIN_COMPARATORS = [
    "RAW",
    "DANN",
    "Reinhard-RAW",
    "Reinhard-DANN",
]

# Contextual methods remain available in Supplementary tables.  Transductive
# ComBat rows are deliberately not interpreted as deployment-equivalent.
CONTEXTUAL_COMPARATORS = [
    "Reinhard-DELTA-v1.3",
    "ComBat-inductive",
    "ComBat-transductive",
    "Reinhard-ComBat-inductive",
    "Reinhard-ComBat-transductive",
]

PAIRWISE_COMPARATORS = [*MAIN_COMPARATORS, *CONTEXTUAL_COMPARATORS]
MANUSCRIPT_METHODS = [
    *MAIN_COMPARATORS,
    TARGET_METHOD,
    "Reinhard-DELTA-v1.3",
]

# Multiplicity-adjusted common-control maxT remains a sensitivity analysis only.
MAX_T_SENSITIVITY_CONTROLS = list(MAIN_COMPARATORS)
PRIMARY_P_VALUE = "p_comparison"
REPORT_UNADJUSTED_P = True
BUILD_MULTIPLICITY_SENSITIVITY = True

BOOTSTRAP_REPLICATES = 20_000
MONTE_CARLO_PERMUTATIONS = 100_000
EXACT_SIGN_FLIP_MAX_BACKBONES = 21
RANDOM_SEED = 42


# %% 6. SMALL INTERACTIVE HELPERS
def resolve_selected_backbones() -> tuple[list[str], pd.DataFrame]:
    """Translate readable names to canonical IDs and validate the selection."""
    unknown = sorted((set(INCLUDE_BACKBONES) | set(EXCLUDE_BACKBONES)) - set(BACKBONE_IDS))
    if unknown:
        raise ValueError(f"Unknown readable backbone names: {unknown}")
    if len(INCLUDE_BACKBONES) != len(set(INCLUDE_BACKBONES)):
        raise ValueError("INCLUDE_BACKBONES contains duplicates")
    excluded = set(EXCLUDE_BACKBONES)
    readable = [name for name in INCLUDE_BACKBONES if name not in excluded]
    if len(readable) < 3:
        raise ValueError("Select at least three backbones")
    canonical = [BACKBONE_IDS[name] for name in readable]
    table = pd.DataFrame(
        {
            "order": np.arange(1, len(readable) + 1),
            "backbone_display": readable,
            "backbone": canonical,
        }
    )
    return canonical, table


def show_compact_results(
    main_table_frame: pd.DataFrame,
    contrast_frame: pd.DataFrame,
) -> None:
    """Print the two tables most useful during line-by-line exploration."""
    print("\n" + "=" * 88)
    print("COMMON-BACKBONE MANUSCRIPT TABLE: UNO C-INDEX")
    print(main_table_frame.round(4).to_string(index=False))
    print(f"\n{TARGET_METHOD}-FOCUSED PAIRED CONTRASTS")
    columns = [
        "control",
        "domain_label",
        "metric_label",
        "n_backbones",
        "favorable_delta_mean",
        "ci95_low",
        "ci95_high",
        "wins",
        "ties",
        "losses",
        "cohen_dz",
        "p_comparison",
        "p_wilcoxon",
        "p_sign",
    ]
    print(contrast_frame[columns].round(4).to_string(index=False))












def write_manuscript_workbook(
    path: Path,
    main_table_frame: pd.DataFrame,
    pairwise_contrast_frame: pd.DataFrame,
    multiplicity_sensitivity_frame: pd.DataFrame,
    broad_summary_frame: pd.DataFrame,
    paired_summary_frame: pd.DataFrame,
    coverage_frame: pd.DataFrame,
    backbone_frame: pd.DataFrame,
    provenance_frame: pd.DataFrame,
    analysis_frame: pd.DataFrame,
) -> Path:
    """Write one publication-style workbook with MAIN_* and SUPP_* groups."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
        from openpyxl.utils import get_column_letter
        from openpyxl.worksheet.page import PageMargins
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Excel output requires openpyxl. In HAIPAI_v1_py310 run once: "
            "python -m pip install openpyxl"
        ) from error

    def excel_value(value):
        if value is None or bool(pd.isna(value)):
            return None
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value

    thin_black = Side(style="thin", color="000000")
    medium_black = Side(style="medium", color="000000")
    header_fill = PatternFill("solid", fgColor="E7E6E6")
    main_tab_color = "1F4E78"
    supplementary_tab_color = "7F7F7F"

    def estimate_ci(mean, low, high, signed: bool = False) -> str:
        prefix = "+" if signed and float(mean) > 0 else ""
        return (
            f"{prefix}{float(mean):.4f} "
            f"({float(low):.4f}\N{EN DASH}{float(high):.4f})"
        )

    def configure_print(worksheet, final_row: int, final_column: int) -> None:
        worksheet.sheet_view.showGridLines = True
        worksheet.sheet_view.zoomScale = 90
        worksheet.print_options.gridLines = False
        worksheet.print_options.headings = False
        worksheet.sheet_properties.pageSetUpPr.fitToPage = True
        worksheet.page_setup.orientation = "landscape"
        worksheet.page_setup.fitToWidth = 1
        worksheet.page_setup.fitToHeight = 0
        worksheet.page_margins = PageMargins(
            left=0.35,
            right=0.35,
            top=0.45,
            bottom=0.45,
            header=0.2,
            footer=0.2,
        )
        worksheet.print_area = (
            f"A1:{get_column_letter(final_column)}{final_row}"
        )

    def add_supplementary_sheet(
        workbook,
        name: str,
        frame: pd.DataFrame,
        freeze: str = "A2",
    ):
        worksheet = workbook.create_sheet(name)
        worksheet.sheet_view.showGridLines = True
        worksheet.sheet_view.zoomScale = 90
        worksheet.freeze_panes = freeze
        worksheet.sheet_properties.tabColor = supplementary_tab_color
        columns = [str(column) for column in frame.columns]
        worksheet.append(columns)
        for row in frame.itertuples(index=False, name=None):
            worksheet.append([excel_value(value) for value in row])
        if columns:
            worksheet.auto_filter.ref = worksheet.dimensions
            for cell in worksheet[1]:
                cell.fill = header_fill
                cell.font = Font(name="Arial", size=9, bold=True, color="000000")
                cell.alignment = Alignment(horizontal="center", vertical="center")
                cell.border = Border(top=thin_black, bottom=thin_black)
            worksheet.row_dimensions[1].height = 22
        for row in worksheet.iter_rows(min_row=2):
            for cell in row:
                cell.font = Font(name="Arial", size=8.5)
                cell.alignment = Alignment(vertical="center")
                if isinstance(cell.value, float):
                    cell.number_format = "0.0000"
            worksheet.row_dimensions[row[0].row].height = 18
        for column_index, column_name in enumerate(columns, start=1):
            values = [column_name]
            for row_index in range(2, min(worksheet.max_row, 500) + 1):
                value = worksheet.cell(row=row_index, column=column_index).value
                values.append("" if value is None else str(value))
            width = min(max(max(map(len, values)) + 2, 10), 44)
            if "source_file" in column_name or "common_backbones" in column_name:
                width = 44
            worksheet.column_dimensions[get_column_letter(column_index)].width = width
        configure_print(worksheet, worksheet.max_row, max(1, worksheet.max_column))
        worksheet.print_title_rows = "1:1"
        return worksheet

    def add_main_absolute_sheet(workbook) -> object:
        worksheet = workbook.create_sheet("MAIN_01_Uno")
        worksheet.sheet_properties.tabColor = main_tab_color
        worksheet.merge_cells("A1:E1")
        worksheet["A1"] = "Table 1 | Absolute Uno C-index across pathology foundation models"
        worksheet["A1"].font = Font(name="Arial", size=11, bold=True)
        worksheet["A1"].alignment = Alignment(horizontal="left", vertical="center")
        worksheet.row_dimensions[1].height = 24

        worksheet["A2"] = ""
        worksheet.merge_cells("B2:C2")
        worksheet["B2"] = "ID"
        worksheet.merge_cells("D2:E2")
        worksheet["D2"] = "OOD"
        for cell in worksheet[2]:
            cell.font = Font(name="Arial", size=9, bold=True)
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = Border(top=medium_black)

        headers = ["Method", "n", "Mean (95% CI)", "n", "Mean (95% CI)"]
        for column, value in enumerate(headers, start=1):
            cell = worksheet.cell(3, column, value)
            cell.font = Font(name="Arial", size=9, bold=True)
            cell.alignment = Alignment(
                horizontal="left" if column == 1 else "center",
                vertical="center",
            )
            cell.border = Border(bottom=thin_black)
        worksheet.row_dimensions[3].height = 22

        data_start = 4
        for row_offset, method in enumerate(MANUSCRIPT_METHODS):
            selected = main_table_frame[main_table_frame["method"].eq(method)]
            if len(selected) != 1:
                raise ValueError(f"Expected one main-table row for method={method}")
            result = selected.iloc[0]
            values = [
                method,
                int(result["ID_n"]),
                estimate_ci(result["ID_mean"], result["ID_ci95_low"], result["ID_ci95_high"]),
                int(result["OOD_n"]),
                estimate_ci(result["OOD_mean"], result["OOD_ci95_low"], result["OOD_ci95_high"]),
            ]
            row_index = data_start + row_offset
            for column, value in enumerate(values, start=1):
                cell = worksheet.cell(row_index, column, value)
                cell.font = Font(
                    name="Arial",
                    size=9,
                    bold=(column == 1 and method == TARGET_METHOD),
                )
                cell.alignment = Alignment(
                    horizontal="left" if column == 1 else "center",
                    vertical="center",
                )
            worksheet.row_dimensions[row_index].height = 19

        data_end = data_start + len(MANUSCRIPT_METHODS) - 1
        for cell in worksheet[data_end]:
            cell.border = Border(bottom=medium_black)

        # Bold best and underline second-best within ID and OOD.
        for mean_key, column in (("ID_mean", 3), ("OOD_mean", 5)):
            ranked = sorted(
                (
                    float(main_table_frame.loc[main_table_frame["method"].eq(method), mean_key].iloc[0]),
                    data_start + index,
                )
                for index, method in enumerate(MANUSCRIPT_METHODS)
            )
            best_row = ranked[-1][1]
            second_row = ranked[-2][1]
            worksheet.cell(best_row, column).font = Font(name="Arial", size=9, bold=True)
            worksheet.cell(second_row, column).font = Font(name="Arial", size=9, underline="single")

        notes = [
            "Values are means with percentile-bootstrap 95% confidence intervals across backbones.",
            "ID, in-center evaluation; OOD, held-out-center evaluation. Bold indicates the highest mean; underline indicates the second highest mean.",
        ]
        note_start = data_end + 2
        for offset, note in enumerate(notes):
            row_index = note_start + offset
            worksheet.merge_cells(start_row=row_index, start_column=1, end_row=row_index, end_column=5)
            cell = worksheet.cell(row_index, 1, note)
            cell.font = Font(name="Arial", size=8, italic=True)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            worksheet.row_dimensions[row_index].height = 22

        widths = {"A": 22, "B": 8, "C": 25, "D": 8, "E": 25}
        for column, width in widths.items():
            worksheet.column_dimensions[column].width = width
        worksheet.freeze_panes = "A4"
        worksheet.print_title_rows = "1:3"
        configure_print(worksheet, note_start + len(notes) - 1, 5)
        return worksheet

    def add_main_contrast_sheet(
        workbook,
        *,
        sheet_name: str,
        table_number: int,
        metric: str,
        metric_label: str,
    ) -> object:
        worksheet = workbook.create_sheet(sheet_name)
        worksheet.sheet_properties.tabColor = main_tab_color
        worksheet.merge_cells("A1:H1")
        worksheet["A1"] = (
            f"Table {table_number} | Prespecified one-to-one paired comparisons "
            f"in {metric_label}"
        )
        worksheet["A1"].font = Font(name="Arial", size=11, bold=True)
        worksheet["A1"].alignment = Alignment(horizontal="left", vertical="center")
        worksheet.row_dimensions[1].height = 24

        headers = [
            "Domain",
            "Comparison",
            f"{TARGET_METHOD} mean",
            "Comparator mean",
            "Mean paired difference (95% CI)",
            "Wins / n",
            "Cohen's dz",
            "P",
        ]
        for column, value in enumerate(headers, start=1):
            cell = worksheet.cell(2, column, value)
            cell.font = Font(name="Arial", size=9, bold=True)
            cell.alignment = Alignment(
                horizontal="left" if column in (1, 2) else "center",
                vertical="center",
                wrap_text=True,
            )
            cell.border = Border(top=medium_black, bottom=thin_black)
        worksheet.row_dimensions[2].height = 32

        metric_contrasts = pairwise_contrast_frame[
            pairwise_contrast_frame["metric"].eq(metric)
            & pairwise_contrast_frame["control"].isin(MAIN_COMPARATORS)
        ].copy()
        data_start = 3
        ordered_rows = []
        for domain_label in ("ID", "OOD"):
            for control in MAIN_COMPARATORS:
                selected = metric_contrasts[
                    metric_contrasts["control"].eq(control)
                    & metric_contrasts["domain_label"].eq(domain_label)
                ]
                if len(selected) != 1:
                    raise ValueError(
                        f"Expected one {metric} contrast for control={control}, "
                        f"domain={domain_label}"
                    )
                ordered_rows.append(selected.iloc[0])

        for row_offset, result in enumerate(ordered_rows):
            values = [
                str(result["domain_label"]),
                str(result["comparison"]),
                float(result["target_mean"]),
                float(result["control_mean"]),
                estimate_ci(
                    result["favorable_delta_mean"],
                    result["ci95_low"],
                    result["ci95_high"],
                    signed=True,
                ),
                f"{int(result['wins'])}/{int(result['n_backbones'])}",
                float(result["cohen_dz"]),
                float(result[PRIMARY_P_VALUE]),
            ]
            row_index = data_start + row_offset
            for column, value in enumerate(values, start=1):
                cell = worksheet.cell(row_index, column, value)
                cell.font = Font(name="Arial", size=9)
                cell.alignment = Alignment(
                    horizontal="left" if column in (1, 2) else "center",
                    vertical="center",
                )
                if column in (3, 4, 7, 8):
                    cell.number_format = "0.0000"
                if column == 8 and float(value) < 0.05:
                    cell.font = Font(name="Arial", size=9, bold=True)
            worksheet.row_dimensions[row_index].height = 19

        data_end = data_start + len(ordered_rows) - 1
        for cell in worksheet[data_end]:
            cell.border = Border(bottom=medium_black)

        difference_note = (
            f"For Integrated Brier score, differences are comparator minus {TARGET_METHOD} "
            "because lower values are favorable."
            if metric == "integrated_brier_score"
            else f"Differences are {TARGET_METHOD} minus comparator."
        )
        notes = [
            "Each row is a separate prespecified one-to-one paired hypothesis. "
            f"{difference_note} Positive values favor {TARGET_METHOD}.",
            "P values are unadjusted, two-sided exact paired sign-flip tests of the studentized mean difference and must be interpreted comparison-wise.",
            "Values are mean paired differences with percentile-bootstrap 95% confidence intervals. Statistical unit: backbone; ID, in-center; OOD, held-out-center.",
            "Multiplicity-adjusted common-control maxT results are retained in Supplementary as a sensitivity analysis.",
        ]
        note_start = data_end + 2
        for offset, note in enumerate(notes):
            row_index = note_start + offset
            worksheet.merge_cells(start_row=row_index, start_column=1, end_row=row_index, end_column=8)
            cell = worksheet.cell(row_index, 1, note)
            cell.font = Font(name="Arial", size=8, italic=True)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            worksheet.row_dimensions[row_index].height = 22

        widths = {
            "A": 10,
            "B": 30,
            "C": 13,
            "D": 16,
            "E": 31,
            "F": 11,
            "G": 12,
            "H": 11,
        }
        for column, width in widths.items():
            worksheet.column_dimensions[column].width = width
        worksheet.freeze_panes = "A3"
        worksheet.print_title_rows = "1:2"
        configure_print(worksheet, note_start + len(notes) - 1, 8)
        return worksheet

    readme_rows = [
        ("Workbook purpose", "Manuscript-facing and supplementary TCGA-BRCA WSI survival results"),
        ("Sheet hierarchy", "MAIN_* sheets are intended for the manuscript; SUPP_* sheets contain detailed evidence and provenance"),
        ("Primary statistical unit", "Backbone"),
        ("Primary pairing", "Common complete backbones within a frozen domain and metric"),
        ("Primary endpoint", "Uno C-index, reported separately for ID and OOD"),
        ("ID", "in_center"),
        ("OOD", "held_out_center"),
        ("Confidence interval", f"Percentile bootstrap across backbones; {BOOTSTRAP_REPLICATES:,} repetitions"),
        ("Main inference", "Separate prespecified one-to-one paired contrasts; two-sided unadjusted exact sign-flip P values"),
        ("Multiplicity sensitivity", "Common-control studentized maxT adjustment is reported in Supplementary only"),
        ("Effect direction", f"Positive always favors {TARGET_METHOD}; IBS is calculated as control minus {TARGET_METHOD}"),
        ("MAIN_01_Uno", "Absolute ID/OOD Uno C-index means and 95% CIs on the common primary-method backbone set"),
        (UNO_CONTRAST_SHEET, f"Comparison-wise {TARGET_METHOD} versus prespecified comparators for ID and OOD Uno C-index"),
        ("MAIN_03_Harrell_Contrasts", f"Comparison-wise {TARGET_METHOD} versus prespecified comparators for ID and OOD Harrell C-index"),
        ("MAIN_04_IBS_Contrasts", f"Comparison-wise {TARGET_METHOD} versus prespecified comparators for ID and OOD Integrated Brier score"),
        ("MAIN_05_tdAUC_Contrasts", f"Comparison-wise {TARGET_METHOD} versus prespecified comparators for ID and OOD mean time-dependent AUC"),
        ("SUPP_01_All_Endpoints", "Absolute results for every available method, domain, and endpoint"),
        ("SUPP_02_All_Pairwise", f"All separate {TARGET_METHOD}-focused endpoint contrasts, CIs, effect sizes, and unadjusted P values"),
        ("SUPP_03_MaxT_Sensitivity", "Multiplicity-adjusted common-control maxT sensitivity results"),
        ("SUPP_04_Paired_Summary", "Common-backbone primary-method summary driving MAIN_01_Uno"),
        ("SUPP_05_Coverage", "Available backbone count for every method/domain/metric"),
        ("SUPP_06_Backbones", "Requested backbone selection and canonical availability"),
        ("SUPP_07_Provenance", "Exact source files, byte sizes, and SHA-256"),
        ("SUPP_08_Analysis_Long", "Validated long-form analysis data including derived mean time-dependent AUC"),
    ]

    workbook = Workbook()
    readme = workbook.active
    readme.title = "README"
    readme.sheet_view.showGridLines = True
    readme.sheet_view.zoomScale = 90
    readme.sheet_properties.tabColor = "000000"
    readme.append(["Section", "Definition"])
    for row in readme_rows:
        readme.append(row)
    for cell in readme[1]:
        cell.fill = header_fill
        cell.font = Font(name="Arial", size=9, bold=True, color="000000")
        cell.border = Border(top=thin_black, bottom=thin_black)
    for row in readme.iter_rows(min_row=2):
        row[0].font = Font(name="Arial", size=9, bold=True)
        row[1].font = Font(name="Arial", size=9)
        row[1].alignment = Alignment(wrap_text=True, vertical="top")
    readme.column_dimensions["A"].width = 30
    readme.column_dimensions["B"].width = 105
    readme.freeze_panes = "A2"
    configure_print(readme, readme.max_row, 2)
    readme.print_title_rows = "1:1"

    add_main_absolute_sheet(workbook)
    add_main_contrast_sheet(
        workbook,
        sheet_name=UNO_CONTRAST_SHEET,
        table_number=2,
        metric="uno_c_index",
        metric_label="Uno C-index",
    )
    add_main_contrast_sheet(
        workbook,
        sheet_name="MAIN_03_Harrell_Contrasts",
        table_number=3,
        metric="harrell_c_index",
        metric_label="Harrell C-index",
    )
    add_main_contrast_sheet(
        workbook,
        sheet_name="MAIN_04_IBS_Contrasts",
        table_number=4,
        metric="integrated_brier_score",
        metric_label="Integrated Brier score",
    )
    add_main_contrast_sheet(
        workbook,
        sheet_name="MAIN_05_tdAUC_Contrasts",
        table_number=5,
        metric="td_auc_mean",
        metric_label="Mean time-dependent AUC",
    )
    add_supplementary_sheet(workbook, "SUPP_01_All_Endpoints", broad_summary_frame)
    add_supplementary_sheet(workbook, "SUPP_02_All_Pairwise", pairwise_contrast_frame)
    add_supplementary_sheet(workbook, "SUPP_03_MaxT_Sensitivity", multiplicity_sensitivity_frame)
    add_supplementary_sheet(workbook, "SUPP_04_Paired_Summary", paired_summary_frame)
    add_supplementary_sheet(workbook, "SUPP_05_Coverage", coverage_frame)
    add_supplementary_sheet(workbook, "SUPP_06_Backbones", backbone_frame)
    add_supplementary_sheet(workbook, "SUPP_07_Provenance", provenance_frame)
    add_supplementary_sheet(workbook, "SUPP_08_Analysis_Long", analysis_frame)

    workbook.properties.title = "TCGA-BRCA WSI survival manuscript results"
    workbook.properties.subject = f"{TARGET_METHOD} survival analysis"
    workbook.properties.creator = f"{TARGET_METHOD} reporting pipeline"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)
    return path


# %% 7. RESOLVE THE USER SELECTION
def configure_branch(branch: str) -> None:
    """Select branch-specific names and comparators without reading study data."""
    global TARGET_METHOD, MAIN_COMPARATORS, CONTEXTUAL_COMPARATORS
    global PAIRWISE_COMPARATORS, MANUSCRIPT_METHODS, MAX_T_SENSITIVITY_CONTROLS
    global WORKBOOK_FILENAME, OUTPUT_DIR, UNO_CONTRAST_SHEET
    if branch not in {"raw", "reinhard"}:
        raise ValueError("branch must be raw or reinhard")
    normalized = branch == "reinhard"
    stem = "reinhard_delta_v1_3_p2" if normalized else "delta_v1_3_p2"
    TARGET_METHOD = "Reinhard-DELTA-v1.3" if normalized else "DELTA-v1.3"
    MAIN_COMPARATORS = ["RAW", "DANN", "Reinhard-RAW", "Reinhard-DANN"]
    if normalized:
        MAIN_COMPARATORS.append("DELTA-v1.3")
    CONTEXTUAL_COMPARATORS = [
        "ComBat-inductive", "ComBat-transductive",
        "Reinhard-ComBat-inductive", "Reinhard-ComBat-transductive",
    ]
    if not normalized:
        CONTEXTUAL_COMPARATORS.insert(0, "Reinhard-DELTA-v1.3")
    PAIRWISE_COMPARATORS = [*MAIN_COMPARATORS, *CONTEXTUAL_COMPARATORS]
    MANUSCRIPT_METHODS = [*MAIN_COMPARATORS, TARGET_METHOD]
    if not normalized:
        MANUSCRIPT_METHODS.append("Reinhard-DELTA-v1.3")
    MAX_T_SENSITIVITY_CONTROLS = list(MAIN_COMPARATORS)
    UNO_CONTRAST_SHEET = "MAIN_02_ReDELTA_Contrasts" if normalized else "MAIN_02_DELTA_Contrasts"
    WORKBOOK_FILENAME = "survival_manuscript_results_" + stem + ".xlsx"
    OUTPUT_DIR = Path(os.environ.get(
        "DELTA_SURVIVAL_REPORT_OUTPUT_DIR",
        PROJECT_ROOT / ("results/tcga_brca_wsi_survival_report_" + stem + "/backbones20"),
    )).expanduser().resolve()


def run_report() -> dict[str, object]:
    """Run the frozen analysis after branch and path configuration."""
    selected_backbone_ids, selected_backbones = resolve_selected_backbones()
    print("Selected backbones:")
    print(selected_backbones.to_string(index=False))


    # %% 8. LOAD AND VALIDATE CANONICAL CONTROLS + DELTA-v1.3 p2
    control_long, control_provenance = load_survival_results(
        RESULTS_ROOTS,
        include_backbones=selected_backbone_ids,
    )
    v13_long, v13_provenance = load_v13_p2_results(
        V13_EXPERIMENT_ROOT,
        include_backbones=selected_backbone_ids,
    )
    analysis_long = pd.concat([control_long, v13_long], ignore_index=True)
    input_provenance = pd.concat(
        [control_provenance, v13_provenance], ignore_index=True, sort=False
    )

    split_ids = sorted(analysis_long["split_id"].astype(str).unique().tolist())
    if len(split_ids) != 1:
        raise ValueError(f"Controls and v1.3 must use one identical split: {split_ids}")
    analysis_key = ["backbone", "split_id", "method", "domain", "metric"]
    if analysis_long.duplicated(analysis_key).any():
        duplicated = analysis_long.loc[
            analysis_long.duplicated(analysis_key, keep=False), analysis_key
        ]
        raise ValueError(
            "Conflicting canonical sources after v1.3 merge: "
            + str(duplicated.head().to_dict("records"))
        )
    analysis_long = analysis_long.sort_values(analysis_key).reset_index(drop=True)

    required_v13_methods = {"DELTA-v1.3", "Reinhard-DELTA-v1.3"}
    v13_method_sets = [
        set(
            analysis_long.loc[
                analysis_long["method"].eq(method), "backbone"
            ].astype(str)
        )
        for method in sorted(required_v13_methods)
    ]
    observed_backbone_ids = set.intersection(*v13_method_sets)
    missing_backbone_ids = [
        backbone for backbone in selected_backbone_ids if backbone not in observed_backbone_ids
    ]
    if missing_backbone_ids:
        missing_readable = selected_backbones[
            selected_backbones["backbone"].isin(missing_backbone_ids)
        ]["backbone_display"].tolist()
        message = (
            "Selected backbones without both official v1.3 p2 branches: "
            f"{missing_readable} ({missing_backbone_ids})"
        )
        if MISSING_BACKBONE_POLICY == "error":
            raise ValueError(message)
        if MISSING_BACKBONE_POLICY != "drop":
            raise ValueError("MISSING_BACKBONE_POLICY must be 'error' or 'drop'")
        print("[INFO]", message)

    selected_backbones["canonical_result_present"] = selected_backbones["backbone"].isin(
        observed_backbone_ids
    )


    # %% 9. BUILD THE ANALYSIS TABLES
    broad_method_summary = summarize_methods(
        analysis_long,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        seed=RANDOM_SEED,
    )

    primary_paired_summary = summarize_common_backbones(
        analysis_long,
        methods=MANUSCRIPT_METHODS,
        metrics=("uno_c_index",),
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        seed=RANDOM_SEED,
    )

    all_endpoint_paired_summary = summarize_common_backbones(
        analysis_long,
        methods=MANUSCRIPT_METHODS,
        metrics=tuple(HIGHER_IS_BETTER),
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        seed=RANDOM_SEED,
    )

    delta_pairwise_contrasts = compare_target_pairwise(
        analysis_long,
        target=TARGET_METHOD,
        controls=PAIRWISE_COMPARATORS,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        permutations=MONTE_CARLO_PERMUTATIONS,
        exact_max_n=EXACT_SIGN_FLIP_MAX_BACKBONES,
        seed=RANDOM_SEED,
    )

    if BUILD_MULTIPLICITY_SENSITIVITY:
        delta_max_t_sensitivity = compare_target(
            analysis_long,
            target=TARGET_METHOD,
            controls=MAX_T_SENSITIVITY_CONTROLS,
            bootstrap_replicates=BOOTSTRAP_REPLICATES,
            permutations=MONTE_CARLO_PERMUTATIONS,
            exact_max_n=EXACT_SIGN_FLIP_MAX_BACKBONES,
            seed=RANDOM_SEED,
        )
    else:
        delta_max_t_sensitivity = pd.DataFrame()

    main_table = build_main_table(
        primary_paired_summary,
        methods=MANUSCRIPT_METHODS,
    )

    coverage = (
        analysis_long[analysis_long["metric"].isin(HIGHER_IS_BETTER)]
        .groupby(["method", "domain", "metric"], sort=True)["backbone"]
        .nunique()
        .rename("n_backbones")
        .reset_index()
    )


    # %% 10. INSPECT RESULTS BEFORE SAVING
    show_compact_results(main_table, delta_pairwise_contrasts)

    # Useful examples for line-by-line inspection:
    id_uno = broad_method_summary[
        broad_method_summary["domain"].eq("in_center")
        & broad_method_summary["metric"].eq("uno_c_index")
    ].copy()
    ood_uno = broad_method_summary[
        broad_method_summary["domain"].eq("held_out_center")
        & broad_method_summary["metric"].eq("uno_c_index")
    ].copy()
    ood_delta_contrasts = delta_pairwise_contrasts[
        delta_pairwise_contrasts["domain"].eq("held_out_center")
    ].copy()


    # %% 11. SAVE TABLES AND ANALYSIS LINEAGE
    output_paths: dict[str, Path] = {}
    if SAVE_OUTPUTS:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_paths = {
            "workbook": OUTPUT_DIR / WORKBOOK_FILENAME,
            "manifest": OUTPUT_DIR / "survival_report_manifest.json",
        }
        write_manuscript_workbook(
            output_paths["workbook"],
            main_table_frame=main_table,
            pairwise_contrast_frame=delta_pairwise_contrasts,
            multiplicity_sensitivity_frame=delta_max_t_sensitivity,
            broad_summary_frame=broad_method_summary,
            paired_summary_frame=primary_paired_summary,
            coverage_frame=coverage,
            backbone_frame=selected_backbones,
            provenance_frame=input_provenance,
            analysis_frame=analysis_long,
        )

        if SAVE_INDIVIDUAL_CSV:
            csv_dir = OUTPUT_DIR / "machine_readable"
            csv_dir.mkdir(parents=True, exist_ok=True)
            csv_tables = {
                "selected_backbones": selected_backbones,
                "analysis_long": analysis_long,
                "input_provenance": input_provenance,
                "coverage": coverage,
                "method_summary": broad_method_summary,
                "primary_paired_summary": primary_paired_summary,
                "all_endpoint_paired_summary": all_endpoint_paired_summary,
                "delta_pairwise_contrasts": delta_pairwise_contrasts,
                "delta_max_t_sensitivity": delta_max_t_sensitivity,
                "main_table": main_table,
            }
            for name, frame in csv_tables.items():
                frame.to_csv(csv_dir / f"{name}.csv", index=False)

        report_manifest = {
            "schema_version": 1,
            "results_roots": [str(path.resolve()) for path in RESULTS_ROOTS],
            "v1_3_experiment_root": str(V13_EXPERIMENT_ROOT.resolve()),
            "v1_3_bank_fraction_percent_per_role": 2,
            "v1_3_pairs_per_branch": 55_988,
            "v1_3_rank_input_rows": 55_988,
            "v1_3_fixed_rank_row_cap": None,
            "v1_3_rank_rule": "null_adjusted_90pct",
            "v1_3_cap_free": True,
            "v1_3_projection": V13_PROJECTION,
            "v1_3_universal_euclidean": True,
            "v1_3_method_contract_sha256": V13_METHOD_CONTRACT_SHA256,
            "selected_backbones": selected_backbones.to_dict("records"),
            "observed_backbones": sorted(observed_backbone_ids),
            "target": TARGET_METHOD,
            "manuscript_methods": MANUSCRIPT_METHODS,
            "main_comparators": MAIN_COMPARATORS,
            "contextual_comparators": CONTEXTUAL_COMPARATORS,
            "pairwise_comparators": PAIRWISE_COMPARATORS,
            "result_filenames": list(RESULT_FILENAMES),
            "forbidden_legacy_methods": sorted(FORBIDDEN_LEGACY_METHODS),
            "max_t_sensitivity_controls": MAX_T_SENSITIVITY_CONTROLS,
            "primary_p_value": PRIMARY_P_VALUE,
            "report_unadjusted_p": REPORT_UNADJUSTED_P,
            "build_multiplicity_sensitivity": BUILD_MULTIPLICITY_SENSITIVITY,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "monte_carlo_permutations_when_not_exact": MONTE_CARLO_PERMUTATIONS,
            "exact_sign_flip_max_backbones": EXACT_SIGN_FLIP_MAX_BACKBONES,
            "random_seed": RANDOM_SEED,
            "statistical_unit": "backbone",
            "pairing": "separate complete paired-backbone set within each target-versus-control contrast",
            "main_inference": "separate prespecified one-to-one paired contrasts; unadjusted two-sided P values",
            "multiplicity_statement": "common-control maxT is supplementary sensitivity only",
            "effect_direction": f"positive favors {TARGET_METHOD}; IBS is control minus {TARGET_METHOD}",
            "domain_labels": DOMAIN_LABELS,
            "outputs": {key: str(path.resolve()) for key, path in output_paths.items()},
        }
        output_paths["manifest"].write_text(
            json.dumps(report_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("\nSaved consolidated results:")
        for name, path in output_paths.items():
            print(f"  {name}: {path}")


    return {"output_paths": output_paths}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=("raw", "reinhard"), required=True)
    parser.add_argument("--control-results-root", type=Path)
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--csv", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    configure_branch(args.branch)
    global RESULTS_ROOTS, V13_EXPERIMENT_ROOT, OUTPUT_DIR, SAVE_INDIVIDUAL_CSV
    if args.control_results_root is not None:
        RESULTS_ROOTS = (args.control_results_root.expanduser().resolve(),)
    if args.experiment_root is not None:
        V13_EXPERIMENT_ROOT = args.experiment_root.expanduser().resolve()
    if args.output_dir is not None:
        OUTPUT_DIR = args.output_dir.expanduser().resolve()
    SAVE_INDIVIDUAL_CSV = args.csv
    run_report()


if __name__ == "__main__":
    main()
