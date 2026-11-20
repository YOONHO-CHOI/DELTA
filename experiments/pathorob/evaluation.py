"""PathoROB representation and supervised ROI-prediction evaluation.

The functions in this module operate on user-provided feature arrays aligned
to explicitly supplied cohort metadata. Experiment metadata are not bundled.
They do not download images or model weights and do not fit DELTA.  DELTA must
already have been fitted under the appropriate full-cohort (representation)
or split-local train-only (ROI prediction) information contract.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from experiments.common.splits import (
    APD_SPECS,
    REPETITION_SEEDS,
    APDSplit,
    build_official_apd_splits,
    prepare_apd_metadata,
)


REPRESENTATION_ENDPOINTS = (
    "robustness_index",
    "biological_knn",
    "confounder_insensitivity",
    "clustering_score",
)


@dataclass(frozen=True)
class RepresentationResult:
    robustness_index: float
    biological_knn: float
    confounder_insensitivity: float
    clustering_score: float
    selected_k: int
    robustness_applicable: bool
    clustering_applicable: bool


def _official_dataset_name(dataset: str) -> str:
    return "tcga" if dataset.startswith("tcga_") else dataset


def export_official_features(
    features: np.ndarray,
    metadata: pd.DataFrame,
    model_name: str,
    dataset: str,
    output_root: Path,
) -> Path:
    """Export one aligned array in PathoROB's center-wise NPZ format."""

    values = np.asarray(features, dtype=np.float32)
    frame = metadata.reset_index(drop=True).copy()
    if values.ndim != 2 or len(values) != len(frame):
        raise ValueError("features must be a matrix aligned to metadata rows")
    required = {"slide_id", "patch_id", "medical_center"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing PathoROB metadata columns: {sorted(missing)}")
    frame["feature_id"] = frame["slide_id"].astype(str) + "-" + frame["patch_id"].astype(str)
    duplicated = frame["feature_id"].duplicated(keep=False)
    if duplicated.any():
        for feature_id, group in frame.loc[duplicated].groupby("feature_id", sort=False):
            positions = group.index.to_numpy(dtype=np.int64)
            if group["medical_center"].astype(str).nunique() != 1:
                raise ValueError(f"duplicated patch has multiple centers: {feature_id}")
            if not np.array_equal(
                values[positions], np.broadcast_to(values[positions[0]], values[positions].shape)
            ):
                raise ValueError(f"duplicated patch has inconsistent features: {feature_id}")

    output = Path(output_root) / model_name / _official_dataset_name(dataset)
    output.mkdir(parents=True, exist_ok=True)
    for center, group in frame.groupby("medical_center", sort=True):
        positions = group.index.to_numpy(dtype=np.int64)
        payload = {
            str(feature_id): values[position]
            for feature_id, position in zip(group["feature_id"], positions)
        }
        np.savez(output / f"{str(center).replace(' ', '_')}.npz", **payload)
    manifest = {
        "schema": "delta-pathorob-feature-export-v1",
        "model": model_name,
        "dataset": _official_dataset_name(dataset),
        "rows": int(len(frame)),
        "unique_patches": int(frame["feature_id"].nunique()),
        "dimension": int(values.shape[1]),
        "centers": sorted(frame["medical_center"].astype(str).unique().tolist()),
    }
    (output / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output


def _load_official_metrics(pathorob_root: Path):
    root = str(Path(pathorob_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    robustness = importlib.import_module(
        "pathorob.robustness_index.robustness_index"
    ).compute
    clustering = importlib.import_module(
        "pathorob.clustering_score.clustering_score"
    ).compute
    return robustness, clustering


def _run_representation_pass(
    features: np.ndarray,
    metadata: pd.DataFrame,
    *,
    dataset: str,
    feature_id: str,
    output_dir: Path,
    metadata_dir: Path,
    pathorob_root: Path,
    run_robustness: bool,
    run_clustering: bool,
    dry_run: bool,
) -> dict[str, object]:
    model_name = f"delta__{feature_id}"
    output = Path(output_dir)
    features_root = output / "official_features"
    export_official_features(features, metadata, model_name, dataset, features_root)
    compute_robustness, compute_clustering = _load_official_metrics(pathorob_root)
    official_dataset = _official_dataset_name(dataset)
    robustness_summary: dict[str, object] = {}
    clustering_summary: dict[str, object] = {}
    if run_robustness:
        result_root = output / "robustness_index"
        compute_robustness(
            model=model_name,
            dataset=official_dataset,
            features_dir=str(features_root),
            metadata_dir=str(metadata_dir),
            results_dir=str(result_root),
            paired_evaluation=False,
            k_opt_param=0,
            max_patches_per_combi=-1,
            num_workers=1 if dry_run else 8,
            plot_graphs=False,
            debug_mode=bool(dry_run),
        )
        summary = result_root / model_name / official_dataset / "-1_0" / "results_summary.json"
        robustness_summary = json.loads(summary.read_text(encoding="utf-8"))
    if run_clustering:
        clustering_summary = compute_clustering(
            model=model_name,
            dataset=official_dataset,
            features_dir=str(features_root),
            metadata_dir=str(metadata_dir),
            results_dir=str(output / "clustering_score"),
            minK=2,
            maxK=3 if dry_run else 30,
            num_trials=2 if dry_run else 50,
            overwrite_results=False,
        )
    return {
        "robustness_index": float(robustness_summary.get("robustness_index", np.nan)),
        "biological_knn": float(robustness_summary.get("balanced_accuracy", np.nan)),
        "confounder_insensitivity": float(
            robustness_summary.get("confounder_insensitivity", np.nan)
        ),
        "clustering_score": float(clustering_summary.get("clustering_score_mean", np.nan)),
        "selected_k": int(
            robustness_summary.get("k_opt", clustering_summary.get("Ks", {}).get("median", -1))
        ),
    }


def evaluate_representation(
    features: np.ndarray,
    metadata: pd.DataFrame,
    *,
    dataset: str,
    feature_id: str,
    output_dir: Path,
    metadata_dir: Path,
    pathorob_root: Path,
    dry_run: bool = False,
) -> RepresentationResult:
    """Run the official PathoROB representation suite.

    TCGA4 needs two official passes: clustering under the standard TCGA4
    contract and robustness metrics with unpaired evaluation.  The two passes
    are merged without changing either upstream metric implementation.
    """

    if dataset not in {"camelyon", "tcga_4x4", "tolkach_esca", "tcga_2x2"}:
        raise ValueError(f"unsupported PathoROB dataset: {dataset}")
    standard = _run_representation_pass(
        features,
        metadata,
        dataset=dataset,
        feature_id=feature_id,
        output_dir=Path(output_dir) / "standard",
        metadata_dir=metadata_dir,
        pathorob_root=pathorob_root,
        run_robustness=dataset != "tcga_4x4",
        run_clustering=dataset != "tcga_2x2",
        dry_run=dry_run,
    )
    if dataset == "tcga_4x4":
        extension = _run_representation_pass(
            features,
            metadata,
            dataset=dataset,
            feature_id=feature_id + "__robustness",
            output_dir=Path(output_dir) / "tcga4_robustness",
            metadata_dir=metadata_dir,
            pathorob_root=pathorob_root,
            run_robustness=True,
            run_clustering=False,
            dry_run=dry_run,
        )
        for endpoint in ("robustness_index", "biological_knn", "confounder_insensitivity"):
            standard[endpoint] = extension[endpoint]
        standard["selected_k"] = extension["selected_k"]
    result = RepresentationResult(
        robustness_index=float(standard["robustness_index"]),
        biological_knn=float(standard["biological_knn"]),
        confounder_insensitivity=float(standard["confounder_insensitivity"]),
        clustering_score=float(standard["clustering_score"]),
        selected_k=int(standard["selected_k"]),
        robustness_applicable=dataset != "tcga_4x4" or np.isfinite(standard["robustness_index"]),
        clustering_applicable=dataset != "tcga_2x2",
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "representation_summary.json").write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def _rows(features: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    positions = frame["row_position"].to_numpy(dtype=np.int64)
    if len(positions) and (positions.min() < 0 or positions.max() >= len(features)):
        raise ValueError("split row_position exceeds the feature array")
    return np.asarray(features[positions], dtype=np.float32)


def evaluate_roi_split(
    features: np.ndarray,
    split: APDSplit,
    *,
    workers: int = 1,
) -> dict[str, float]:
    """Run the exact PathoROB shallow-classifier C grid for one APD split."""

    train_x = _rows(features, split.train)
    validation_x = _rows(features, split.validation)
    id_x = _rows(features, split.id_test)
    ood_x = _rows(features, split.ood_test)
    train_y = split.train["biological_class"].to_numpy()
    validation_y = split.validation["biological_class"].to_numpy()
    id_y = split.id_test["biological_class"].to_numpy()
    ood_y = split.ood_test["biological_class"].to_numpy()
    cs = 10 ** np.linspace(-8, 4, 15)

    def validation_score(c_value: float) -> float:
        # Match the pinned PathoROB implementation exactly.  The upstream
        # validation models fix ``random_state=0`` and otherwise retain the
        # scikit-learn defaults, including ``max_iter=100``.
        model = LogisticRegression(C=float(c_value), random_state=0)
        model.fit(train_x, train_y)
        return float(balanced_accuracy_score(validation_y, model.predict(validation_x)))

    n_workers = max(1, min(int(workers), len(cs)))
    if n_workers == 1:
        validation_scores = [validation_score(value) for value in cs]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            validation_scores = list(pool.map(validation_score, cs))
    best_c = float(cs[int(np.argmax(validation_scores))])
    # The final upstream fit specifies only C; do not introduce a repetition
    # seed or a different iteration budget here.
    model = LogisticRegression(C=best_c)
    model.fit(train_x, train_y)
    return {
        "best_c": best_c,
        "id_balanced_accuracy": float(balanced_accuracy_score(id_y, model.predict(id_x))),
        "ood_balanced_accuracy": float(balanced_accuracy_score(ood_y, model.predict(ood_x))),
    }


def evaluate_roi_prediction(
    features: np.ndarray,
    metadata: pd.DataFrame,
    *,
    dataset: str,
    method: str,
    repetitions: int = 20,
    workers: int = 1,
    tolkach_split_resource: Path | None = None,
) -> pd.DataFrame:
    """Evaluate one split-independent feature condition on all APD splits.

    This is appropriate for RAW and other operators that do not depend on an
    APD training partition. Split-local DELTA must use
    :func:`evaluate_roi_prediction_inventory` instead.
    """

    values = np.asarray(features)
    frame = prepare_apd_metadata(metadata)
    if values.ndim != 2 or len(values) != len(frame):
        raise ValueError("features must be a matrix aligned to metadata rows")
    return _evaluate_roi_prediction(
        lambda _split_id: values,
        frame,
        dataset=dataset,
        method=method,
        repetitions=repetitions,
        workers=workers,
        tolkach_split_resource=tolkach_split_resource,
    )


def evaluate_roi_prediction_inventory(
    feature_paths: Mapping[str, Path],
    metadata: pd.DataFrame,
    *,
    dataset: str,
    method: str,
    repetitions: int = 20,
    workers: int = 1,
    tolkach_split_resource: Path | None = None,
) -> pd.DataFrame:
    """Evaluate split-local DELTA arrays keyed by exact official split ID."""

    paths = {str(key): Path(value) for key, value in feature_paths.items()}
    if not paths:
        raise ValueError("feature inventory is empty")
    used: set[str] = set()

    def load(split_id: str) -> np.ndarray:
        if split_id not in paths:
            raise ValueError(f"missing split-local feature array: {split_id}")
        used.add(split_id)
        return np.load(paths[split_id], mmap_mode="r", allow_pickle=False)

    result = _evaluate_roi_prediction(
        load,
        prepare_apd_metadata(metadata),
        dataset=dataset,
        method=method,
        repetitions=repetitions,
        workers=workers,
        tolkach_split_resource=tolkach_split_resource,
    )
    unused = set(paths).difference(used)
    if unused and repetitions == len(REPETITION_SEEDS):
        raise ValueError(f"unexpected split IDs in feature inventory: {sorted(unused)[:5]}")
    return result


def _evaluate_roi_prediction(
    feature_resolver: Callable[[str], np.ndarray],
    frame: pd.DataFrame,
    *,
    dataset: str,
    method: str,
    repetitions: int,
    workers: int,
    tolkach_split_resource: Path | None,
) -> pd.DataFrame:
    """Shared APD loop with an explicit per-split feature resolver."""

    if dataset not in APD_SPECS:
        raise ValueError(f"unsupported PathoROB ROI dataset: {dataset}")
    if not 1 <= int(repetitions) <= len(REPETITION_SEEDS):
        raise ValueError(f"repetitions must be in [1, {len(REPETITION_SEEDS)}]")
    rows: list[dict[str, object]] = []
    split_index = 0
    for repetition, seed in enumerate(REPETITION_SEEDS[: int(repetitions)]):
        splits = build_official_apd_splits(
            frame,
            dataset,
            int(seed),
            tolkach_split_resource=tolkach_split_resource,
        )
        for split in splits:
            values = np.asarray(feature_resolver(split.split_id))
            if values.ndim != 2 or len(values) != len(frame):
                raise ValueError(
                    f"features for {split.split_id} must align to all metadata rows"
                )
            result = evaluate_roi_split(values, split, workers=workers)
            for domain in ("ID", "OOD"):
                rows.append(
                    {
                        "evaluation": "supervised_roi_prediction",
                        "dataset": dataset,
                        "method": method,
                        "repetition": repetition,
                        "repetition_seed": int(seed),
                        "split_index": split_index,
                        "split_order": int(split.split_order),
                        "split_id": split.split_id,
                        "domain": domain,
                        "balanced_accuracy": float(
                            result[f"{domain.lower()}_balanced_accuracy"]
                        ),
                        "best_c": float(result["best_c"]),
                    }
                )
            split_index += 1
    return pd.DataFrame(rows)


def average_performance_drop(
    scores: pd.DataFrame, *, reference_scores: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Compute robustification APD against a common RAW balanced-split mean.

    PathoROB Eq. 8 uses ``mean(score_step[1:] / mean_RAW_step0) - 1``.
    The denominator is averaged over RAW repetitions separately for each
    dataset/domain, not taken from the evaluated method's own split 0.
    Supply RAW rows in ``scores`` or via ``reference_scores``. This function
    evaluates one encoder; references must come from that same encoder.
    """

    required = {
        "dataset", "method", "domain", "repetition_seed", "split_order",
        "balanced_accuracy",
    }
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"missing ROI result columns: {sorted(missing)}")
    raw = scores if reference_scores is None else reference_scores
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"missing RAW reference columns: {sorted(missing)}")
    raw = raw.loc[raw['method'].eq('RAW')].copy()
    if raw.empty:
        raise ValueError('APD requires RAW scores; provide reference_scores explicitly')
    if not np.isfinite(scores['balanced_accuracy']).all() or not np.isfinite(raw['balanced_accuracy']).all():
        raise ValueError('APD scores and RAW references must be finite')
    identity = ['dataset','method','domain','repetition_seed','split_order']
    if scores.duplicated(identity).any() or raw.duplicated(identity).any():
        raise ValueError('duplicate APD score/reference keys')
    references = raw.loc[raw['split_order'].eq(0)].groupby(
        ['dataset','domain'])['balanced_accuracy'].mean()
    if references.empty or references.le(0).any():
        raise ValueError('RAW balanced-split APD references must be positive')
    rows: list[dict[str, object]] = []
    keys = ["dataset", "method", "domain", "repetition_seed"]
    for key, group in scores.groupby(keys, sort=False):
        dataset, method, domain, seed = key
        values = group.groupby("split_order")["balanced_accuracy"].mean().sort_index()
        expected = set(range(int(APD_SPECS[str(dataset)]["n_splits"])))
        if set(values.index) != expected:
            raise ValueError(f"incomplete APD curve: {key}")
        reference_key = (dataset,domain)
        if reference_key not in references.index:
            raise ValueError(f'missing RAW balanced-split reference: {reference_key}')
        reference = float(references.loc[reference_key])
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "domain": domain,
                "repetition_seed": int(seed),
                "average_performance_drop": float((values.loc[1:] / reference).mean() - 1.0),
            }
        )
    return pd.DataFrame(rows)


def thread_count_from_environment(default: int = 1) -> int:
    """Return a bounded classifier worker count for command-line runners."""

    return max(1, int(os.environ.get("DELTA_PATHOROB_CPU_WORKERS", str(default))))
