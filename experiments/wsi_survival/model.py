#!/usr/bin/env python3
"""Locked patient-level Cox/DANN evaluation for BRCA full-WSI aggregates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(
    os.environ.get(
        "DELTA_PROJECT_ROOT",
        REPOSITORY_ROOT,
    )
).expanduser().resolve()
EXPERIMENT = ROOT / (
    "experiments/tcga_brca_full-wsi_survival__20backbone__"
    "delta-v1-3-p2__null-adjusted90__v001"
)
IMPLEMENTATION_FREEZE = (
    EXPERIMENT / "controls/baseline_protocol/survival_implementation_freeze_v1.json"
)
BACKBONES = (
    "ciga", "retccl", "ctranspath", "hipt", "kaiko_vitb8", "virchow",
    "beph", "uni", "phikonv2", "virchow2", "conch", "conchv15",
    "uni2h", "hoptimus0", "h0mini", "exaonepath25", "gpfm",
    "moozy_patch", "gigapath", "musk",
)
WSI_AGGREGATE_COLLECTION = "full-wsi-patient-aggregate-v1"
SEEDS = (7028, 17028, 27028)
EVALUATION_RULE = "cox_train_reference__km_left_limit"
STOP_REQUESTED = False


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def handle_signal(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    log(f"STOP_REQUESTED signal={signum}")


from .metrics import (
    censoring_km, pairwise_concordance, harrell_c, uno_c,
    breslow_baseline, survival_probability, brier_score_ipcw, time_dependent_auc,
)

from experiments.baselines import combat_inductive
from experiments.baselines import combat as combat_transductive


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, weight: float) -> torch.Tensor:
        ctx.weight = float(weight)
        return tensor.view_as(tensor)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.weight * gradient, None


class CoxHead(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.20),
        )
        self.risk = torch.nn.Linear(256, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.risk(self.encoder(values)).squeeze(1)


class DANNCoxHead(torch.nn.Module):
    def __init__(self, input_dim: int, domains: int):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.20),
        )
        self.risk = torch.nn.Linear(256, 1)
        self.domain = torch.nn.Sequential(
            torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Linear(128, domains)
        )

    def forward(
        self, values: torch.Tensor, lambda_weight: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(values)
        risk = self.risk(encoded).squeeze(1)
        domain = self.domain(GradientReverse.apply(encoded, lambda_weight))
        return risk, domain

    def predict(self, values: torch.Tensor) -> torch.Tensor:
        return self.risk(self.encoder(values)).squeeze(1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def cox_efron_loss(
    risk: torch.Tensor, time_values: torch.Tensor, events: torch.Tensor
) -> torch.Tensor:
    event_times = torch.unique(time_values[events > 0.5])
    if len(event_times) == 0:
        raise ValueError("Cox training partition has no events")
    log_likelihood = risk.new_zeros(())
    event_count = 0
    for current_time in event_times:
        deaths = (time_values == current_time) & (events > 0.5)
        risk_set = time_values >= current_time
        death_risk = risk[deaths]
        risk_values = risk[risk_set]
        maximum = torch.max(risk_values)
        denominator = torch.exp(risk_values - maximum).sum()
        tied = torch.exp(death_risk - maximum).sum()
        count = int(deaths.sum().item())
        log_likelihood = log_likelihood + death_risk.sum()
        for offset in range(count):
            adjusted = denominator - (float(offset) / float(count)) * tied
            log_likelihood = log_likelihood - (
                maximum + torch.log(torch.clamp(adjusted, min=1e-12))
            )
        event_count += count
    return -log_likelihood / float(event_count)


















def standardize(
    values: np.ndarray, train_positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values[train_positions].mean(axis=0, dtype=np.float64)
    std = values[train_positions].std(axis=0, dtype=np.float64)
    std = np.where(std < 1e-8, 1.0, std)
    output = ((values - mean) / std).astype(np.float32)
    return output, mean.astype(np.float32), std.astype(np.float32)


def train_one(
    values: np.ndarray,
    index: pd.DataFrame,
    seed: int,
    *,
    dann: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    set_seed(seed)
    partitions = index["partition"].astype(str).to_numpy()
    train_positions = np.flatnonzero(partitions == "train")
    validation_positions = np.flatnonzero(partitions == "validation")
    standardized, _, _ = standardize(values, train_positions)
    x = torch.from_numpy(standardized)
    times = torch.from_numpy(index["time_days"].to_numpy(dtype=np.float32))
    events = torch.from_numpy(index["event"].to_numpy(dtype=np.float32))
    if dann:
        centers = sorted(index.iloc[train_positions]["medical_center"].astype(str).unique())
        mapping = {center: offset for offset, center in enumerate(centers)}
        domains = torch.tensor(
            [mapping[str(value)] for value in index.iloc[train_positions]["medical_center"]],
            dtype=torch.long,
        )
        model: Any = DANNCoxHead(values.shape[1], len(centers))
    else:
        centers = []
        domains = None
        model = CoxHead(values.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    best_score = -math.inf
    best_state = None
    best_epoch = -1
    patience = 0
    train_tensor = torch.from_numpy(train_positions)
    val_time = index.iloc[validation_positions]["time_days"].to_numpy(dtype=np.float64)
    val_event = index.iloc[validation_positions]["event"].to_numpy(dtype=np.int64)
    train_time = index.iloc[train_positions]["time_days"].to_numpy(dtype=np.float64)
    train_event = index.iloc[train_positions]["event"].to_numpy(dtype=np.int64)
    for epoch in range(200):
        if STOP_REQUESTED:
            raise KeyboardInterrupt("stop requested")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if dann:
            risk, domain_logits = model(
                x[train_tensor], float(epoch / 199.0)
            )
            loss = cox_efron_loss(
                risk, times[train_tensor], events[train_tensor]
            ) + torch.nn.functional.cross_entropy(domain_logits, domains)
        else:
            risk = model(x[train_tensor])
            loss = cox_efron_loss(risk, times[train_tensor], events[train_tensor])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            all_risk = (
                model.predict(x).numpy() if dann else model(x).numpy()
            )
        validation = uno_c(
            train_time,
            train_event,
            val_time,
            val_event,
            all_risk[validation_positions],
        )
        score = -math.inf if not np.isfinite(validation) else float(validation)
        if score > best_score + 1e-8:
            best_score = score
            best_epoch = epoch
            best_state = {
                name: parameter.detach().clone() for name, parameter in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        if patience >= 20:
            break
    if best_state is None:
        raise RuntimeError("survival training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model.predict(x).numpy() if dann else model(x).numpy()
    return prediction.astype(np.float32), {
        "seed": seed,
        "best_validation_uno_c": best_score,
        "best_epoch": best_epoch,
        "epochs_executed": epoch + 1,
        "dann": dann,
        "train_centers": centers,
    }


def patient_frames(index: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames = {}
    for partition in ("train", "validation", "id_test", "ood_test"):
        selected = index["partition"].astype(str).eq(partition)
        frame = index.loc[selected, ["patient_id", "medical_center"]].copy()
        frame["patch_uid"] = frame["patient_id"].astype(str)
        frame["row_position"] = np.flatnonzero(selected.to_numpy())
        frames[partition] = frame
    return frames


def representations(
    raw: np.ndarray, reinhard: np.ndarray, index: pd.DataFrame
) -> list[tuple[str, np.ndarray | None, bool, tuple[str, ...], dict[str, Any]]]:
    frames = patient_frames(index)
    result: list[tuple[str, np.ndarray | None, bool, tuple[str, ...], dict[str, Any]]] = [
        ("RAW", raw, False, ("in_center", "held_out_center"), {}),
        ("Reinhard-RAW", reinhard, False, ("in_center", "held_out_center"), {}),
        ("DANN", raw, True, ("in_center", "held_out_center"), {"train_only": True}),
        ("Reinhard-DANN", reinhard, True, ("in_center", "held_out_center"), {"train_only": True}),
    ]
    for label, values in (("ComBat-inductive", raw), ("Reinhard-ComBat-inductive", reinhard)):
        try:
            model = combat_inductive.fit_combat_inductive(values, frames["train"])
            corrected = np.asarray(values, dtype=np.float32).copy()
            for partition in ("train", "validation", "id_test"):
                frame = frames[partition]
                corrected[frame["row_position"].to_numpy(dtype=np.int64)] = (
                    combat_inductive.transform_combat_inductive(values, frame, model)
                )
            result.append((label, corrected, False, ("in_center",), {"train_centers": list(model.centers)}))
        except Exception as error:
            result.append((label, None, False, tuple(), {"failure": f"{type(error).__name__}: {error}"}))
    for label, values in (("ComBat-transductive", raw), ("Reinhard-ComBat-transductive", reinhard)):
        try:
            corrected = combat_transductive.combat_correct_official_split(values, frames)
            result.append((label, corrected, False, ("in_center", "held_out_center"), {"transductive": True, "inductive_ranking_eligible": False}))
        except Exception as error:
            result.append((label, None, False, tuple(), {"failure": f"{type(error).__name__}: {error}"}))
    return result


def evaluate_method(
    method: str,
    values: np.ndarray,
    dann: bool,
    domains: tuple[str, ...],
    index: pd.DataFrame,
    details: dict[str, Any],
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    seed_predictions = []
    training = []
    for seed in SEEDS:
        prediction, status = train_one(values, index, seed, dann=dann)
        seed_predictions.append(prediction)
        training.append(status)
    prediction_matrix = np.stack(seed_predictions)
    risk = prediction_matrix.mean(axis=0)
    partitions = index["partition"].astype(str).to_numpy()
    train_positions = np.flatnonzero(partitions == "train")
    train_time = index.iloc[train_positions]["time_days"].to_numpy(dtype=np.float64)
    train_event = index.iloc[train_positions]["event"].to_numpy(dtype=np.int64)
    risk_reference = float(np.max(risk[train_positions]))
    baseline_times, baseline_hazard = breslow_baseline(
        train_time, train_event, risk[train_positions], risk_reference=risk_reference,
    )
    train_event_times = train_time[train_event.astype(bool)]
    auc_times = np.quantile(train_event_times, [0.25, 0.50, 0.75])
    ibs_times = np.unique(np.quantile(train_event_times, np.linspace(0.10, 0.90, 20)))
    rows = []
    domain_partition = {"in_center": "id_test", "held_out_center": "ood_test"}
    for domain, partition in domain_partition.items():
        positions = np.flatnonzero(partitions == partition)
        if domain not in domains:
            for metric in ("uno_c_index", "harrell_c_index", "integrated_brier_score", "td_auc_q25", "td_auc_q50", "td_auc_q75"):
                rows.append({"method": method, "domain": domain, "metric": metric, "value": math.nan, "status": "not_applicable"})
            continue
        test_time = index.iloc[positions]["time_days"].to_numpy(dtype=np.float64)
        test_event = index.iloc[positions]["event"].to_numpy(dtype=np.int64)
        test_risk = risk[positions]
        survival = survival_probability(
            baseline_times, baseline_hazard, test_risk, ibs_times,
            risk_reference=risk_reference,
        )
        brier = brier_score_ipcw(
            train_time, train_event, test_time, test_event, survival, ibs_times
        )
        ibs = float(np.trapz(brier, ibs_times) / max(ibs_times[-1] - ibs_times[0], 1e-12))
        metrics = {
            "uno_c_index": uno_c(train_time, train_event, test_time, test_event, test_risk),
            "harrell_c_index": harrell_c(test_time, test_event, test_risk),
            "integrated_brier_score": ibs,
            "td_auc_q25": time_dependent_auc(train_time, train_event, test_time, test_event, test_risk, float(auc_times[0])),
            "td_auc_q50": time_dependent_auc(train_time, train_event, test_time, test_event, test_risk, float(auc_times[1])),
            "td_auc_q75": time_dependent_auc(train_time, train_event, test_time, test_event, test_risk, float(auc_times[2])),
        }
        for metric, value in metrics.items():
            rows.append({"method": method, "domain": domain, "metric": metric, "value": float(value), "status": "complete"})
    predictions = index[["patient_id", "partition", "medical_center", "event", "time_days"]].copy()
    predictions.insert(0, "method", method)
    predictions["risk_seed_7028"] = prediction_matrix[0]
    predictions["risk_seed_17028"] = prediction_matrix[1]
    predictions["risk_seed_27028"] = prediction_matrix[2]
    predictions["risk_seed_mean"] = risk
    details = dict(details)
    details["training"] = training
    details["evaluation_rule"] = EVALUATION_RULE
    details["head_sha256"] = sha256_file(Path(__file__))
    details["training_risk_reference"] = risk_reference
    return rows, predictions, details


def run(backbone: str) -> None:
    started = time.time()
    freeze = json.loads(IMPLEMENTATION_FREEZE.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen_before_efficacy_evaluation":
        raise ValueError("implementation freeze is not active")
    aggregate_root = (
        ROOT / "data/features" / backbone / "tcga_brca"
        / WSI_AGGREGATE_COLLECTION
    )
    aggregate_path = aggregate_root / "patient_allpatch_mean.npz"
    aggregate_manifest = json.loads(
        (aggregate_root / "aggregate_manifest.json").read_text(encoding="utf-8")
    )
    if aggregate_manifest.get("status") != "complete" or sha256_file(aggregate_path) != aggregate_manifest.get("output_sha256"):
        raise ValueError("aggregate contract failed")
    arrays = np.load(aggregate_path, allow_pickle=False)
    index = pd.read_csv(aggregate_root / "patient_index.csv", dtype={"patient_id": str})
    patient_ids = arrays["patient_id"].astype(str)
    if patient_ids.tolist() != index["patient_id"].astype(str).tolist():
        raise ValueError("aggregate/index patient order differs")
    raw = np.asarray(arrays["raw_mean"], dtype=np.float32)
    reinhard = np.asarray(arrays["reinhard_mean"], dtype=np.float32)
    all_rows: list[dict[str, Any]] = []
    all_predictions = []
    method_details = {}
    for method, values, dann, domains, details in representations(raw, reinhard, index):
        if STOP_REQUESTED:
            raise KeyboardInterrupt("stop requested")
        log(f"SURVIVAL_METHOD_START backbone={backbone} method={method}")
        if values is None:
            for domain in ("in_center", "held_out_center"):
                for metric in ("uno_c_index", "harrell_c_index", "integrated_brier_score", "td_auc_q25", "td_auc_q50", "td_auc_q75"):
                    all_rows.append({"method": method, "domain": domain, "metric": metric, "value": math.nan, "status": "runtime_failure"})
            method_details[method] = details
            continue
        rows, predictions, completed_details = evaluate_method(
            method, values, dann, domains, index, details
        )
        all_rows.extend(rows)
        all_predictions.append(predictions)
        method_details[method] = completed_details
        log(f"SURVIVAL_METHOD_COMPLETE backbone={backbone} method={method}")
    output_root = EXPERIMENT / "controls/baseline_results" / backbone
    output_root.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(all_rows)
    results.insert(0, "backbone", backbone)
    results.insert(1, "split_id", "tcga_brca_overall_survival__rep-00__train-centers-6")
    results_path = output_root / "survival_metrics_long.csv"
    predictions_path = output_root / "patient_predictions.csv"
    for frame, path in ((results, results_path), (pd.concat(all_predictions, ignore_index=True), predictions_path)):
        temporary = path.with_suffix(path.suffix + ".partial")
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    manifest = {
        "schema": "delta-brca-wsi-survival-control-results-v1",
        "status": "complete",
        "backbone": backbone,
        "methods": sorted(results["method"].unique().tolist()),
        "patients": len(index),
        "implementation_freeze": str(IMPLEMENTATION_FREEZE),
        "implementation_freeze_sha256": sha256_file(IMPLEMENTATION_FREEZE),
        "aggregate_manifest_sha256": sha256_file(aggregate_root / "aggregate_manifest.json"),
        "results_sha256": sha256_file(results_path),
        "predictions_sha256": sha256_file(predictions_path),
        "method_details": method_details,
        "elapsed_seconds": time.time() - started,
        "completed_at_kst": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "delta_included": False,
        "evaluation_rule": EVALUATION_RULE,
        "head_sha256": sha256_file(Path(__file__)),
    }
    atomic_json(output_root / "control_results_manifest.json", manifest)
    log(f"BRCA_WSI_SURVIVAL_CONTROLS_COMPLETE backbone={backbone}")


def self_test() -> None:
    # Exact hand-calculated Cox example and test-batch invariance regression.
    toy_time = np.asarray([1.0, 2.0])
    toy_event = np.asarray([1, 1])
    toy_risk = np.log(np.asarray([1.0, 2.0]))
    reference = float(toy_risk.max())
    bt, bh = breslow_baseline(toy_time, toy_event, toy_risk, risk_reference=reference)
    alone = survival_probability(bt, bh, np.asarray([0.0]), np.asarray([1.0]), risk_reference=reference)
    together = survival_probability(bt, bh, np.log(np.asarray([1.0, 4.0])), np.asarray([1.0]), risk_reference=reference)
    np.testing.assert_allclose(alone[0, 0], math.exp(-1.0 / 3.0), rtol=1e-13)
    np.testing.assert_array_equal(alone[0], together[0])
    # Left limits include all censoring decrements strictly BEFORE a query.
    km = censoring_km(np.asarray([1.0, 2.0, 3.0]), np.asarray([0, 1, 1]))
    np.testing.assert_allclose(km(np.asarray([1.0, 1.5, 2.0, 2.5]), left=True), [1.0, 2/3, 2/3, 2/3])
    np.testing.assert_allclose(km(np.asarray([1.0, 1.5, 2.0, 2.5])), [2/3]*4)
    rng = np.random.default_rng(7)
    n = 40
    values = rng.normal(size=(n, 12)).astype(np.float32)
    times = rng.integers(30, 2000, size=n).astype(np.float64)
    events = (rng.random(n) < 0.45).astype(np.int64)
    events[:3] = 1
    risk = -times / times.std() + rng.normal(scale=0.1, size=n)
    train = np.arange(24)
    test = np.arange(24, n)
    uno = uno_c(times[train], events[train], times[test], events[test], risk[test])
    harrell = harrell_c(times[test], events[test], risk[test])
    if not (np.isfinite(uno) and 0 <= uno <= 1 and np.isfinite(harrell) and 0 <= harrell <= 1):
        raise AssertionError("concordance self-test failed")
    tensor = torch.from_numpy(values[:24])
    loss = cox_efron_loss(
        torch.from_numpy(risk[:24].astype(np.float32)),
        torch.from_numpy(times[:24].astype(np.float32)),
        torch.from_numpy(events[:24].astype(np.float32)),
    )
    if not torch.isfinite(loss):
        raise AssertionError("Cox loss self-test failed")
    print("BRCA_WSI_SURVIVAL_HEAD_SELF_TEST_COMPLETE", flush=True)


def main() -> None:
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test")
    run_parser = subparsers.add_parser("run-controls")
    run_parser.add_argument("--backbone", choices=BACKBONES, required=True)
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    elif args.command == "run-controls":
        run(args.backbone)
    else:
        raise RuntimeError(args.command)


if __name__ == "__main__":
    main()
