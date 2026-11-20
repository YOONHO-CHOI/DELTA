"""Shared frozen PathoROB cohorts, metrics, MIL and DANN evaluators.

These learners consume externally supplied embeddings and declared partitions.
Operator estimation, image transformation and embedding correction are not
implemented here.
"""
from __future__ import annotations
import math
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                            brier_score_loss, f1_score, roc_auc_score)
def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")

def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    prediction = probabilities.argmax(1)
    confidence = probabilities.max(1)
    correct = prediction == labels
    score = 0.0
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (confidence > left) & (confidence <= right)
        if mask.any():
            score += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return float(score)

def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    prediction = probabilities.argmax(1)
    classes = probabilities.shape[1]
    result = {
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
        "ece_10bin": expected_calibration_error(labels, probabilities),
    }
    if classes == 2:
        positive = probabilities[:, 1]
        result.update(
            auroc=float(roc_auc_score(labels, positive)),
            auprc=float(average_precision_score(labels, positive)),
            brier=float(brier_score_loss(labels, positive)),
        )
    else:
        onehot = np.eye(classes, dtype=np.float64)[labels]
        result.update(
            auroc=float(
                roc_auc_score(onehot, probabilities, average="macro", multi_class="ovr")
            ),
            auprc=float(average_precision_score(onehot, probabilities, average="macro")),
            brier=float(np.mean(np.sum((onehot - probabilities) ** 2, axis=1))),
        )
    return result

class GatedAttentionMIL(torch.nn.Module):
    def __init__(self, input_dim: int, classes: int, hidden: int = 256, dropout: float = 0.25):
        super().__init__()
        self.project = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.attention_v = torch.nn.Linear(hidden, hidden)
        self.attention_u = torch.nn.Linear(hidden, hidden)
        self.attention_w = torch.nn.Linear(hidden, 1)
        self.classifier = torch.nn.Linear(hidden, classes)

    def encode(self, bags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = self.project(bags)
        attention = self.attention_w(
            torch.tanh(self.attention_v(values))
            * torch.sigmoid(self.attention_u(values))
        ).squeeze(-1)
        attention = attention.masked_fill(~mask, torch.finfo(attention.dtype).min)
        weights = torch.softmax(attention, dim=1)
        return torch.sum(weights.unsqueeze(-1) * values, dim=1)

    def forward(self, bags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(bags, mask))

class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, weight: float) -> torch.Tensor:
        ctx.weight = float(weight)
        return tensor.view_as(tensor)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.weight * gradient, None

class JointAnalysisUnitDANN(torch.nn.Module):
    """DANN attached to the same slide/patient representation used for prediction."""

    def __init__(self, input_dim: int, classes: int, domains: int):
        super().__init__()
        self.mil = GatedAttentionMIL(input_dim, classes)
        self.domain_discriminator = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, domains),
        )

    def forward(
        self, bags: torch.Tensor, mask: torch.Tensor, lambda_weight: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.mil.encode(bags, mask)
        class_logits = self.mil.classifier(pooled)
        domain_logits = self.domain_discriminator(
            GradientReverse.apply(pooled, lambda_weight)
        )
        return class_logits, domain_logits

    def predict(self, bags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.mil(bags, mask)

class BagData:
    def __init__(self, features: np.ndarray, metadata: pd.DataFrame, cohort: pd.DataFrame):
        if cohort["slide_id"].duplicated().any():
            raise ValueError("cohort slide_id is duplicated")
        frame = metadata[["row_position", "slide_id"]].merge(
            cohort[
                [
                    "slide_id",
                    "patient_id",
                    "label",
                    "label_code",
                    "medical_center",
                    "analysis_unit",
                ]
            ],
            on="slide_id",
            how="inner",
            validate="many_to_one",
            sort=False,
            suffixes=("", "_cohort"),
        )
        if frame.empty:
            raise ValueError("no feature rows map to cohort")
        analysis_unit = str(cohort["analysis_unit"].iloc[0])
        self.analysis_unit = analysis_unit
        frame["unit_id"] = (
            frame["slide_id"].astype(str)
            if analysis_unit == "slide"
            else frame["patient_id"].astype(str)
        )
        self.features = features
        self.frame = frame
        self.rows = {
            str(unit): group["row_position"].to_numpy(dtype=np.int64)
            for unit, group in frame.groupby("unit_id", sort=True)
        }
        center_counts = frame.groupby("unit_id", sort=True)["medical_center"].nunique()
        if (center_counts != 1).any():
            raise ValueError("analysis unit spans multiple medical centers")
        self.info = frame.groupby("unit_id", sort=True).agg(
            patient_id=("patient_id", "first"),
            label_code=("label_code", "first"),
            medical_center=("medical_center", "first"),
        )

    def partitions(self, split: pd.DataFrame) -> dict[str, list[str]]:
        result = {}
        for name in ("train", "validation", "id_test", "ood_test"):
            patients = set(
                split.loc[split["partition"].eq(name), "patient_id"].astype(str)
            )
            result[name] = sorted(
                self.info.index[
                    self.info["patient_id"].astype(str).isin(patients)
                ].astype(str)
            )
            if not result[name]:
                raise ValueError(f"empty downstream partition: {name}")
        flat = [unit for values in result.values() for unit in values]
        if len(flat) != len(set(flat)):
            raise ValueError("analysis-unit leakage between partitions")
        return result

def pad_bags(tensors: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(item) for item in tensors], device=tensors[0].device)
    padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    positions = torch.arange(padded.shape[1], device=padded.device).unsqueeze(0)
    return padded, positions < lengths.unsqueeze(1)

def evaluate_units(
    model: GatedAttentionMIL,
    gpu_bags: dict[str, torch.Tensor],
    units: list[str],
    labels: dict[str, int],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(units), batch_size):
            current = units[start : start + batch_size]
            padded, mask = pad_bags([gpu_bags[unit] for unit in current])
            probabilities.append(torch.softmax(model(padded, mask), 1).cpu().numpy())
    y = np.asarray([labels[unit] for unit in units], dtype=np.int64)
    return y, np.concatenate(probabilities, axis=0)

def evaluate_dann_units(
    model: JointAnalysisUnitDANN,
    gpu_bags: dict[str, torch.Tensor],
    units: list[str],
    labels: dict[str, int],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(units), batch_size):
            current = units[start : start + batch_size]
            padded, mask = pad_bags([gpu_bags[unit] for unit in current])
            probabilities.append(
                torch.softmax(model.predict(padded, mask), 1).cpu().numpy()
            )
    y = np.asarray([labels[unit] for unit in units], dtype=np.int64)
    return y, np.concatenate(probabilities, axis=0)

def fit_mil_dann_once(
    features: np.ndarray,
    bag_data: BagData,
    partitions: dict[str, list[str]],
    seed: int,
    weight_decay: float,
    *,
    epochs: int,
    batch_size: int,
    domains: tuple[str, ...] = ("ID", "OOD"),
) -> tuple[float, dict[str, dict[str, float]], dict]:
    """Train a joint DANN at the exact downstream analysis-unit level."""

    set_seed(seed)
    device = torch.device("cuda")
    units = sorted({unit for values in partitions.values() for unit in values})
    labels = {
        unit: int(float(bag_data.info.loc[unit, "label_code"])) for unit in units
    }
    label_values = sorted(set(labels.values()))
    if label_values != list(range(len(label_values))):
        raise ValueError(f"labels are not contiguous: {label_values}")

    train_units = list(partitions["train"])
    train_centers = sorted(
        {str(bag_data.info.loc[unit, "medical_center"]) for unit in train_units}
    )
    if len(train_centers) < 2:
        raise ValueError("joint DANN requires at least two training medical centers")
    center_mapping = {center: index for index, center in enumerate(train_centers)}

    gpu_bags = {
        unit: torch.from_numpy(
            np.asarray(features[bag_data.rows[unit]], dtype=np.float32)
        ).to(device)
        for unit in units
    }
    model = JointAnalysisUnitDANN(
        int(features.shape[1]), len(label_values), len(train_centers)
    ).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.002, weight_decay=float(weight_decay)
    )
    criterion = torch.nn.CrossEntropyLoss()
    best_score = -math.inf
    best_state = None

    for epoch in range(int(epochs)):
        lambda_weight = float(epoch / max(1, int(epochs) - 1))
        model.train()
        order = train_units.copy()
        random.Random(int(seed) + epoch).shuffle(order)
        for start in range(0, len(order), batch_size):
            current = order[start : start + batch_size]
            padded, mask = pad_bags([gpu_bags[unit] for unit in current])
            biology_target = torch.tensor(
                [labels[unit] for unit in current], dtype=torch.long, device=device
            )
            domain_target = torch.tensor(
                [
                    center_mapping[str(bag_data.info.loc[unit, "medical_center"])]
                    for unit in current
                ],
                dtype=torch.long,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            class_logits, domain_logits = model(padded, mask, lambda_weight)
            loss = criterion(class_logits, biology_target) + criterion(
                domain_logits, domain_target
            )
            loss.backward()
            optimizer.step()

        val_y, val_probability = evaluate_dann_units(
            model, gpu_bags, partitions["validation"], labels, batch_size
        )
        score = balanced_accuracy_score(val_y, val_probability.argmax(1))
        if float(score) > float(best_score):
            best_score = float(score)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("joint DANN generated no checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    outputs = {}
    partition_name = {"ID": "id_test", "OOD": "ood_test"}
    for domain in domains:
        y, probability = evaluate_dann_units(
            model,
            gpu_bags,
            partitions[partition_name[domain]],
            labels,
            batch_size,
        )
        outputs[domain] = metrics(y, probability)
    details = {
        "protocol": "joint_analysis_unit_dann_v1",
        "analysis_unit": str(bag_data.analysis_unit),
        "train_only_fit": True,
        "biology_supervision": "analysis_unit_label_only",
        "domain_supervision": "analysis_unit_medical_center_only",
        "lambda_schedule": "linear_0_to_1",
        "domain_centers": train_centers,
        "domain_discriminator": "Linear(256,128)-ReLU-Linear",
        "optimizer": "SGD",
        "learning_rate": 0.002,
    }
    del gpu_bags, model
    torch.cuda.empty_cache()
    return float(best_score), outputs, details

def fit_mil_dann(
    features: np.ndarray,
    bag_data: BagData,
    partitions: dict[str, list[str]],
    seed: int,
    *,
    epochs: int,
    batch_size: int,
    domains: tuple[str, ...] = ("ID", "OOD"),
) -> tuple[float, float, dict[str, dict[str, float]], dict]:
    candidates = []
    for weight_decay in (0.0, 0.01):
        validation, outputs, details = fit_mil_dann_once(
            features,
            bag_data,
            partitions,
            seed,
            weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            domains=domains,
        )
        candidates.append(
            (validation, -weight_decay, weight_decay, outputs, details)
        )
    chosen = max(candidates, key=lambda item: (item[0], item[1]))
    return (
        float(chosen[0]),
        float(chosen[2]),
        chosen[3],
        chosen[4],
    )

def fit_mil_once(
    features: np.ndarray,
    bag_data: BagData,
    partitions: dict[str, list[str]],
    seed: int,
    weight_decay: float,
    *,
    epochs: int,
    batch_size: int,
    domains: tuple[str, ...] = ("ID", "OOD"),
) -> tuple[float, dict[str, dict[str, float]]]:
    set_seed(seed)
    device = torch.device("cuda")
    units = sorted({unit for values in partitions.values() for unit in values})
    labels = {
        unit: int(float(bag_data.info.loc[unit, "label_code"])) for unit in units
    }
    label_values = sorted(set(labels.values()))
    if label_values != list(range(len(label_values))):
        raise ValueError(f"labels are not contiguous: {label_values}")
    gpu_bags = {
        unit: torch.from_numpy(
            np.asarray(features[bag_data.rows[unit]], dtype=np.float32)
        ).to(device)
        for unit in units
    }
    model = GatedAttentionMIL(int(features.shape[1]), len(label_values)).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.002, weight_decay=float(weight_decay)
    )
    criterion = torch.nn.CrossEntropyLoss()
    best_score = -math.inf
    best_state = None
    train_units = list(partitions["train"])
    for epoch in range(int(epochs)):
        model.train()
        order = train_units.copy()
        random.Random(int(seed) + epoch).shuffle(order)
        for start in range(0, len(order), batch_size):
            current = order[start : start + batch_size]
            padded, mask = pad_bags([gpu_bags[unit] for unit in current])
            target = torch.tensor(
                [labels[unit] for unit in current], dtype=torch.long, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(padded, mask), target)
            loss.backward()
            optimizer.step()
        val_y, val_probability = evaluate_units(
            model,
            gpu_bags,
            partitions["validation"],
            labels,
            batch_size,
        )
        score = balanced_accuracy_score(val_y, val_probability.argmax(1))
        if float(score) > float(best_score):
            best_score = float(score)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("MIL generated no checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    outputs = {}
    partition_name = {"ID": "id_test", "OOD": "ood_test"}
    for domain in domains:
        partition = partition_name[domain]
        y, probability = evaluate_units(
            model, gpu_bags, partitions[partition], labels, batch_size
        )
        outputs[domain] = metrics(y, probability)
    del gpu_bags, model
    torch.cuda.empty_cache()
    return float(best_score), outputs

def fit_mil(
    features: np.ndarray,
    bag_data: BagData,
    partitions: dict[str, list[str]],
    seed: int,
    *,
    epochs: int,
    batch_size: int,
    domains: tuple[str, ...] = ("ID", "OOD"),
) -> tuple[float, float, dict[str, dict[str, float]]]:
    candidates = []
    for weight_decay in (0.0, 0.01):
        validation, outputs = fit_mil_once(
            features,
            bag_data,
            partitions,
            seed,
            weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            domains=domains,
        )
        candidates.append((validation, -weight_decay, weight_decay, outputs))
    _, _, selected, outputs = max(candidates, key=lambda item: (item[0], item[1]))
    return float(max(item[0] for item in candidates)), float(selected), outputs

def build_frames(
    metadata: pd.DataFrame, cohort: pd.DataFrame, split: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    slide_map = cohort.set_index("slide_id")
    frame = metadata.copy()
    frame["patient_id"] = frame["slide_id"].astype(str).map(
        slide_map["patient_id"].astype(str)
    )
    frame["label_code"] = frame["slide_id"].astype(str).map(slide_map["label_code"])
    frame["label"] = frame["slide_id"].astype(str).map(slide_map["label"])
    if frame[["patient_id", "label_code"]].isna().any().any():
        raise ValueError("raw feature rows do not map to downstream cohort")
    result = {}
    for name in ("train", "validation", "id_test", "ood_test"):
        patients = set(
            split.loc[split["partition"].eq(name), "patient_id"].astype(str)
        )
        result[name] = frame[
            frame["patient_id"].astype(str).isin(patients)
        ].copy()
        if result[name].empty:
            raise ValueError(f"empty tile partition: {name}")
    return result
