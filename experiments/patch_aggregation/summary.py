"""Six-endpoint paired patch-aggregation inference from audited tables."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import openpyxl
import pandas as pd
from experiments.common.resources import repository_root

COMBAT_SHA256 = "318cd5f08d0675f68ea2ce8a3c366085996bc005ca3013c47a261882fad35997"
METHODS = (
    "RAW", "Reinhard-RAW", "DANN", "Reinhard-DANN", "DELTA",
    "Reinhard-DELTA", "ComBat-transductive", "Reinhard-ComBat-transductive",
)
METRICS = (
    ("balanced_accuracy", "Balanced accuracy", "×100"),
    ("macro_f1", "Macro-F1", "×100"),
    ("auroc", "AUROC", "×100"),
    ("auprc", "AUPRC", "×100"),
    ("brier", "Brier score ↓", "×100"),
    ("ece_10bin", "ECE ↓", "×100"),
)
LOWER_IS_BETTER = {"brier", "ece_10bin"}
BRANCHES = (
    ("DELTA", ("RAW", "Reinhard-RAW", "DANN", "ComBat-transductive")),
    ("Reinhard-DELTA", (
        "RAW", "Reinhard-RAW", "Reinhard-DANN",
        "Reinhard-ComBat-transductive",
    )),
)
BOOTSTRAPS = 10_000
HALF_SIGNS = ((np.arange(1024, dtype=np.uint16)[:, None] >>
               np.arange(10, dtype=np.uint16)) & 1).astype(np.int8) * 2 - 1

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def load_data(workbook: Path, combat_means: Path) -> pd.DataFrame:
    book = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    sheet = book["All Metrics"]
    rows = sheet.values
    for _ in range(3):
        next(rows)
    columns = next(rows)
    frame = pd.DataFrame((row for row in rows if row[0] is not None), columns=columns)
    book.close()
    selected = frame[frame.method.isin(METHODS[:-1])].copy()
    expected = 20 * 3 * 2 * 6 * 7
    assert len(selected) == expected, (len(selected), expected)
    assert selected.split_n.eq(100).all()
    assert np.isfinite(selected.value_mean.to_numpy(dtype=float)).all()
    selected = selected[["backbone", "backbone_key", "task_id", "dataset",
                         "method", "domain", "metric", "split_n", "value_mean"]]
    assert sha256(combat_means) == COMBAT_SHA256
    extra = pd.read_csv(combat_means)
    assert len(extra) == 20 * 3 * 2 * 6
    assert extra.backbone_key.nunique() == 20
    assert set(extra.domain) == {"ID", "OOD"}
    assert set(extra.metric) == {name for name, _, _ in METRICS}
    assert np.isfinite(extra.mean_across_100_splits.to_numpy()).all()
    extra = extra.rename(columns={"mean_across_100_splits": "value_mean"})
    extra["method"] = "Reinhard-ComBat-transductive"
    extra["split_n"] = 100
    metadata = selected[["backbone", "backbone_key", "task_id", "dataset"]].drop_duplicates()
    assert len(metadata) == 20 * 3
    extra = extra.merge(metadata, on=["backbone_key", "task_id"], validate="many_to_one")
    extra = extra[selected.columns]
    result = pd.concat([selected, extra], ignore_index=True)
    keys = ["backbone_key", "task_id", "method", "domain", "metric"]
    assert len(result) == 20 * 3 * 2 * 6 * 8
    assert not result.duplicated(keys).any()
    assert result.groupby(["method", "metric", "domain"]).size().eq(60).all()
    assert result.groupby(["backbone_key", "method", "metric", "domain"]).size().eq(3).all()
    return result

def exact_signed_rank_p(diff: np.ndarray) -> float:
    """Two-sided exact sign-randomization of signed midranks, Wilcox zeros."""
    rounded = np.round(np.asarray(diff, dtype=float), 12)
    nonzero = rounded[rounded != 0]
    if not len(nonzero):
        return 1.0
    ranks = (pd.Series(np.abs(nonzero)).rank(method="average") * 2).astype(int)
    weights = ranks.to_numpy()
    total = int(weights.sum())
    observed = min(int(weights[nonzero > 0].sum()),
                   int(weights[nonzero < 0].sum()))
    counts = np.zeros(total + 1, dtype=np.int64)
    counts[0] = 1
    for weight in weights:
        counts[weight:] += counts[:-weight].copy()
    assert counts.sum() == 2 ** len(nonzero)
    return min(1.0, 2 * int(counts[:observed + 1].sum()) / 2 ** len(nonzero))

def exact_signflip_p(diff: np.ndarray) -> float:
    """Two-sided exact paired sign-flip of the unranked mean, as in Figure 2."""
    values = np.asarray(diff, dtype=float)
    assert values.shape == (20,) and np.isfinite(values).all()
    sums = (HALF_SIGNS @ values[:10])[:, None] + (HALF_SIGNS @ values[10:])[None, :]
    return float(np.mean(np.abs(sums) >= abs(float(values.sum())) - 1e-14))

def paired_test(diff: np.ndarray, test_name: str) -> float:
    if test_name == "sign-flip":
        return exact_signflip_p(diff)
    if test_name == "signed-rank":
        return exact_signed_rank_p(diff)
    raise ValueError(f"Unknown paired test: {test_name}")

def holm(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for step, index in enumerate(order):
        running = max(running, min(1.0, float(values[index]) * (len(values) - step)))
        adjusted[index] = running
    return adjusted

def favourable_wins(differences: np.ndarray, metric: str) -> int:
    """Count backbones helped by the target without changing the subtraction."""
    return int(((differences < 0) if metric in LOWER_IS_BETTER else
                (differences > 0)).sum())

def summarize(frame: pd.DataFrame, test_name: str = "signed-rank") -> pd.DataFrame:
    pooled = frame.groupby(["backbone_key", "method", "domain", "metric"],
                           as_index=False).value_mean.mean()
    records = []
    for target, comparators in BRANCHES:
        for domain in ("ID", "OOD"):
            for metric, _, units in METRICS:
                current = pooled[(pooled.domain == domain) & (pooled.metric == metric)]
                wide = current.pivot(index="backbone_key", columns="method", values="value_mean")
                assert wide.shape == (20, 8) and np.isfinite(wide.to_numpy()).all()
                family = []
                for comparator in comparators:
                    native_difference = wide[target].to_numpy() - wide[comparator].to_numpy()
                    delta = native_difference * 100.0
                    rng = np.random.default_rng(42)
                    sampled = delta[rng.integers(0, 20, size=(BOOTSTRAPS, 20))].mean(axis=1)
                    lo, hi = np.quantile(sampled, (0.025, 0.975))
                    p_value = paired_test(native_difference, test_name)
                    assert math.isclose(p_value, paired_test(-native_difference, test_name), abs_tol=1e-15)
                    family.append(dict(target=target, domain=domain, metric=metric,
                                       units=units, comparator=comparator,
                                       statistical_test=test_name,
                                       mean=float(delta.mean()), ci_low=float(lo),
                                       ci_high=float(hi),
                                       wins=favourable_wins(delta, metric),
                                       p=p_value, n_backbones=20,
                                       difference=f"{target} − {comparator}",
                                       favourable_side=("negative" if metric in LOWER_IS_BETTER
                                                        else "positive")))
                corrected = holm(np.array([record["p"] for record in family]))
                for record, value in zip(family, corrected):
                    record["holm_p"] = float(value)
                    records.append(record)
    result = pd.DataFrame(records)
    assert len(result) == 2 * 2 * 6 * 4
    assert result.holm_p.between(0, 1).all()
    return result

def summarize_tasks(frame: pd.DataFrame, test_name: str = "signed-rank") -> pd.DataFrame:
    """Preserve the established 12-comparison Holm family per branch/domain/metric."""
    records = []
    tasks = ("camelyon_slide_tumor", "tcga4_four_class", "tcga_nsclc_subtype")
    for target_index, (target, comparators) in enumerate(BRANCHES):
        for domain_index, domain in enumerate(("ID", "OOD")):
            for metric, _, units in METRICS:
                family = []
                for task_index, task_id in enumerate(tasks):
                    current = frame[(frame.domain == domain) & (frame.metric == metric) &
                                    (frame.task_id == task_id)]
                    wide = current.pivot(index="backbone_key", columns="method",
                                         values="value_mean").sort_index()
                    assert wide.shape == (20, 8)
                    for comparator_index, comparator in enumerate(comparators):
                        native_difference = wide[target].to_numpy() - wide[comparator].to_numpy()
                        effect = native_difference * 100.0
                        seed = (4200 + task_index * 100 + comparator_index * 10 +
                                target_index + domain_index * 1000)
                        rng = np.random.default_rng(seed)
                        boot = effect[rng.integers(0, 20, size=(BOOTSTRAPS, 20))].mean(axis=1)
                        lo, hi = np.quantile(boot, (.025, .975))
                        p_value = paired_test(native_difference, test_name)
                        assert math.isclose(p_value, paired_test(-native_difference, test_name), abs_tol=1e-15)
                        family.append(dict(target=target, domain=domain, metric=metric,
                                           units=units, task_id=task_id,
                                           statistical_test=test_name,
                                           comparator=comparator, mean=float(effect.mean()),
                                           ci_low=float(lo), ci_high=float(hi),
                                           wins=favourable_wins(effect, metric),
                                           p=p_value, n_backbones=20,
                                           difference=f"{target} − {comparator}",
                                           favourable_side=("negative" if metric in LOWER_IS_BETTER
                                                            else "positive")))
                adjusted = holm(np.array([entry["p"] for entry in family]))
                for entry, value in zip(family, adjusted):
                    entry["holm_p"] = float(value)
                    records.append(entry)
    result = pd.DataFrame(records)
    assert len(result) == 2 * 2 * 6 * 3 * 4
    assert result.holm_p.between(0, 1).all()
    return result

def pooled_backbone_effects(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Pool the three tasks equally; share each metric's scale across branches."""
    pooled = frame.groupby(["backbone", "method", "domain", "metric"],
                           as_index=False).value_mean.mean()
    pivot = pooled.pivot(index=["backbone", "domain", "metric"],
                         columns="method", values="value_mean")
    effects = []
    for target, matched_raw in (("DELTA", "RAW"),
                                ("Reinhard-DELTA", "Reinhard-RAW")):
        for backbone in pivot.index.get_level_values("backbone").unique():
            for domain in ("ID", "OOD"):
                for metric, _, _ in METRICS:
                    value = 100 * (pivot.loc[(backbone, domain, metric), target] -
                                   pivot.loc[(backbone, domain, metric), matched_raw])
                    assert math.isfinite(value)
                    effects.append(dict(target=target, backbone=backbone,
                                        domain=domain, metric=metric,
                                        matched_raw=matched_raw, effect=float(value),
                                        difference=f"{target} − {matched_raw}",
                                        favourable_side=("negative" if metric in LOWER_IS_BETTER
                                                         else "positive")))
    result = pd.DataFrame(effects)
    assert len(result) == 2 * 20 * 2 * 6
    limits = result.groupby("metric").effect.apply(
        lambda values: max(abs(float(values.min())), abs(float(values.max())))).to_dict()
    assert all(value > 0 and math.isfinite(value) for value in limits.values())
    return result, limits

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--reinhard-combat-means", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Use a new or empty output directory: {output}")
    frame = load_data(args.workbook, args.reinhard_combat_means)
    pooled = summarize(frame)
    tasks = summarize_tasks(frame)
    effects, _ = pooled_backbone_effects(frame)
    output.mkdir(parents=True, exist_ok=True)
    tables = {"Pooled comparisons": pooled, "Task comparisons": tasks, "Backbone effects": effects}
    with pd.ExcelWriter(output / "aggregation_statistics.xlsx", engine="openpyxl") as writer:
        for name, values in tables.items():
            values.to_csv(output / (name.lower().replace(" ", "_") + ".csv"), index=False)
            values.to_excel(writer, sheet_name=name, index=False)
    print("AGGREGATION_STATISTICS_COMPLETE")


if __name__ == "__main__":
    main()
