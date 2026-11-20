"""Patch-aggregation statistics retain the paper's paired inference families."""
from __future__ import annotations
import itertools

import numpy as np
import pandas as pd
import pytest

from experiments.patch_aggregation import report, statistics, summary


def test_exact_signed_rank_handles_ties_zeros_and_native_rounding():
    from scipy.stats import rankdata
    differences = np.array([0., 1., -1., 2., 2., -3.])
    nonzero = differences[differences != 0]
    ranks = rankdata(abs(nonzero), method="average")
    observed = abs(float((ranks * np.sign(nonzero)).sum()))
    expected = np.mean([abs(float(np.dot(signs, ranks))) >= observed
                        for signs in itertools.product((-1, 1), repeat=len(ranks))])
    assert statistics.exact_signed_rank_p(differences) == expected
    assert statistics.exact_signed_rank_p(-differences) == expected
    assert statistics.exact_signed_rank_p(np.full(20, 1e-13)) == 1


def condition_fixture():
    return pd.DataFrame([
        dict(backbone=backbone, task_id=task, method=method, domain=domain,
             metric="balanced_accuracy", value_mean=index / 100 + model / 10000)
        for model, backbone in enumerate(report.BACKBONE_ORDER)
        for task, domain in itertools.product(report.TASK_ORDER, ("ID", "OOD"))
        for index, method in enumerate(report.METHOD_ORDER)
    ])


def test_detailed_report_pools_tasks_and_uses_four_and_twelve_contrasts(monkeypatch):
    monkeypatch.setattr(report, "BOOTSTRAP_REPETITIONS", 32)
    frame = condition_fixture()
    result = report.pairwise_statistics(frame, "balanced_accuracy")
    assert len(result) == 2 * 2 * 4 * 4
    assert result[result.task_id == "ALL"].holm_family_size.eq(4).all()
    assert result[result.task_id != "ALL"].holm_family_size.eq(12).all()
    assert result[result.transductive].holm_p.notna().all()
    assert set(report.paired_p_against_delta(frame, "balanced_accuracy", "ID")) == set(report.BRANCH_COMPARATORS["DELTA"])
    with pytest.raises(ValueError, match="Missing canonical branch comparison"):
        report.pairwise_statistics(frame[frame.method != "Reinhard-ComBat-transductive"], "balanced_accuracy")


def six_metric_fixture():
    return pd.DataFrame([
        dict(backbone_key=str(model), backbone=str(model),
             task_id=report.TASK_ORDER[task], method=method, domain=domain,
             metric=metric, value_mean=.5 + task * .001 + model * .0001 + index * .01)
        for model, task, (index, method), domain, (metric, _, _) in itertools.product(
            range(20), range(3), enumerate(summary.METHODS), ("ID", "OOD"), summary.METRICS)
    ])


def test_six_metric_summary_rounds_before_scaling_and_keeps_both_combat_arms(monkeypatch):
    monkeypatch.setattr(summary, "BOOTSTRAPS", 32)
    rows = []
    for model, task, method, domain, (metric, _, _) in itertools.product(
            range(20), range(3), summary.METHODS, ("ID", "OOD"), summary.METRICS):
        rows.append(dict(backbone_key=str(model), task_id=report.TASK_ORDER[task], method=method,
                         domain=domain, metric=metric,
                         value_mean=1e-13 if method in {"DELTA", "Reinhard-DELTA"} else 0.))
    result = summary.summarize(pd.DataFrame(rows))
    assert len(result) == 96
    assert result.p.eq(1).all() and result.holm_p.eq(1).all()
    assert set(result[result.comparator.str.contains("ComBat")].comparator) == {
        "ComBat-transductive", "Reinhard-ComBat-transductive"}


def test_six_metric_task_families_and_backbone_effects(monkeypatch):
    monkeypatch.setattr(summary, "BOOTSTRAPS", 32)
    frame = six_metric_fixture()
    pooled = summary.summarize(frame)
    tasks = summary.summarize_tasks(frame)
    effects, limits = summary.pooled_backbone_effects(frame)
    assert len(pooled) == 96 and len(tasks) == 288 and len(effects) == 480
    assert len(limits) == 6
    assert set(tasks.comparator.unique()) == set(method for _, methods in summary.BRANCHES for method in methods)
    for _, family in tasks.groupby(["target", "domain", "metric"]):
        assert len(family) == 12
        np.testing.assert_array_equal(summary.holm(family.p.to_numpy()), family.holm_p.to_numpy())
    assert effects[effects.metric.eq("brier")].favourable_side.eq("negative").all()


def test_aggregation_report_preserves_existing_output(tmp_path):
    sentinel = tmp_path / "existing.csv"
    sentinel.write_bytes(b"retain original results")
    with pytest.raises(FileExistsError, match="new or empty"):
        report.main(["--input-csv", str(tmp_path / "missing.csv"), "--output-dir", str(tmp_path)])
    assert sentinel.read_bytes() == b"retain original results"


def test_aggregation_workbook_retains_all_metric_and_paired_tables(tmp_path, monkeypatch):
    from openpyxl import load_workbook
    monkeypatch.setattr(report, "BOOTSTRAP_REPETITIONS", 32)
    frame = six_metric_fixture()
    names = dict(zip(map(str, range(20)), report.BACKBONE_ORDER))
    frame["backbone"] = frame.backbone.map(names)
    frame["split_n"] = 100
    baseline = frame[frame.method.eq("RAW")].set_index(["backbone", "task_id", "domain", "metric"]).value_mean
    frame["raw_mean"] = [baseline.loc[(r.backbone, r.task_id, r.domain, r.metric)] for r in frame.itertuples()]
    direction = np.where(frame.metric.isin(("brier", "ece_10bin")), -1., 1.)
    frame["improvement_pp"] = direction * (frame.value_mean - frame.raw_mean) * 100
    frame["relative_gain_pct"] = frame.improvement_pp / frame.raw_mean.abs()
    frame["protocol"] = frame.method.map(report.method_protocol)
    frame["rank_eligible"] = frame.method.isin(report.RANK_ELIGIBLE_METHODS)
    frame["transductive"] = frame.method.isin(report.TRANSDUCTIVE_METHODS)
    metric = "balanced_accuracy"
    source = pd.DataFrame([dict(method="RAW", domain="ID", status="complete")])
    qc = dict(source_rows=1, backbone_count=20, task_count=3, split_count_min=100, split_count_max=100)
    output = tmp_path / "numeric.xlsx"
    report.build_workbook(output, source, source, frame, report.summarize_scope(frame, metric),
                          report.backbone_summary(frame, metric), report.task_summary(frame, metric),
                          report.pairwise_statistics(frame, metric), tmp_path / "synthetic.csv",
                          "synthetic-non-study-input", qc, "fixture")
    book = load_workbook(output, read_only=True, data_only=True)
    try:
        assert book.sheetnames == ["README", "Table 1 - BA", "Table 2 - APD", "Backbone Summary", "Task Summary",
                                   "Pairwise Statistics", "Condition Means", "All Metrics", "QC and Provenance"]
        assert book["All Metrics"].max_row == len(frame) + 4
        assert book["Pairwise Statistics"].max_row == 64 + 4
        assert book["Condition Means"].max_row == len(frame[frame.metric.eq(metric)]) + 4
    finally:
        book.close()
