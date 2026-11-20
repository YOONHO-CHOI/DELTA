"""Numerical patch-report contracts without graphics dependencies."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from openpyxl import load_workbook
from experiments.pathorob import evaluation, report, summary, workbook


def _toy_signflip(differences):
    """Independent exact reference from integer sum multiplicities, not study data."""
    integers = np.rint(np.asarray(differences) * 4096).astype(int)
    np.testing.assert_array_equal(integers / 4096, differences)
    counts = {0: 1}
    for value in integers:
        following = {}
        for total, count in counts.items():
            for signed in (-int(value), int(value)):
                following[total + signed] = following.get(total + signed, 0) + count
        counts = following
    observed = abs(int(integers.sum()))
    return sum(count for total, count in counts.items() if abs(total) >= observed) / (2 ** len(integers))


def _toy_holm(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [None] * len(values)
    for rank, index in enumerate(order):
        adjusted[index] = min(1., max((len(values) - prior) * values[order[prior]]
                                     for prior in range(rank + 1)))
    return adjusted


@pytest.fixture(scope="module")
def synthetic_patch_snapshot(tmp_path_factory):
    """Generate the fixed panel dimensions entirely from artificial toy values."""
    methods = ["RAW", "Reinhard-RAW", "DANN", "Reinhard-DANN", "ComBat", "Reinhard-ComBat",
               summary.DELTA, summary.RDELTA]
    models = [f"synthetic_encoder_{index:02d}" for index in range(20)]
    endpoints = [entry[0] for entry in summary.METRICS]
    tables = {name: [] for name in ("Representation Backbone", "BA Backbone", "APD Backbone",
                                    "Representation Paired", "APD Paired")}
    datasets = {name: [] for name in ("representation", "dataset_ba", "dataset_apd", "curves")}
    pattern = [-2, -1, 0, 1, 3]

    def toy(model, method, base):
        shift = method * (2 + pattern[(model + method) % len(pattern)])
        return (base + 3 * model + shift) / 4096

    for model, backbone in enumerate(models):
        for method, label in enumerate(methods):
            for endpoint_index, endpoint in enumerate(endpoints):
                value = toy(model, method, 2048 + endpoint_index * 64)
                tables["Representation Backbone"].append(dict(
                    Backbone=backbone, Method=label, Endpoint=endpoint, Score_value=value))
                for offset, dataset in zip((-1, 0, 1), summary.POPULATIONS[endpoint]):
                    datasets["representation"].append(dict(backbone_std=backbone, method_std=label,
                        endpoint=endpoint, dataset=dataset, value=value + offset / 4096))
            for domain_index, domain in enumerate(("ID", "OOD")):
                for key, sheet, column, base in (
                    ("dataset_ba", "BA Backbone", "BA_value", 2048 - 128 * domain_index),
                    ("dataset_apd", "APD Backbone", "APD_value", -512 - 256 * domain_index),
                ):
                    value = toy(model, method, base)
                    tables[sheet].append(dict(Backbone=backbone, Method=label, Domain=domain, **{column: value}))
                    for offset, dataset in zip((-1, 0, 1), ("camelyon", "tcga_4x4", "tolkach_esca")):
                        datasets[key].append(dict(backbone_std=backbone, method_std=label, domain=domain,
                                                  dataset=dataset, value=value + offset / 4096))
    datasets["curves"] = [dict(backbone_std=models[0], method_std=method, dataset="camelyon",
                                domain="OOD", condition=index, value=(2000 + index * 8) / 4096)
                           for index, method in enumerate(methods)]

    def differences(sheet, column, candidate, control, field, value):
        frame = pd.DataFrame(tables[sheet])
        matrix = frame.loc[frame[field].eq(value)].pivot(index="Backbone", columns="Method", values=column)
        return (matrix[candidate] - matrix[control]).to_numpy()

    for candidate, controls in ((summary.DELTA, methods[:3] + ["ComBat"]),
                                (summary.RDELTA, methods[:2] + ["Reinhard-DANN", "Reinhard-ComBat"])):
        family = []
        for endpoint in endpoints:
            for control in controls:
                diff = differences("Representation Backbone", "Score_value", candidate, control, "Endpoint", endpoint)
                family.append(dict(candidate=candidate, control=control, endpoint=endpoint,
                                   exact_signflip_p=_toy_signflip(diff), mean_difference=float(diff.mean())))
        for row, adjusted in zip(family, _toy_holm([row["exact_signflip_p"] for row in family])):
            row["holm_exact_p"] = adjusted
        tables["Representation Paired"].extend(family)
        for metric, sheet, column in (("balanced_accuracy", "BA Backbone", "BA_value"),
                                      ("average_performance_drop", "APD Backbone", "APD_value")):
            family = []
            for domain in ("ID", "OOD"):
                for control in methods[:6]:
                    diff = differences(sheet, column, candidate, control, "Domain", domain)
                    rng = np.random.default_rng(20260819 + len(family))
                    resampled = rng.choice(diff, size=(10000, len(diff)), replace=True).mean(axis=1)
                    low, high = np.percentile(resampled, [2.5, 97.5])
                    family.append(dict(candidate=candidate, control=control, metric=metric, domain=domain,
                        exact_signflip_p=_toy_signflip(diff), mean_difference=float(diff.mean()),
                        ci_low=float(low), ci_high=float(high)))
            for row, adjusted in zip(family, _toy_holm([row["exact_signflip_p"] for row in family])):
                row["holm_exact_p"] = adjusted
            tables["APD Paired"].extend(family)
    datasets["sources"] = [{"kind": "synthetic-test-fixture", "description": "No images or study outcomes"}]
    payload = dict(schema_version=1, workbook=dict(name="Artificial regression panel",
        sha256=hashlib.sha256(b"synthetic workbook identity").hexdigest(), tables=tables), dataset_summaries=datasets)
    path = tmp_path_factory.mktemp("synthetic-patch-source") / "toy_snapshot.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_synthetic_patch_source_reconciles_all_values_and_contrasts(synthetic_patch_snapshot):
    before = hashlib.sha256(synthetic_patch_snapshot.read_bytes()).hexdigest()
    frames, rep, ba, apd, curves, audit = summary.load_and_verify(synthetic_patch_snapshot)
    assert audit["representation_values_reconciled"] == 640
    assert audit["BA_values_reconciled"] == audit["APD_values_reconciled"] == 320
    assert audit["all_80_existing_contrasts_rechecked"]
    assert audit["representation_max_error"] == 0
    assert audit["max_statistic_error"] < 1e-12
    assert audit["dataset_aggregate_max_error"] == {"BA": 0, "APD": 0}
    assert len(frames["Representation Paired"]) + len(frames["APD Paired"]) == 80
    assert frames["APD Paired"].exact_signflip_p.nunique() > 1
    assert frames["APD Paired"].exact_signflip_p.between(0, 1, inclusive="neither").any()
    assert apd.value.lt(0).all() and ba.value.between(0, 1).all()
    assert rep.backbone_std.str.startswith("synthetic_encoder_").all()
    assert not curves.empty
    assert audit["inputs_sha256"] == before
    assert before == hashlib.sha256(synthetic_patch_snapshot.read_bytes()).hexdigest()


def test_patch_summary_exports_only_numeric_artifacts(tmp_path, synthetic_patch_snapshot):
    before = synthetic_patch_snapshot.read_bytes()
    output = tmp_path / "new-report"
    summary.main(["--input", str(synthetic_patch_snapshot), "--output-dir", str(output)])
    artifacts = list(output.iterdir())
    assert artifacts and {p.suffix for p in artifacts} == {".csv", ".xlsx", ".json"}
    audit = json.loads((output / "reconciliation.json").read_text())
    assert audit["max_statistic_error"] < 1e-12
    assert audit["inputs_sha256"] == hashlib.sha256(before).hexdigest()
    assert synthetic_patch_snapshot.read_bytes() == before
    original = json.loads(before)["workbook"]["tables"]
    for name, rows in original.items():
        actual = pd.read_csv(output / (name.lower().replace(" ", "_") + ".csv"), float_precision="round_trip")
        pd.testing.assert_frame_equal(actual, pd.DataFrame(rows))
    book = load_workbook(output / "patch_numerical_source.xlsx", read_only=True, data_only=True)
    try:
        assert "Representation Paired" in book.sheetnames
        assert "APD Paired" in book.sheetnames
        assert book["Representation Backbone"].max_row == 641
        assert book["BA Backbone"].max_row == 321
    finally:
        book.close()


def test_bad_patch_source_schema_is_rejected(tmp_path):
    path = tmp_path / "unsupported.json"
    path.write_text('{"schema_version":999}')
    with pytest.raises(ValueError, match="schema version"):
        summary.load_and_verify(path)


def test_patch_summary_requires_an_explicit_input(tmp_path):
    with pytest.raises(SystemExit) as caught:
        summary.main(["--output-dir", str(tmp_path / "absent")])
    assert caught.value.code == 2
    assert not (tmp_path / "absent").exists()
    assert not hasattr(summary, "DEFAULT_INPUTS") and not hasattr(summary, "INPUTS")


@pytest.mark.parametrize("sheet,column", [
    ("Representation Backbone", "Score_value"), ("BA Backbone", "BA_value"),
    ("APD Backbone", "APD_value"), ("Representation Paired", "holm_exact_p"),
    ("APD Paired", "ci_low"),
])
def test_inconsistent_synthetic_patch_tables_fail_closed(tmp_path, synthetic_patch_snapshot, sheet, column):
    payload = json.loads(synthetic_patch_snapshot.read_text())
    payload["workbook"]["tables"][sheet][0][column] += .125
    altered = tmp_path / "inconsistent.json"
    altered.write_text(json.dumps(payload), encoding="utf-8")
    before = altered.read_bytes()
    output = tmp_path / "never-exported"
    with pytest.raises(AssertionError):
        summary.main(["--input", str(altered), "--output-dir", str(output)])
    assert altered.read_bytes() == before
    assert not output.exists()


def test_patch_summary_preserves_nonempty_output_and_input(tmp_path, synthetic_patch_snapshot):
    sentinel = tmp_path / "existing.csv"
    sentinel.write_bytes(b"retain numerical result")
    before = synthetic_patch_snapshot.read_bytes()
    with pytest.raises(FileExistsError, match="new or empty"):
        summary.main(["--input", str(synthetic_patch_snapshot), "--output-dir", str(tmp_path)])
    assert sentinel.read_bytes() == b"retain numerical result"
    assert synthetic_patch_snapshot.read_bytes() == before


def test_workbook_roundtrip_preserves_payload_and_literal_text(tmp_path):
    frame = pd.DataFrame([
        ["toy", 0.123456789012345, 0.000012, True, None],
        ["=literal-not-a-formula", -2.5, 1.0, False, "text"],
    ], columns=["Model", "Score", "P", "Flag", "Optional"])
    payload = {"title": "Toy serialization contract", "sheets": [
        report.workbook_sheet("Toy Summary", frame, {"Score": "0.000%", "P": "0.000000"}),
        report.workbook_sheet("Provenance", pd.DataFrame([["Fixture", "not study data"]], columns=["Field", "Value"])),
    ]}
    source, output = tmp_path / "workbook_payload.json", tmp_path / "toy.xlsx"
    original = json.dumps(payload).encode()
    source.write_bytes(original)
    assert workbook.write_workbook(source, output) == output
    assert source.read_bytes() == original
    book = load_workbook(output, data_only=False)
    try:
        assert book.sheetnames == [sheet["name"] for sheet in payload["sheets"]]
        for sheet in payload["sheets"]:
            worksheet = book[sheet["name"]]
            assert list(worksheet.values) == [tuple(column["name"] for column in sheet["columns"]),
                                               *(tuple(row) for row in sheet["rows"])]
            for row in worksheet.iter_rows(min_row=2):
                assert [cell.number_format for cell in row] == [column["format"] for column in sheet["columns"]]
        assert book["Toy Summary"]["A3"].data_type == "s"
    finally:
        book.close()
    frozen = output.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        workbook.write_workbook(source, output)
    assert output.read_bytes() == frozen


def test_workbook_rejects_bad_rows_without_output(tmp_path):
    source, output = tmp_path / "invalid.json", tmp_path / "never.xlsx"
    source.write_text(json.dumps({"sheets": [{"name": "Toy", "columns": [{"name": "A"}], "rows": [[1, 2]]}]}))
    with pytest.raises(ValueError, match="row width"):
        workbook.write_workbook(source, output)
    assert not output.exists()


@pytest.mark.parametrize("mode", ["p2", "coverage"])
def test_patch_report_preserves_existing_output(tmp_path, monkeypatch, mode):
    sentinel = tmp_path / "existing-result.txt"
    sentinel.write_bytes(b"preserve this result")
    monkeypatch.setattr(sys, "argv", ["report", "--mode", mode, "--project-root",
                                    str(tmp_path / "missing-inputs"), "--output-dir", str(tmp_path)])
    with pytest.raises(FileExistsError, match="new or empty"):
        report.main()
    assert sentinel.read_bytes() == b"preserve this result"
    assert list(tmp_path.iterdir()) == [sentinel]


def apd_fixture():
    rows = []
    for seed, raw0, delta0, raw1, delta1 in [(42, .5, .6, .3, .4), (43, .9, .8, .6, .5)]:
        for method, zero, shifted in [("RAW", raw0, raw1), ("DELTA", delta0, delta1)]:
            for step in range(8):
                rows.append(dict(backbone_std="toy", dataset="camelyon", domain="ID", method_std=method,
                                 repetition_seed=seed, split_index=(seed - 42) * 8 + step,
                                 split_order=step, value=zero if step == 0 else shifted))
    return pd.DataFrame(rows)


def test_shared_raw_reference_is_mean_before_division():
    rows = apd_fixture()
    result = report.backbone_average_performance_drop(rows).set_index("method_std").value
    expected = (.4 + .5) / 2 / ((.5 + .9) / 2) - 1
    assert result["DELTA"] == pytest.approx(expected, abs=1e-14)
    assert result["RAW"] == pytest.approx((.3 + .6) / 2 / .7 - 1, abs=1e-14)
    assert result["DELTA"] != pytest.approx(((.4 / .6) + (.5 / .8)) / 2 - 1, abs=1e-6)
    coverage = rows.loc[rows.method_std.eq("DELTA")].assign(coverage=2)
    assert report.coverage_average_performance_drop(coverage, rows).value.iloc[0] == pytest.approx(expected, abs=1e-14)
    public = rows.rename(columns={"method_std": "method", "value": "balanced_accuracy"})
    values = evaluation.average_performance_drop(public)
    assert values.loc[values.method.eq("DELTA"), "average_performance_drop"].mean() == pytest.approx(expected, abs=1e-14)


def test_missing_or_duplicated_raw_references_fail_closed():
    rows = apd_fixture()
    delta = rows.loc[rows.method_std.eq("DELTA")]
    with pytest.raises(ValueError, match="RAW"):
        report.backbone_average_performance_drop(delta)
    with pytest.raises(ValueError, match="RAW"):
        evaluation.average_performance_drop(delta.rename(columns={"method_std": "method", "value": "balanced_accuracy"}))
    with pytest.raises(ValueError, match="duplicate"):
        report.backbone_average_performance_drop(pd.concat([rows, rows.iloc[:1]]))
    with pytest.raises(ValueError, match="incomplete"):
        report.backbone_average_performance_drop(rows.iloc[1:])
