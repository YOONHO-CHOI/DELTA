"""Synthetic molecular-evaluation regressions without correction or model code."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.stpbench_molecular_prediction import external_statistics as external
from experiments.stpbench_molecular_prediction import report, statistics

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "tests/fixtures/molecular_statistics.json").read_text())


def assert_record(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, float):
            assert actual[key] == pytest.approx(value, rel=0, abs=1e-12), key
        else:
            assert actual[key] == value, key


def test_section_statistics_match_pre_refactor_fixture():
    actual = statistics.task_statistics(pd.DataFrame(FIXTURE["section_inputs"]), "synthetic", 128, 1000, 42)
    assert_record(actual, FIXTURE["section_expected"])


def test_patient_statistics_match_pre_refactor_fixture():
    assert_record(external.cluster_statistics(pd.DataFrame(FIXTURE["patient_inputs"]), "pcc", 128, 42),
                  FIXTURE["patient_expected"])


def test_statistics_default_schema_resolves_only_prepared_inputs(monkeypatch, tmp_path):
    prepared = tmp_path / "supplied-prepared-assets"
    config = {"paths": {"prepared_asset_root_env": "SYNTHETIC_STP_ASSETS",
                        "defaults": {"prepared_asset_root": "inputs/prepared"}}}
    monkeypatch.setenv("SYNTHETIC_STP_ASSETS", str(prepared))
    raw, actual = statistics.statistics_input_roots(config, tmp_path / "run")
    assert raw is None and actual == prepared.resolve()
    assert not prepared.exists()


def test_statistics_explicit_recorded_roots_are_not_replaced(tmp_path):
    prepared, raw = tmp_path / "prepared", tmp_path / "raw-results"
    config = {"paths": {"prepared_asset_root": str(prepared), "official_raw_result_root": str(raw)}}
    assert statistics.statistics_input_roots(config, tmp_path / "run") == (raw.resolve(), prepared.resolve())
    with pytest.raises(ValueError, match="Prepared STP-Bench asset root"):
        statistics.statistics_input_roots({"paths": {}}, tmp_path / "run")


def test_prediction_relocation_and_truth_paths_remain_explicit(tmp_path):
    relocated = tmp_path / "relocated" / "01_runs" / "source" / "panel" / "prediction"
    relocated.mkdir(parents=True)
    result = {"results": [{"artifacts": {"folds": {"0": {"prediction_dir": "/absent/01_runs/source/panel/prediction"}}}}]}
    assert statistics.prediction_dir(result, 0, tmp_path / "relocated", "source", "panel") == relocated
    with pytest.raises(FileNotFoundError, match="Prediction directory"):
        statistics.prediction_dir(result, 0)
    with pytest.raises(RuntimeError, match="Truth data root"):
        statistics.truth_root({}, 0)


def test_gene_mapping_preserves_exact_symbols_and_unambiguous_ensembl():
    genes = pd.DataFrame({"symbol": ["A", "B", "DUP", "DUP"],
                          "ensembl_gene_id": ["ENSG1", "ENSG2", "ENSG3", "ENSG4"]})
    assert external.approved_index(["a", "ENSG2.4", "DUP", "unknown"], genes) == {"A": 0, "B": 1}
    with pytest.raises(ValueError, match="Ambiguous target gene"):
        external.approved_index(["A", "ENSG1.2"], genes)


def test_external_cohort_statistics_preserve_sign_units_and_holm():
    original = pd.DataFrame(FIXTURE["patient_inputs"])
    original["mae_raw"] = 1.0
    original["mae_delta"] = 0.9
    groups = []
    for cohort in ("cohort-a", "cohort-b"):
        group = original.copy()
        group["cohort"], group["panel"] = cohort, "panel-a"
        groups.append(group)
    frame = pd.concat(groups, ignore_index=True)
    result = external.summarize_predictions(frame, 128, 42)
    assert len(result) == 4
    pcc = result[result.metric.eq("pcc")]
    np.testing.assert_allclose(pcc.effect, FIXTURE["patient_expected"]["effect"], atol=1e-12)
    assert pcc.patients.eq(3).all() and pcc.sections.eq(5).all()
    expected = statistics.holm(pcc.p)
    np.testing.assert_array_equal(pcc.holm_p, expected)
    assert result.holm_family.nunique() == 2
    assert result[result.metric.eq("mae")].effect.gt(0).all()
    bad = frame.copy(); bad.loc[0, "pcc_raw"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        external.summarize_predictions(bad)


def write_shard(root, dataset, fold, panel, arm, predictor="LinearProb", external_pairs=()):
    path = root / "shards" / dataset / f"fold_{fold}" / panel / arm
    path.mkdir(parents=True)
    marker = dict(status="complete", dataset=dataset, fold=fold, panel=panel, arm=arm,
                  canary=False, checkpoint_sha256="synthetic-checkpoint", completed_at="synthetic")
    result = {"summary": {predictor: {"metrics": {str(fold): {
        "test_PearsonCorrCoef": .3 if arm == "raw" else .4,
        "test_MeanAbsoluteError": .8 if arm == "raw" else .7}}}}}
    workflow = {**marker, "internal": result, "external": {pair: result for pair in external_pairs}}
    (path / "_COMPLETE.json").write_text(json.dumps(marker))
    (path / "workflow_result.json").write_text(json.dumps(workflow))
    return path


def test_completed_report_audits_shards_and_exports_without_core(tmp_path):
    import yaml
    config = {"upstream": {"predictor": "LinearProb"}, "internal_datasets": {"source": {"n_folds": 1}},
              "panels": {"panel": {}}, "arms": {"raw": {}, "delta": {}},
              "external_pairs": {"transfer": {"source": "source", "source_fold": 0, "target": "target"}}}
    config_path = tmp_path / "config.yaml"; config_path.write_text(yaml.safe_dump(config))
    root, output = tmp_path / "run", tmp_path / "report"
    for arm in ("raw", "delta"):
        write_shard(root, "source", 0, "panel", arm, external_pairs=("transfer",))
    report.build_report(config_path, root, output)
    paired = pd.read_csv(output / "internal_paired_differences.csv")
    np.testing.assert_allclose(paired.delta_minus_raw_pcc, .1)
    np.testing.assert_allclose(paired.raw_minus_delta_mae, .1)
    assert json.loads((output / "audit.json").read_text())["expected_shards"] == 2
    bad = root / "shards/source/fold_0/panel/delta/_COMPLETE.json"
    value = json.loads(bad.read_text()); value["checkpoint_sha256"] = "tampered"
    bad.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="Checkpoint hash mismatch"):
        report.load_complete_shard(root, "source", 0, "panel", "delta")


def test_molecular_release_contains_no_correction_implementation():
    folder = ROOT / "experiments/stpbench_molecular_prediction"
    assert not (folder / "run.py").exists() and not (folder / "external.py").exists()
    assert "projection_inputs" not in FIXTURE and "projection_expected" not in FIXTURE
    for path in folder.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not node.module or not node.module.startswith("delta")
            elif isinstance(node, ast.Import):
                assert all(not alias.name.startswith("delta") for alias in node.names)
