"""Portable report interfaces and the numerical contracts shared by branches."""
from __future__ import annotations

import ast
import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_report(name: str):
    path = ROOT / name
    spec = importlib.util.spec_from_file_location("test_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_survival_report_import_and_help_do_not_read_or_write_study_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = load_report("experiments/wsi_survival/report.py")
    with pytest.raises(SystemExit) as result:
        report.main(["--help"])
    assert result.value.code == 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("branch", ["raw", "reinhard"])
def test_survival_branch_names_and_comparison_contract(branch, monkeypatch):
    monkeypatch.delenv("DELTA_SURVIVAL_REPORT_OUTPUT_DIR", raising=False)
    report = load_report("experiments/wsi_survival/report.py")
    report.configure_branch(branch)
    normalized = branch == "reinhard"
    target = "Reinhard-DELTA-v1.3" if normalized else "DELTA-v1.3"
    stem = "reinhard_delta_v1_3_p2" if normalized else "delta_v1_3_p2"
    assert report.TARGET_METHOD == target
    assert report.WORKBOOK_FILENAME == "survival_manuscript_results_" + stem + ".xlsx"
    assert report.OUTPUT_DIR == report.PROJECT_ROOT / ("results/tcga_brca_wsi_survival_report_" + stem + "/backbones20")
    assert report.UNO_CONTRAST_SHEET == ("MAIN_02_ReDELTA_Contrasts" if normalized else "MAIN_02_DELTA_Contrasts")
    assert report.MAIN_COMPARATORS == ["RAW", "DANN", "Reinhard-RAW", "Reinhard-DANN"] + (["DELTA-v1.3"] if normalized else [])
    assert report.PAIRWISE_COMPARATORS == report.MAIN_COMPARATORS + report.CONTEXTUAL_COMPARATORS
    assert report.MANUSCRIPT_METHODS == ["RAW", "DANN", "Reinhard-RAW", "Reinhard-DANN", "DELTA-v1.3", "Reinhard-DELTA-v1.3"]
    assert report.MAX_T_SENSITIVITY_CONTROLS == report.MAIN_COMPARATORS
    assert report.BOOTSTRAP_REPLICATES == 20_000
    assert report.EXACT_SIGN_FLIP_MAX_BACKBONES == 21
    assert report.RANDOM_SEED == 42
    assert report.PRIMARY_P_VALUE == "p_comparison"


def test_survival_cli_configures_paths_before_running(tmp_path, monkeypatch):
    report = load_report("experiments/wsi_survival/report.py")
    observed = []
    monkeypatch.setattr(report, "run_report", lambda: observed.append((report.TARGET_METHOD, report.OUTPUT_DIR, report.SAVE_INDIVIDUAL_CSV)))
    report.main(["--branch", "reinhard", "--output-dir", str(tmp_path / "new-report"), "--no-csv"])
    assert observed == [("Reinhard-DELTA-v1.3", tmp_path / "new-report", False)]
    assert not (tmp_path / "new-report").exists()


def test_survival_pairwise_inference_retains_effect_direction_and_exact_test():
    report = load_report("experiments/wsi_survival/report.py")
    rows = []
    for backbone, value in enumerate([0.50, 0.55, 0.60, 0.65]):
        for method, gain in (("RAW", 0.0), ("DELTA-v1.3", 0.02)):
            for domain in ("in_center", "held_out_center"):
                for metric in report.HIGHER_IS_BETTER:
                    rows.append(dict(backbone=f"b{backbone}", split_id="fixed", method=method,
                                     domain=domain, metric=metric, value=value + gain, status="complete"))
    result = report.compare_target_pairwise(pd.DataFrame(rows), target="DELTA-v1.3",
                                           controls=["RAW"], bootstrap_replicates=64, seed=42)
    assert len(result) == 8
    expected = np.where(result.metric.eq("integrated_brier_score"), -0.02, 0.02)
    np.testing.assert_allclose(result.favorable_delta_mean, expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(result.ci95_low, expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(result.ci95_high, expected, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(result.p_comparison, np.full(8, 0.125))
    assert result.n_backbones.eq(4).all() and result.permutations.eq(16).all()
    assert result.permutation_mode.eq("exact").all()
    assert not result.multiplicity_adjusted.any()


def test_screening_report_imports_the_canonical_statistics():
    path = ROOT / "experiments/humc_normative/report.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    required = {"bootstrap_backbone_mean", "hierarchical_four_arm_inference"}
    imports = {alias.name for node in tree.body if isinstance(node, ast.ImportFrom)
               and node.module == "scoring" for alias in node.names}
    assert required.issubset(imports)
    definitions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert not required.intersection(definitions)


def test_screening_cli_exposes_only_the_four_arm_report():
    path = ROOT / "experiments/humc_normative/report.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "build_parser"]
    namespace = dict(argparse=argparse, Path=Path, BACKBONES=("toy",),
                     DEFAULT_NT_PATCHES=Path("non-tumor"), DEFAULT_CANCER_PATCHES=Path("cancer"),
                     DEFAULT_OUTPUT_ROOT=Path("new-report"), command_v13_report=lambda args: None)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    parser = namespace["build_parser"]()
    args = parser.parse_args(["report", "--backbones", "toy", "--baseline-root", "controls",
                              "--output-root", "reports", "--color-audit", "colors.parquet"])
    assert args.func is namespace["command_v13_report"]
    assert args.bootstrap_reps == 20_000
    assert args.baseline_root == Path("controls")
    for legacy in ("v13-report", "v13-clinical-report"):
        with pytest.raises(SystemExit):
            parser.parse_args([legacy])


def test_screening_shared_helpers_preserve_workbook_options():
    report = ast.parse((ROOT / "experiments/humc_normative/report.py").read_text(encoding="utf-8"))
    runner = ast.parse((ROOT / "experiments/humc_normative/report_io.py").read_text(encoding="utf-8"))
    helpers = {"write_consolidated_results_workbook", "PUBLICATION_OPTIONS"}
    imports = {alias.name for node in report.body if isinstance(node, ast.ImportFrom)
               and node.module == "report_io" for alias in node.names}
    assert helpers.issubset(imports)
    options = next(node for node in runner.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "PUBLICATION_OPTIONS" for target in node.targets))
    assert ast.literal_eval(options.value) == {"excel": {
        "filename": "HUMC_normative_screening_consolidated_results.xlsx",
        "freeze_panes": "A2", "header_row_height": 24,
        "body_row_height": 16, "max_column_width": 42,
    }}


@pytest.mark.parametrize("path", [
    "experiments/wsi_survival/report.py", "experiments/wsi_survival/merge.py",
    "experiments/wsi_survival/summary.py", "experiments/humc_normative/report.py",
    "experiments/humc_normative/summary.py", "experiments/humc_normative/report_io.py",
    "experiments/bank_coverage/report.py",
])
def test_clinical_reporting_has_no_image_rendering_dependencies(path):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    assert not any(name.startswith(("matplotlib", "seaborn", "plotly", "visualization")) for name in modules)
    assert not any(isinstance(node, ast.Attribute) and node.attr in {"savefig", "subplots"}
                   for node in ast.walk(tree))


def manuscript_contrasts():
    report = load_report("experiments/wsi_survival/summary.py")
    backbones = [f"b{index}" for index in range(20)]
    rows = []
    for target, controls in report.TCGA_COMPARATORS:
        for domain, _ in report.DOMAINS:
            for metric, _ in report.METRICS_TCGA:
                for control, p in zip(controls, [.01, .04, .2, .9]):
                    ibs = metric == "integrated_brier_score"
                    rows.append({"target": target, "control": control, "domain": domain,
                                 "metric": metric, "p_raw": p, "n_backbones": 20,
                                 "common_backbones": ";".join(backbones),
                                 "target_mean": .48 if ibs else .52, "control_mean": .5,
                                 "favorable_delta_mean": .02, "ci95_low": .01, "ci95_high": .03})
    return report, backbones, pd.DataFrame(rows)


def test_survival_manuscript_holm_uses_four_controls_and_preserves_source_values():
    report, backbones, frame = manuscript_contrasts()
    before = frame.copy(deep=True)
    result = report.tcga_holm_pairwise(frame)
    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_frame_equal(result[before.columns], before)
    assert len(result) == 64
    assert result.holm_family_size.eq(4).all()
    expected = np.tile([.04, .12, .4, .9], 16)
    np.testing.assert_allclose(result.p_holm_within_family, expected, rtol=0, atol=1e-14)
    rows = report.tcga_comparison_rows(result, backbones)
    assert {key: len(value) for key, value in rows.items()} == {"in_center": 8, "held_out_center": 8}
    for entries in rows.values():
        for row in entries:
            ibs = row["effects"]["integrated_brier_score"]
            assert ibs["mean_pp"] == pytest.approx(-2)
            assert (ibs["ci95_low_pp"], ibs["ci95_high_pp"]) == pytest.approx((-3, -1))
            assert not {"color", "label"}.intersection(row)
    effects = report.tcga_effects(result)
    assert effects[(0, "in_center", "integrated_brier_score")][:3] == pytest.approx((-.02, -.03, -.01))


def test_survival_manuscript_holm_rejects_missing_comparator():
    report, _, frame = manuscript_contrasts()
    with pytest.raises(ValueError, match="Incomplete TCGA Holm family"):
        report.tcga_holm_pairwise(frame.iloc[1:])


def coverage_fixture():
    report = load_report("experiments/bank_coverage/report.py")
    rows = []
    controls = []
    for backbone in report.BACKBONES:
        for branch in report.BRANCHES:
            for domain in report.DOMAINS:
                for metric in report.RAW_METRICS:
                    value = {"td_auc_q25": .6, "td_auc_q50": .7, "td_auc_q75": .8}.get(metric, .5)
                    controls.append(dict(backbone=backbone, split_id="fixed", method=report.MATCHED_BASELINE[branch],
                                         domain=domain, metric=metric, value=value, status="complete"))
                    for fraction in report.BANK_FRACTIONS:
                        rows.append(dict(backbone=backbone, branch=branch, bank_fraction_percent=fraction,
                                         split_id="fixed", method=report.BRANCH_LABELS[branch], domain=domain,
                                         metric=metric, value=value + fraction / 1000, status="complete"))
    return report, pd.DataFrame(rows), pd.DataFrame(controls)


def test_coverage_report_preserves_three_horizon_mean_and_all_source_rows(tmp_path):
    report, frame, controls = coverage_fixture()
    candidates = report.add_mean_time_dependent_auc(frame)
    references = report.add_mean_time_dependent_auc(controls)
    means = candidates.loc[candidates.metric.eq("mean_time_dependent_auc")]
    np.testing.assert_allclose(means.value, .7 + means.bank_fraction_percent / 1000)
    pd.testing.assert_frame_equal(report.add_mean_time_dependent_auc(candidates), candidates)
    source = report.source_table(candidates, references)
    assert len(candidates) == 1120 and len(source) == 720
    assert source.record_type.value_counts().to_dict() == {"bank_coverage": 640, "matched_baseline": 80}
    curve = report.candidate_curve(candidates, backbone="conch", branch="raw",
                                   domain="in_center", metric="mean_time_dependent_auc")
    np.testing.assert_allclose(curve, .7 + np.asarray(report.BANK_FRACTIONS) / 1000)
    pytest.importorskip("openpyxl")
    report.build_workbook(candidates, references, source, tmp_path / "coverage.xlsx")
    sheets = pd.read_excel(tmp_path / "coverage.xlsx", sheet_name=None)
    assert list(sheets) == ["README", "Coverage_Summary", "Figure_Source", "Candidate_Long", "Matched_Controls", "QC"]
    assert len(sheets["Candidate_Long"]) == 1120
    assert sheets["QC"].Result.eq("PASS").all()


def test_coverage_report_rejects_incomplete_auc_horizons():
    report, frame, _ = coverage_fixture()
    bad = frame.drop(frame.index[frame.metric.eq("td_auc_q75")][0])
    with pytest.raises(ValueError, match="all three locked AUC horizons"):
        report.add_mean_time_dependent_auc(bad)

