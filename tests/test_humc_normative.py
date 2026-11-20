"""Normative scoring and reporting tests on artificial values only."""
import ast
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from experiments.humc_normative import scoring, report_io, report


def test_case_aggregation_and_holm_contract():
    assert scoring.top_fraction_mean(np.arange(1000)) == np.mean(np.arange(990, 1000))
    raw = np.asarray([.03, .001, .2, .04])
    adjusted = scoring.holm_adjust(raw)
    np.testing.assert_allclose(adjusted, [.09, .004, .2, .09])


def test_case_bootstrap_keeps_constant_paired_effect():
    low, high = scoring.bootstrap_backbone_mean(np.repeat(.125, 20), 100, 42)
    assert low == high == .125


def test_public_report_requires_external_color_table():
    parser = report.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["report", "--output-root", "synthetic-output"])
    args = parser.parse_args(["report", "--output-root", "synthetic-output", "--color-audit", "synthetic.parquet"])
    assert args.color_audit == Path("synthetic.parquet")


def test_atomic_workbook_preserves_supplied_numeric_cells(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    output = tmp_path / "numeric.xlsx"
    report_io.write_consolidated_results_workbook(output, {"Synthetic": pd.DataFrame({"score": [.25, .75]})})
    book = openpyxl.load_workbook(output, read_only=True, data_only=True)
    try:
        assert list(book["Synthetic"].values) == [("score",), (.25,), (.75,)]
    finally:
        book.close()


def test_humc_modules_have_no_operator_or_transform_dependencies():
    root = Path(report.__file__).parent
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("delta")
            if isinstance(node, ast.FunctionDef):
                assert node.name not in {"fit_delta", "project_features", "delta_counterfactual_features", "control_bases"}
