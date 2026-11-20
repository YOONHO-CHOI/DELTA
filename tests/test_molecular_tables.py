from pathlib import Path

import pandas as pd
import pytest

from experiments.stpbench_molecular_prediction.tables import export_tables


def test_csv_workbook_preserves_values_and_inputs(tmp_path: Path):
    openpyxl = pytest.importorskip("openpyxl")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    frame = pd.DataFrame({"task": ["=literal", "cohort"], "pcc": [0.123456789012345, -0.002], "holm_p": [0.000012, 1.0], "section": ["001", "002"]})
    source = inputs / "internal_task_statistics.csv"
    frame.to_csv(source, index=False)
    original = source.read_bytes()
    target = tmp_path / "report.xlsx"
    export_tables(inputs, target)
    book = openpyxl.load_workbook(target, data_only=False)
    try:
        sheet = book[book.sheetnames[0]]
        assert list(sheet.values) == [tuple(frame.columns), *frame.itertuples(index=False, name=None)]
        assert sheet["A2"].data_type == "s"
        assert book["Sources"]["C2"].value == 2
    finally:
        book.close()
    assert source.read_bytes() == original
    with pytest.raises(FileExistsError):
        export_tables(inputs, target)


def test_empty_csv_directory_fails_without_an_output(tmp_path: Path):
    pytest.importorskip("openpyxl")
    target = tmp_path / "report.xlsx"
    with pytest.raises(ValueError, match="No completed CSV"):
        export_tables(tmp_path, target)
    assert not target.exists()
