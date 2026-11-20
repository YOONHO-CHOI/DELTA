"""Reporting-only paths, receipts and atomic table writers. No operator estimation."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
DEFAULT_NT_PATCHES = None
DEFAULT_CANCER_PATCHES = None
DEFAULT_OUTPUT_ROOT = None
SCHEMA_VERSION = "humc_normative_v13_p2_v2"
PUBLICATION_OPTIONS = {'excel': {'filename': 'HUMC_normative_screening_consolidated_results.xlsx', 'freeze_panes': 'A2', 'header_row_height': 24, 'body_row_height': 16, 'max_column_width': 42}}

def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)

def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)

def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, quoting=csv.QUOTE_MINIMAL)
    os.replace(temporary, path)

def v13_prediction_paths(output_root: Path, backbone: str, arm: str) -> dict[str, Path]:
    root = Path(output_root) / "predictions" / backbone / arm
    return {
        "root": root,
        "patch": root / "patch_scores.parquet",
        "slide": root / "slide_scores.parquet",
        "case": root / "case_scores.parquet",
        "thresholds": root / "fold_thresholds.parquet",
        "metrics": root / "metrics.json",
        "complete": root / "_COMPLETE.json",
    }

def write_consolidated_results_workbook(
    path: Path,
    sheets: dict[str, pd.DataFrame],
) -> None:
    """Atomically write the sole publication workbook for the full experiment.

    Per-backbone clinical jobs retain their CSV/Parquet artifacts and never
    concurrently touch this file.  The final multi-backbone report calls this
    function once, after every requested backbone is available.
    """

    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if not sheets:
        raise ValueError("consolidated workbook requires at least one sheet")
    if len(set(sheets)) != len(sheets):
        raise ValueError("duplicate consolidated workbook sheet names")
    invalid_names = [name for name in sheets if len(name) > 31 or any(char in name for char in "[]:*?/\\")]
    if invalid_names:
        raise ValueError(f"invalid Excel sheet names: {invalid_names}")

    workbook = Workbook()
    workbook.remove(workbook.active)
    navy = "225B50"
    green = "DDEDD8"
    light = "F5F8F6"
    thin = Side(style="thin", color="D7E4DE")
    percent_tokens = ("sensitivity", "specificity", "fraction", "power")
    for sheet_name, frame in sheets.items():
        frame = frame.copy()
        worksheet = workbook.create_sheet(sheet_name)
        worksheet.sheet_view.showGridLines = False
        worksheet.freeze_panes = str(PUBLICATION_OPTIONS["excel"]["freeze_panes"])
        if len(frame.columns):
            worksheet.auto_filter.ref = f"A1:{get_column_letter(len(frame.columns))}{len(frame) + 1}"
        for column_index, column in enumerate(frame.columns, start=1):
            cell = worksheet.cell(row=1, column=column_index, value=str(column))
            cell.fill = PatternFill("solid", fgColor=navy)
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = Border(bottom=thin)
        for row_index, row in enumerate(frame.itertuples(index=False, name=None), start=2):
            for column_index, value in enumerate(row, start=1):
                if isinstance(value, (dict, list, tuple, set)):
                    value = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
                elif pd.isna(value):
                    value = None
                elif isinstance(value, np.generic):
                    value = value.item()
                elif isinstance(value, Path):
                    value = str(value)
                cell = worksheet.cell(row=row_index, column=column_index, value=value)
                cell.fill = PatternFill("solid", fgColor=light if row_index % 2 == 0 else "FFFFFF")
                cell.alignment = Alignment(vertical="top", wrap_text=sheet_name == "Definitions")
                column_name = str(frame.columns[column_index - 1]).lower()
                if any(token in column_name for token in percent_tokens) and isinstance(value, (int, float)):
                    cell.number_format = "0.00%"
                elif isinstance(value, float):
                    cell.number_format = "0.0000"
        p_columns = [
            index + 1 for index, column in enumerate(frame.columns)
            if str(column).startswith("p_holm")
        ]
        for p_column in p_columns:
            for row_index in range(2, len(frame) + 2):
                value = worksheet.cell(row_index, p_column).value
                if isinstance(value, (int, float)) and value < 0.05:
                    worksheet.cell(row_index, p_column).fill = PatternFill("solid", fgColor=green)
        for column_index, column in enumerate(frame.columns, start=1):
            values = [str(column)] + ["" if value is None else str(value) for value in frame[column].head(200)]
            width = min(
                int(PUBLICATION_OPTIONS["excel"]["max_column_width"]),
                max(11, max(len(value) for value in values) + 2),
            )
            worksheet.column_dimensions[get_column_letter(column_index)].width = width
        worksheet.row_dimensions[1].height = float(PUBLICATION_OPTIONS["excel"]["header_row_height"])
        for row_index in range(2, len(frame) + 2):
            worksheet.row_dimensions[row_index].height = float(
                PUBLICATION_OPTIONS["excel"]["body_row_height"]
            )
    workbook.save(temporary)
    os.replace(temporary, path)

def evaluation_paths(output_root: Path, backbone: str) -> dict[str, Path]:
    root = output_root / "predictions" / backbone / "raw"
    return {
        "root": root,
        "patch": root / "patch_scores.parquet",
        "slide": root / "slide_scores.parquet",
        "case": root / "case_scores.parquet",
        "thresholds": root / "fold_thresholds.parquet",
        "metrics": root / "metrics.json",
        "complete": root / "_COMPLETE.json",
    }

def script_identity():
    directory = Path(__file__).resolve().parent
    names = ("report.py", "report_io.py", "scoring.py", "summary.py")
    return {"sha256": hashlib.sha256((directory / "report.py").read_bytes()).hexdigest(),
            "implementation": {name: {"sha256": hashlib.sha256((directory / name).read_bytes()).hexdigest()}
                               for name in names}}

