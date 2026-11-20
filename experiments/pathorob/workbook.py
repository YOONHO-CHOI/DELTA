#!/usr/bin/env python3
"""Export existing PathoROB workbook_payload.json sheets with public openpyxl.

This is a serialization step only: values, sheet order/names and column number
formats come directly from the payload. It never estimates study statistics.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font


def write_workbook(payload_path: Path, output: Path) -> Path:
    """Write a new workbook without replacing an existing result."""
    payload_path, output = Path(payload_path), Path(output)
    if output.suffix.lower() != ".xlsx":
        raise ValueError("Workbook output must have an .xlsx extension")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing workbook: {output}")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    sheets = payload.get("sheets")
    if not isinstance(sheets, list) or not sheets:
        raise ValueError("Payload must contain a nonempty sheets list")

    workbook = Workbook()
    workbook.remove(workbook.active)
    workbook.properties.title = str(payload.get("title", "PathoROB results"))
    names: set[str] = set()
    for sheet in sheets:
        name = sheet["name"]
        if (not isinstance(name, str) or not name or len(name) > 31
                or any(char in name for char in "\\/*?:[]") or name.casefold() in names):
            raise ValueError(f"Invalid or duplicate Excel sheet name: {name!r}")
        names.add(name.casefold())
        columns, rows = sheet["columns"], sheet["rows"]
        if not isinstance(columns, list) or not columns or not isinstance(rows, list):
            raise ValueError(f"Invalid columns/rows in sheet {name!r}")
        headers = [column["name"] for column in columns]
        formats = [column.get("format", "General") for column in columns]
        if not all(isinstance(value, str) for value in [*headers, *formats]):
            raise ValueError(f"Column names and formats must be strings: {name!r}")
        worksheet = workbook.create_sheet(name)
        worksheet.append(headers)
        for cell in worksheet[1]:
            cell.data_type = "s"
            cell.font = Font(bold=True)
        worksheet.freeze_panes = "A2"
        for row in rows:
            if not isinstance(row, list) or len(row) != len(columns):
                raise ValueError(f"Payload row width does not match columns: {name!r}")
            if any(not isinstance(value, (str, int, float, bool, type(None)))
                   or isinstance(value, float) and not math.isfinite(value) for value in row):
                raise ValueError(f"Payload cells must be finite JSON scalars: {name!r}")
            worksheet.append(row)
            for cell, value, number_format in zip(worksheet[worksheet.max_row], row, formats):
                # JSON strings are literal text, never executable Excel formulas.
                if isinstance(value, str):
                    cell.data_type = "s"
                cell.number_format = number_format

    output.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        # Exclusive creation also closes the race after the existence preflight.
        with output.open("xb") as stream:
            created = True
            workbook.save(stream)
    except Exception:
        if created:
            output.unlink()
        raise
    finally:
        workbook.close()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True, help="Existing workbook_payload.json")
    parser.add_argument("--output", type=Path, required=True, help="New .xlsx file; never overwritten")
    args = parser.parse_args()
    print(write_workbook(args.payload, args.output))


if __name__ == "__main__":
    main()
