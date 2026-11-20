"""Export completed molecular-result CSV tables without recalculating statistics."""
from __future__ import annotations

import argparse
import hashlib
from itertools import chain
from pathlib import Path
import tempfile

import pandas as pd


def export_tables(input_dir: Path, output: Path) -> Path:
    from openpyxl import Workbook
    from openpyxl.cell import WriteOnlyCell

    input_dir, output = Path(input_dir).resolve(), Path(output).resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    if output.suffix.lower() != ".xlsx":
        raise ValueError("Output must be an .xlsx workbook")
    if output.exists():
        raise FileExistsError("Choose a new workbook path")
    sources = sorted(input_dir.glob("*.csv"))
    if not sources:
        raise ValueError("No completed CSV tables in the supplied directory")
    book = Workbook(write_only=True)
    ledger = []
    for index, source in enumerate(sources, 1):
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        columns = pd.read_csv(source, nrows=0).columns
        identity_columns = {"dataset", "task", "pair", "panel", "arm", "section", "patient", "patient_id", "sample_id", "slide_id", "gene", "split_id"}
        frame = pd.read_csv(source, float_precision="round_trip",
                            dtype={name: str for name in columns if name in identity_columns})
        if len(frame) >= 1_048_576 or len(frame.columns) > 16_384:
            raise ValueError(f"Table exceeds Excel limits: {source.name}")
        title = f"{index:02d}_{source.stem}"[:31]
        sheet = book.create_sheet(title)
        for row in chain([list(frame.columns)], frame.itertuples(index=False, name=None)):
            cells = []
            for value in row:
                if pd.isna(value):
                    value = None
                cell = WriteOnlyCell(sheet, value=value)
                if isinstance(value, str):
                    cell.data_type = "s"
                cells.append(cell)
            sheet.append(cells)
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source changed during export: {source}")
        ledger.append([source.name, title, len(frame), digest])
    audit = book.create_sheet("Sources")
    audit.append(["CSV file", "Worksheet", "Rows", "SHA256"])
    for row in ledger:
        audit.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".molecular_tables_", suffix=".xlsx", dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        book.save(temporary)
        if output.exists():
            raise FileExistsError("Output appeared during export")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(export_tables(args.input_dir, args.output))


if __name__ == "__main__":
    main()
