"""Combine the two frozen TCGA-BRCA WSI survival report workbooks.

This is a presentation-only step. It copies the existing statistical tables;
it does not recompute endpoints, confidence intervals, or P values. The two
source workbooks remain untouched. Run ``--check-only`` before creating the
unified workbook.

Example (paths may be local mounts or server paths)::

    python -m experiments.wsi_survival.merge \
      --delta-workbook DELTA.xlsx --reinhard-workbook ReinhardDELTA.xlsx \
      --canonical-main MAIN_RUN --controls-root CONTROL_RESULTS \
      --output-dir MAIN_RUN/report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from copy import copy
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.workbook.workbook import Workbook
from openpyxl.worksheet.worksheet import Worksheet


WORKBOOK_NAME = "tcga_brca_wsi_survival_manuscript_results.xlsx"
MANIFEST_NAME = "report_manifest.json"
COMMON_SHEETS = (
    "MAIN_01_Uno",
    "SUPP_01_All_Endpoints",
    "SUPP_04_Paired_Summary",
    "SUPP_05_Coverage",
    "SUPP_06_Backbones",
    "SUPP_07_Provenance",
    "SUPP_08_Analysis_Long",
)
DELTA_SHEETS = {
    "MAIN_02_DELTA_Contrasts": "MAIN_02_DELTA_Uno",
    "MAIN_03_Harrell_Contrasts": "MAIN_03_DELTA_Harrell",
    "MAIN_04_IBS_Contrasts": "MAIN_04_DELTA_IBS",
    "MAIN_05_tdAUC_Contrasts": "MAIN_05_DELTA_tdAUC",
    "SUPP_02_All_Pairwise": "SUPP_02_DELTA_Pairwise",
    "SUPP_03_MaxT_Sensitivity": "SUPP_03_DELTA_MaxT",
}
REINHARD_SHEETS = {
    "MAIN_02_ReDELTA_Contrasts": "MAIN_06_ReDELTA_Uno",
    "MAIN_03_Harrell_Contrasts": "MAIN_07_ReDELTA_Harrell",
    "MAIN_04_IBS_Contrasts": "MAIN_08_ReDELTA_IBS",
    "MAIN_05_tdAUC_Contrasts": "MAIN_09_ReDELTA_tdAUC",
    "SUPP_02_All_Pairwise": "SUPP_09_ReDELTA_Pairwise",
    "SUPP_03_MaxT_Sensitivity": "SUPP_10_ReDELTA_MaxT",
}
AVAILABILITY_NAME = "SUPP_05_Method_Availability"
EXPECTED_METHODS = {"DELTA-v1.3", "Reinhard-DELTA-v1.3"}
EXPECTED_BACKBONES = 20
EXPECTED_ENDPOINTS = 480
EXPECTED_ANALYSIS_ROWS = 2520


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(sheet: Worksheet) -> list[tuple[object, ...]]:
    return list(sheet.values)


def check_static_workbook(workbook: Workbook, expected: set[str]) -> None:
    if set(workbook.sheetnames) != expected:
        raise ValueError(
            f"Unexpected sheets: missing={sorted(expected - set(workbook.sheetnames))}, "
            f"extra={sorted(set(workbook.sheetnames) - expected)}"
        )
    if workbook.defined_names or workbook._external_links:
        raise ValueError("Source workbook has defined names or external links")
    for sheet in workbook:
        if sheet.tables or sheet._charts or sheet._images or sheet.conditional_formatting:
            raise ValueError(f"Unsupported workbook object in {sheet.title}")
        for row in sheet:
            for cell in row:
                if cell.data_type in {"f", "e"}:
                    raise ValueError(f"Formula or Excel error in {sheet.title}!{cell.coordinate}")


def check_shared_sheets(delta: Workbook, reinhard: Workbook) -> None:
    for name in COMMON_SHEETS:
        if rows(delta[name]) != rows(reinhard[name]):
            raise ValueError(f"Shared source sheets differ: {name}")
    if delta["MAIN_01_Uno"].max_row != 12:
        raise ValueError("Unexpected common-backbone Uno table shape")
    if delta["SUPP_08_Analysis_Long"].max_row != EXPECTED_ANALYSIS_ROWS + 1:
        raise ValueError("Unexpected long-form analysis row count")


def check_target_sheet(sheet: Worksheet, target: str) -> None:
    if sheet.title.startswith("MAIN_"):
        if target not in str(sheet.cell(2, 3).value):
            raise ValueError(f"Wrong target in {sheet.title}")
    elif sheet.title == "SUPP_02_All_Pairwise":
        if {row[0] for row in rows(sheet)[1:]} != {target}:
            raise ValueError(f"Wrong pairwise target in {sheet.title}")
        if {row[26] for row in rows(sheet)[1:]} != {False}:
            raise ValueError("Primary pairwise P values must remain unadjusted")
    elif sheet.title == "SUPP_03_MaxT_Sensitivity":
        if {row[0] for row in rows(sheet)[1:]} != {target}:
            raise ValueError(f"Wrong maxT target in {sheet.title}")


def verify_sources(
    delta: Workbook, canonical_main: Path, controls_root: Path
) -> dict[str, object]:
    combined = canonical_main / "results" / "delta_v1_3_p2.csv"
    promotion = canonical_main / "audit" / "final_promotion.json"
    validation = canonical_main / "audit" / "final_validation.json"
    for path in (combined, promotion, validation):
        if not path.is_file():
            raise FileNotFoundError(path)
    combined_hash = sha256(combined)
    for audit_path in (promotion, validation):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (
            audit.get("status") != "complete"
            or audit.get("results_sha256") != combined_hash
            or audit.get("backbone_count") != EXPECTED_BACKBONES
            or audit.get("endpoint_count") != EXPECTED_ENDPOINTS
            or audit.get("pair_count") != 55_988
            or audit.get("rank_input_rows") != 55_988
            or audit.get("projection") != "centered_euclidean_hard_projection"
        ):
            raise ValueError(f"Canonical audit mismatch: {audit_path}")

    provenance = rows(delta["SUPP_07_Provenance"])
    columns = {name: index for index, name in enumerate(provenance[0])}
    checked: dict[str, dict[str, str]] = {"controls": {}, "delta": {}}
    for record in provenance[1:]:
        backbone = str(record[columns["backbone"]])
        source_hash = record[columns["sha256"]]
        source_methods = str(record[columns["source_methods"]] or "")
        if "RAW" in source_methods.split(";"):
            kind = "controls"
            path = controls_root / backbone / "survival_metrics_long.csv"
        else:
            kind = "delta"
            path = canonical_main / "artifacts" / backbone / "ratio_results_delta_v1_3_p2.csv"
            if record[columns["combined_result_sha256"]] != combined_hash:
                raise ValueError(f"Combined-result hash mismatch for {backbone}")
        if not path.is_file() or sha256(path) != source_hash:
            raise ValueError(f"Source hash mismatch: {path}")
        if backbone in checked[kind]:
            raise ValueError(f"Duplicate {kind} provenance: {backbone}")
        checked[kind][backbone] = source_hash
    if any(len(group) != EXPECTED_BACKBONES for group in checked.values()):
        raise ValueError("Expected 20 control files and 20 DELTA result files")
    if set(checked["controls"]) != set(checked["delta"]):
        raise ValueError("Control and DELTA backbone sets differ")

    analysis = rows(delta["SUPP_08_Analysis_Long"])
    analysis_columns = {name: index for index, name in enumerate(analysis[0])}
    if "openpath" in {str(r[analysis_columns["backbone"]]).lower() for r in analysis[1:]}:
        raise ValueError("OpenPath must not enter the final 20-backbone report")
    if not EXPECTED_METHODS.issubset(
        {str(r[analysis_columns["method"]]) for r in analysis[1:]}
    ):
        raise ValueError("Both final DELTA branches are required")
    return {
        "combined_result": str(combined),
        "combined_result_sha256": combined_hash,
        "promotion_audit_sha256": sha256(promotion),
        "validation_audit_sha256": sha256(validation),
        "backbones": sorted(checked["delta"]),
        "source_hashes": checked,
    }


def copy_sheet(source: Worksheet, workbook: Workbook, name: str) -> Worksheet:
    target = workbook.create_sheet(name)
    for row in source:
        for cell in row:
            destination = target[cell.coordinate]
            destination.value = cell.value
            if cell.has_style:
                destination.font = copy(cell.font)
                destination.fill = copy(cell.fill)
                destination.border = copy(cell.border)
                destination.alignment = copy(cell.alignment)
                destination.number_format = cell.number_format
                destination.protection = copy(cell.protection)
            if cell.hyperlink:
                destination._hyperlink = copy(cell.hyperlink)
            if cell.comment:
                destination.comment = copy(cell.comment)
    for key, dimension in source.row_dimensions.items():
        target.row_dimensions[key] = copy(dimension)
    for key, dimension in source.column_dimensions.items():
        target.column_dimensions[key] = copy(dimension)
    for merged in source.merged_cells.ranges:
        target.merge_cells(str(merged))
    target.freeze_panes = source.freeze_panes
    target.sheet_format = copy(source.sheet_format)
    target.sheet_properties = copy(source.sheet_properties)
    target.views = copy(source.views)
    target.sheet_state = source.sheet_state
    target.print_options = copy(source.print_options)
    target.page_margins = copy(source.page_margins)
    target.page_setup = copy(source.page_setup)
    target.auto_filter.ref = source.auto_filter.ref
    return target


def build_readme(delta: Workbook, reinhard: Workbook) -> None:
    sheet = delta["README"]
    original = rows(sheet)
    delta_descriptions = {str(key): str(value) for key, value in original[12:]}
    reinhard_descriptions = {
        str(key): str(value) for key, value in rows(reinhard["README"])[12:]
    }
    delta_original = {value: key for key, value in DELTA_SHEETS.items()}
    reinhard_original = {value: key for key, value in REINHARD_SHEETS.items()}
    readme = list(original[:11])
    readme.append(("Effect direction", "Positive favors the named target; IBS is comparator minus target"))
    readme.append(("Scope", "20-backbone TCGA-BRCA final p2 report; bank-fraction development and mechanistic controls are separate experiments"))
    readme.append(("Provenance paths", "SUPP_07 records generation-time paths; their hashes match current canonical results and controls"))
    for name in delta.sheetnames:
        if name == "README":
            continue
        original_name = next((old for old, new in DELTA_SHEETS.items() if new == name), name)
        if name == AVAILABILITY_NAME:
            description = "Number of available backbones per method/domain/metric; not the p1-p50 bank-fraction experiment"
        elif name in reinhard_original:
            description = reinhard_descriptions[reinhard_original[name]]
        elif name in delta_original:
            description = delta_descriptions[delta_original[name]]
        else:
            description = delta_descriptions.get(original_name, "")
        readme.append((name, description))
    for row in sheet:
        for cell in row:
            cell.value = None
    for index, (label, definition) in enumerate(readme, 1):
        for column, value in ((1, label), (2, definition)):
            cell = sheet.cell(index, column)
            cell.value = value
            if index > 25:
                reference = sheet.cell(24, column)
                cell.font = copy(reference.font)
                cell.fill = copy(reference.fill)
                cell.border = copy(reference.border)
                cell.alignment = copy(reference.alignment)
                cell.number_format = reference.number_format


def create_unified(
    delta: Workbook, reinhard: Workbook, output: Path
) -> dict[str, dict[str, str]]:
    expected_rows = {
        ("DELTA", name): rows(delta[name]) for name in (*COMMON_SHEETS, *DELTA_SHEETS)
    }
    expected_rows.update(
        {("Reinhard-DELTA", name): rows(reinhard[name]) for name in REINHARD_SHEETS}
    )
    for old, new in DELTA_SHEETS.items():
        delta[old].title = new
    delta["SUPP_05_Coverage"].title = AVAILABILITY_NAME
    source_map: dict[str, dict[str, str]] = {
        new: {"source": "DELTA", "sheet": old} for old, new in DELTA_SHEETS.items()
    }
    for name in COMMON_SHEETS:
        current = AVAILABILITY_NAME if name == "SUPP_05_Coverage" else name
        source_map[current] = {"source": "shared", "sheet": name}
    for old, new in REINHARD_SHEETS.items():
        copy_sheet(reinhard[old], delta, new)
        source_map[new] = {"source": "Reinhard-DELTA", "sheet": old}
    source_map["README"] = {"source": "combined", "sheet": "README"}
    order = [
        "README", "MAIN_01_Uno",
        *list(DELTA_SHEETS.values())[:4],
        *list(REINHARD_SHEETS.values())[:4],
        "SUPP_01_All_Endpoints",
        DELTA_SHEETS["SUPP_02_All_Pairwise"],
        REINHARD_SHEETS["SUPP_02_All_Pairwise"],
        DELTA_SHEETS["SUPP_03_MaxT_Sensitivity"],
        REINHARD_SHEETS["SUPP_03_MaxT_Sensitivity"],
        "SUPP_04_Paired_Summary", AVAILABILITY_NAME,
        "SUPP_06_Backbones", "SUPP_07_Provenance", "SUPP_08_Analysis_Long",
    ]
    if set(order) != set(delta.sheetnames) or len(order) != 20:
        raise ValueError("Unified sheet inventory is incomplete")
    delta._sheets = [delta[name] for name in order]
    build_readme(delta, reinhard)
    delta.properties.title = "TCGA-BRCA WSI survival results: DELTA and Reinhard-DELTA"
    delta.properties.subject = "Final p2 survival analysis on 20 backbones"
    delta.properties.creator = "DELTA reporting pipeline"
    temporary = output.with_name(output.name + ".partial.xlsx")
    if output.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to replace existing report: {output}")
    delta.save(temporary)
    reopened = load_workbook(temporary, data_only=False)
    if reopened.sheetnames != order:
        raise ValueError("Saved sheet order changed")
    for name, lineage in source_map.items():
        if name == "README":
            continue
        source_type = "Reinhard-DELTA" if lineage["source"] == "Reinhard-DELTA" else "DELTA"
        if rows(reopened[name]) != expected_rows[(source_type, lineage["sheet"])]:
            raise ValueError(f"Saved source sheet values changed: {name}")
    check_static_workbook(reopened, set(order))
    os.replace(temporary, output)
    return source_map






def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta-workbook", type=Path, required=True)
    parser.add_argument("--reinhard-workbook", type=Path, required=True)
    parser.add_argument("--canonical-main", type=Path, required=True)
    parser.add_argument("--controls-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    for path in (args.delta_workbook, args.reinhard_workbook):
        if not path.is_file():
            raise FileNotFoundError(path)
    delta_hash = sha256(args.delta_workbook)
    reinhard_hash = sha256(args.reinhard_workbook)
    delta = load_workbook(args.delta_workbook, data_only=False)
    reinhard = load_workbook(args.reinhard_workbook, data_only=False)
    check_static_workbook(delta, {"README", *COMMON_SHEETS, *DELTA_SHEETS})
    check_static_workbook(reinhard, {"README", *COMMON_SHEETS, *REINHARD_SHEETS})
    check_shared_sheets(delta, reinhard)
    for name in DELTA_SHEETS:
        check_target_sheet(delta[name], "DELTA-v1.3")
    for name in REINHARD_SHEETS:
        check_target_sheet(reinhard[name], "Reinhard-DELTA-v1.3")
    source_audit = verify_sources(delta, args.canonical_main, args.controls_root)
    if args.check_only:
        print(json.dumps({"status": "validated", "source_hashes": [delta_hash, reinhard_hash], "backbones": len(source_audit["backbones"])}, indent=2))
        return
    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / WORKBOOK_NAME
    source_map = create_unified(delta, reinhard, output)
    manifest = {
        "schema": "tcga-brca-wsi-survival-unified-report-v1",
        "status": "complete",
        "statistical_unit": "backbone",
        "primary_p_values": "two-sided unadjusted exact paired sign-flip",
        "max_t": "supplementary multiplicity sensitivity only",
        "source_workbooks": {
            "delta": {"path": str(args.delta_workbook), "sha256": delta_hash},
            "reinhard_delta": {"path": str(args.reinhard_workbook), "sha256": reinhard_hash},
        },
        "canonical_sources": source_audit,
        "sheet_lineage": source_map,
        "output_workbook": {"path": str(output), "sha256": sha256(output)},
        "builder_sha256": sha256(Path(__file__)),
    }
    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"workbook": str(output), "sha256": sha256(output), "sheets": 20}, indent=2))


if __name__ == "__main__":
    main()
