"""Numeric PLISM summaries, paired inference and CSV/Excel exports.

The fixed encoder panel is averaged before tissue-group inference. No image,
montage, font or rendering artifact is required.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from .statistics import exact_signed_rank, holm_adjust, fixed_panel_tissue_values, tissue_bootstrap_description
from .provenance import validate_current_release_output, bind_release_output

BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 42
TEST_DECIMALS = 12
FAMILY_SIZE = 36
BACKBONES = (
    "beph", "ciga", "conch", "conchv15", "ctranspath", "exaonepath25",
    "gigapath", "gpfm", "h0mini", "hipt", "hoptimus0", "kaiko_vitb8",
    "moozy_patch", "musk", "phikonv2", "retccl", "uni", "uni2h", "virchow", "virchow2",
)
PAIR_TYPES = ("scanner_within_stain", "stain_within_scanner", "stain_and_scanner")
PAIR_LABELS = dict(zip(PAIR_TYPES, ("Scanner changes", "Stain-condition changes", "Scanner and stain-condition changes")))
METHODS = ("Original", "DELTA", "PCA", "Unpaired", "Random")
COMPARATORS = tuple(method for method in METHODS if method != "DELTA")
METHOD_LABELS = {method: "RAW" if method == "Original" else method for method in METHODS}
RAW_METHODS = {"Original embedding": "Original", "DELTA-v1.3-p2": "DELTA",
               "fitting_pca": "PCA", "unpaired": "Unpaired",
               **{f"random_{i}": "Random" for i in range(5)}}
METRICS = ("distance_reduction", "top1", "tissue")
METRIC_LABELS = {"distance_reduction": "Distance reduction",
                 "top1": "Top-1 retrieval", "tissue": "Tissue concordance"}
DISTANCE_PAIRS = dict(zip(PAIR_TYPES, (273, 546, 3276)))
RETRIEVAL_PAIRS = dict(zip(PAIR_TYPES, (78, 84, 72)))
INPUT_FIELDS = {"distance_reduction": "tissue_macro_distance_reduction",
                "top1": "tissue_macro_top1_retrieval", "tissue": "tissue_macro_tissue_concordance"}
EXPECTED_BUNDLE_SHA256 = "e33c54c3ebd9426b088d5e63986c76a2da8e8d276704f9258f1df10394e20dbf"

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def group_name(pair: str, row: dict, *, retrieval: bool) -> str:
    if pair == PAIR_TYPES[0]:
        return row["query_stain"] if retrieval else row["stain_a"]
    if pair == PAIR_TYPES[1]:
        return row["query_scanner"] if retrieval else row["scanner_a"]
    if pair == PAIR_TYPES[2]:
        return "all"
    raise ValueError(f"Unexpected pair type: {pair}")

def group_means(path: Path, *, retrieval: bool) -> dict:
    """Tissue means are in the CSV; then average groups equally within encoder."""
    metrics = METRICS[1:] if retrieval else METRICS[:1]
    sums = defaultdict(lambda: [0.0] * len(metrics) + [0])
    raw_counts, seen = defaultdict(int), set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            pair, raw_method = row["pair_type"], row["method"]
            if pair not in PAIR_TYPES or raw_method not in RAW_METHODS:
                raise ValueError(f"Unexpected pair/method in {path}")
            columns = ("query_stain", "query_scanner", "reference_stain", "reference_scanner") if retrieval else ("stain_a", "scanner_a", "stain_b", "scanner_b")
            identity = (pair, raw_method, *(row[k] for k in columns))
            if identity in seen:
                raise ValueError(f"Duplicate condition: {path}, {identity}")
            seen.add(identity)
            raw_counts[(pair, raw_method)] += 1
            key = (pair, group_name(pair, row, retrieval=retrieval), RAW_METHODS[raw_method])
            for i, metric in enumerate(metrics):
                value = float(row[INPUT_FIELDS[metric]])
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite value: {path}, {identity}")
                sums[key][i] += value
            sums[key][-1] += 1
    counts = RETRIEVAL_PAIRS if retrieval else DISTANCE_PAIRS
    for pair in PAIR_TYPES:
        for method in RAW_METHODS:
            if raw_counts[(pair, method)] != counts[pair]:
                raise ValueError(f"Incomplete panel: {path}, {pair}, {method}")
    grouped = defaultdict(list)
    for (pair, group, method), vals in sums.items():
        grouped[(pair, method)].append({m: vals[i] / vals[-1] for i, m in enumerate(metrics)})
    result = {}
    for pair, expected in zip(PAIR_TYPES, (13, 7, 1)):
        for method in METHODS:
            values = grouped[(pair, method)]
            if len(values) != expected:
                raise ValueError(f"Incomplete group panel: {path}, {pair}, {method}")
            result[(pair, method)] = {m: sum(v[m] for v in values) / expected for m in metrics}
    return result

def summarize(results_root: Path) -> dict:
    """Standard-library-only, so this can run read-only on the source server."""
    records, provenance = {}, {}
    for backbone in BACKBONES:
        folder = results_root / backbone
        marker_path = folder / "CONDITIONS_COMPLETE.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        physical, retrieval = folder / "conditions_summary.csv", folder / "condition_retrieval_summary.csv"
        hashes = {"summary_sha256": sha256(physical), "retrieval_sha256": sha256(retrieval)}
        if (marker.get("status") != "complete" or marker.get("backbone") != backbone
                or marker.get("pair_count") != 4095 or marker.get("retrieval_comparisons") != 234
                or any(marker.get(k) != v for k, v in hashes.items())):
            raise ValueError(f"Unvalidated source: {backbone}")
        distance, search = group_means(physical, retrieval=False), group_means(retrieval, retrieval=True)
        records[backbone] = {pair: {m: {**distance[(pair, m)], **search[(pair, m)]} for m in METHODS} for pair in PAIR_TYPES}
        for pair in PAIR_TYPES:
            if abs(records[backbone][pair]["Original"]["distance_reduction"]) > 1e-12:
                raise ValueError("Original reduction must be zero by definition")
        provenance[backbone] = {**hashes, "manifest_sha256": sha256(marker_path)}
    return {"schema": "plism-figure5-summary-v2", "source_root": str(results_root), "n_backbones": 20,
            "aggregation": "tissue macro, then equal groups within encoder, then equal encoder mean",
            "source_hashes": provenance, "backbones": records}

def summarize_tissue_groups(results_root: Path) -> dict:
    """Retain the 46 groups; do not treat the fixed encoders as replicates."""
    summary = summarize(results_root)
    panels, strata, provenance, tissue_names = {}, {}, {}, None
    expected_groups = {PAIR_TYPES[0]: set(("GIV", "GIVH", "GM", "GMH", "GV", "GVH", "HR", "HRH", "KR", "KRH", "LM", "LMH", "MY")),
                       PAIR_TYPES[1]: set(("S60", "AT2", "SQ", "S360", "S210", "P", "GT450")),
                       PAIR_TYPES[2]: {"all"}}
    for backbone in BACKBONES:
        folder = results_root / backbone
        path, marker_path = folder / "condition_tissue_groups.csv", folder / "TISSUE_GROUPS_COMPLETE.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (marker.get("status") != "complete" or marker.get("backbone") != backbone
                or marker.get("tissue_groups") != 46 or marker.get("rows") != 8694
                or not marker.get("original_scores_reproduced")
                or marker.get("group_summary_sha256") != sha256(path)
                or marker.get("original_manifest_sha256") != summary["source_hashes"][backbone]["manifest_sha256"]):
            raise ValueError(f"Unvalidated tissue-group source: {backbone}")
        raw, seen = {}, set()
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (row["pair_type"], row["contrast_group"], row["method"], row["tissue_type"])
                if key in seen or key[0] not in PAIR_TYPES or key[1] not in expected_groups[key[0]] or key[2] not in RAW_METHODS:
                    raise ValueError(f"Invalid/duplicate tissue-group identity: {key}")
                seen.add(key)
                raw[key] = {m: float(row[m]) for m in METRICS}
                if not all(math.isfinite(x) for x in raw[key].values()):
                    raise ValueError("Nonfinite tissue-group endpoint")
        names = sorted({key[3] for key in raw})
        if len(raw) != 8694 or len(names) != 46 or (tissue_names is not None and names != tissue_names):
            raise ValueError("Incomplete or misaligned tissue-group panel")
        tissue_names = names
        panels[backbone] = {}
        for pair in PAIR_TYPES:
            panels[backbone][pair] = {}
            strata.setdefault(pair, {})
            for method in METHODS:
                raw_names = [name for name, mapped in RAW_METHODS.items() if mapped == method]
                by_group = {}
                for group in sorted(expected_groups[pair]):
                    by_group[group] = {metric: [sum(raw[(pair, group, name, tissue)][metric] for name in raw_names) / len(raw_names)
                                              for tissue in tissue_names] for metric in METRICS}
                for group, endpoints in by_group.items():
                    target = strata[pair].setdefault(group, {}).setdefault(method, {m: [0.0] * 46 for m in METRICS})
                    for metric in METRICS:
                        for i, value in enumerate(endpoints[metric]):
                            target[metric][i] += value / len(BACKBONES)
                panels[backbone][pair][method] = {metric: [sum(v[metric][i] for v in by_group.values()) / len(by_group)
                                                         for i in range(46)] for metric in METRICS}
                for metric in METRICS:
                    actual = sum(panels[backbone][pair][method][metric]) / 46
                    if abs(actual - summary["backbones"][backbone][pair][method][metric]) > 1e-12:
                        raise ValueError(f"Original point estimate changed: {backbone}/{pair}/{method}/{metric}")
        provenance[backbone] = {"summary_sha256": sha256(path), "manifest_sha256": sha256(marker_path),
                                "analysis_source_sha256": marker["source_sha256"]}
    summary["schema"] = "plism-figure5-summary-v3-tissue-groups"
    summary["tissue_groups"] = {"labels": tissue_names, "n": 46, "backbones": panels,
                                 "strata": strata, "source_hashes": provenance}
    return summary

def analyze(summary: dict, bundle_path: Path) -> dict:
    import numpy as np

    if summary.get("n_backbones") != 20 or set(summary["backbones"]) != set(BACKBONES):
        raise ValueError("Expected all 20 encoders")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if sha256(bundle_path) != EXPECTED_BUNDLE_SHA256:
        if bundle.get("schema") != "plism-numeric-bundle-v1":
            raise ValueError("Unrecognized frozen PLISM bundle")
        validate_current_release_output(bundle)
        validate_numeric_bundle_sources(bundle, summary)
    method_map = dict(zip(METHODS, ("Original embedding", "DELTA-v1.3 p2", "Fitting-bank PCA", "Unpaired displacement", "Random subspace (five-basis mean)")))
    for b, source in summary.get("source_hashes", {}).items():
        if source["manifest_sha256"] != bundle["multistain"]["manifest_sha256"][b]:
            raise ValueError(f"Manifest differs from frozen bundle: {b}")
    for pair in PAIR_TYPES:
        for method in METHODS:
            rows = [x for x in bundle["multistain"]["table"] if x["pair_type"] == pair and x["method"] == method_map[method]]
            if len(rows) != dict(zip(PAIR_TYPES, (13, 7, 1)))[pair]:
                raise ValueError("Incomplete independent bundle check")
            for metric in METRICS:
                actual = sum(summary["backbones"][b][pair][method][metric] for b in BACKBONES) / 20
                expected = sum(x[INPUT_FIELDS[metric]] for x in rows) / len(rows)
                if abs(actual - expected) > 1e-12:
                    raise ValueError(f"Macro differs from bundle: {pair}/{method}/{metric}")
    group_source = summary.get("tissue_groups")
    if not group_source or group_source.get("n") != 46 or len(group_source["labels"]) != 46:
        raise ValueError("Recompute the 46 tissue-group summaries before statistical export")
    rng = random.Random(BOOTSTRAP_SEED)
    indices = np.asarray([[rng.randrange(46) for _ in range(46)] for _ in range(BOOTSTRAP_REPLICATES)])

    def describe(values):
        return tissue_bootstrap_description(values, indices)

    absolute, contrasts = [], []
    for pair in PAIR_TYPES:
        for metric in METRICS:
            # Average the complete fixed encoder panel first. The only
            # resampling unit is the numbered tissue group, shared by all arms.
            values = {m: fixed_panel_tissue_values([group_source["backbones"][b][pair][m][metric] for b in BACKBONES])
                      for m in METHODS}
            for method in METHODS:
                diff = [x - y for x, y in zip(values[method], values["Original"])]
                absolute.append({"pair_type": pair, "metric": metric, "method": method,
                                 "absolute": describe(values[method]), "change_from_original": describe(diff)})
            for method in METHODS:
                if method == "DELTA":
                    continue
                diff = [x - y for x, y in zip(values["DELTA"], values[method])]
                test_diff = [round(x, TEST_DECIMALS) for x in diff]
                statistic, probability, nonzero = exact_signed_rank(diff)
                contrasts.append({"pair_type": pair, "metric": metric, "reference": "DELTA", "comparator": method,
                                  "difference": describe(diff), "wilcoxon_statistic": statistic, "p_raw": probability,
                                  "n_backbones": 20, "n_tissue_groups": 46, "n_nonzero": nonzero, "delta_higher": sum(x > 0 for x in test_diff),
                                  "delta_equal": sum(x == 0 for x in test_diff), "delta_lower": sum(x < 0 for x in test_diff)})
    if len(contrasts) != FAMILY_SIZE:
        raise ValueError("Unexpected multiplicity family")
    for row, adjusted in zip(contrasts, holm_adjust([x["p_raw"] for x in contrasts])):
        row["p_holm_36"] = adjusted
        row["p_bonferroni_36"] = min(1.0, FAMILY_SIZE * row["p_raw"])
    stratified = []
    for pair, groups in group_source["strata"].items():
        for group, methods in groups.items():
            for method, endpoints in methods.items():
                stratified.append({"pair_type": pair, "group": group, "method": method,
                                   **{metric: describe(values) for metric, values in endpoints.items()}})
    return {"schema": "plism-figure5-analysis-v3-tissue-groups", "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED, "unit": "46 numbered tissue groups; fixed 20-encoder panel and fixed acquisition/reference galleries",
            "n_tissue_groups": 46, "n_backbones_fixed": 20,
            "bootstrap_index_sha256": hashlib.sha256(indices.astype("<i2").tobytes()).hexdigest(),
            "test": "two-sided conditional exact Wilcoxon signed-rank, zero_method=wilcox",
            "test_difference_decimals": TEST_DECIMALS,
            "multiplicity": "Holm across all 36 DELTA contrasts; Bonferroni-36 sensitivity also supplied",
            "confidence_intervals": "pointwise paired tissue-group percentile bootstrap, not simultaneous",
            "inference_scope": "Conditional on the observed tissue/acquisition/encoder panel; no verified patient or donor mapping. Bootstrap assumes exchangeable tissue groups and signed-rank tests assume independent sign-symmetric group differences; public metadata do not establish group independence.",
            "bundle_sha256": sha256(bundle_path), "source_summary": summary,
            "absolute": absolute, "contrasts": contrasts, "stratified": stratified}

def write_tables(analysis, output_dir):
    # Export a new workbook with the table writer distributed in this script.
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, Side
    from openpyxl.utils import get_column_letter
    wb = Workbook()
    wb.remove(wb.active)
    thin = Side(style="thin", color="D5DADF")

    def sheet(name, title, headers, records, widths):
        ws = wb.create_sheet(name)
        ws.sheet_view.showGridLines = True
        ws.append([title])
        ws.append([])
        ws.append(headers)
        for row in records:
            ws.append(row)
        for row in ws:
            for cell in row:
                cell.font = Font(name="Arial", size=11)
                cell.alignment = Alignment(vertical="center", horizontal="left" if isinstance(cell.value, str) else "right")
                if isinstance(cell.value, (float, int)):
                    heading = str(ws.cell(3, cell.column).value)
                    cell.number_format = "0.000000" if " P" in heading or heading.startswith("P ") else "0.00"
                    if heading in ("DELTA higher", "Equal", "DELTA lower"):
                        cell.number_format = "0"
        ws["A1"].font = Font(name="Arial", size=12, bold=True)
        for cell in ws[3]:
            cell.font = Font(name="Arial", size=11, bold=True)
            cell.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")
            cell.border = Border(top=thin, bottom=thin)
        ws.row_dimensions[3].height = 43
        for i, width in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(i)].width = width
        for i in range(4, ws.max_row + 1):
            ws.row_dimensions[i].height = 20
        ws.freeze_panes = "C4"
        ws.auto_filter.ref = f"A3:{get_column_letter(len(headers))}{ws.max_row}"
        return ws

    scores = []
    lookup = {(r["pair_type"], r["metric"], r["method"]): r for r in analysis["absolute"]}
    tests_lookup = {(r["pair_type"], r["metric"], r["comparator"]): r for r in analysis["contrasts"]}
    for pair in PAIR_TYPES:
        for method in METHODS:
            record = [PAIR_LABELS[pair], METHOD_LABELS[method]]
            for metric in METRICS:
                a = lookup[(pair, metric, method)]["absolute"]
                probability = None if method == "DELTA" else tests_lookup[(pair, metric, method)]["p_holm_36"]
                record.extend([a["mean"] * 100, *[x * 100 for x in a["ci95"]], probability])
            scores.append(record)
    score_headers = ["Acquisition contrast", "Method"]
    for endpoint in ("Distance reduction", "Region top-1", "Tissue concordance"):
        score_headers.extend([f"{endpoint}\nMean (%)", f"{endpoint}\n95% CI lower", f"{endpoint}\n95% CI upper", f"{endpoint}\nHolm P vs DELTA"])
    overview = sheet("Scores", "PLISM Figure 5: absolute scores and DELTA comparisons", score_headers, scores,
                     [37, 15] + [20, 20, 20, 22] * 3)
    for number, values in enumerate(scores, 4):
        if values[1] == "DELTA":
            for cell in overview[number]:
                cell.font = Font(name="Arial", size=11, bold=True)
        if number in (9, 14):
            for cell in overview[number]:
                cell.border = Border(top=thin)
    tests = []
    for row in analysis["contrasts"]:
        d = row["difference"]
        tests.append([PAIR_LABELS[row["pair_type"]], METRIC_LABELS[row["metric"]], METHOD_LABELS[row["comparator"]],
                      d["mean"] * 100, *[x * 100 for x in d["ci95"]], row["p_raw"], row["p_holm_36"], row["p_bonferroni_36"],
                      row["delta_higher"], row["delta_equal"], row["delta_lower"]])
    test_headers = ["Acquisition contrast", "Endpoint", "Comparator", "Difference (pp)", "95% CI lower", "95% CI upper", "P raw", "P Holm (36)", "P Bonferroni (36)", "DELTA higher", "Equal", "DELTA lower"]
    sheet("DELTA comparisons", "Paired differences: DELTA minus comparator, 46 tissue groups; 20 encoders fixed", test_headers, tests,
          [37, 31, 15, 18, 16, 16, 16, 17, 21, 16, 12, 16])
    source = []
    for b in BACKBONES:
        for pair in PAIR_TYPES:
            for m in METHODS:
                v = analysis["source_summary"]["backbones"][b][pair][m]
                source.append([b, PAIR_LABELS[pair], METHOD_LABELS[m], *[v[k] * 100 for k in METRICS]])
    sheet("Encoder scores", "Tissue-balanced scores for each frozen encoder and correction method",
          ["Encoder", "Acquisition contrast", "Method", "Distance reduction (%)", "Region top-1 (%)", "Tissue concordance (%)"],
          source, [20, 37, 15, 25, 23, 27])
    # No tab colors or filled backgrounds; default worksheet gridlines remain.
    wb.save(output_dir / "PLISM_Figure_5_statistics.xlsx")
    for name, headers, records in (("PLISM_Figure_5_scores.csv", score_headers, scores),
                                   ("PLISM_Figure_5_paired_statistics.csv", test_headers, tests)):
        with (output_dir / name).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerows(records)

def probe_table_rows(source):
    """Validate saved probe estimates; negate the stored paired CI correctly."""
    if (source.get('schema') != 'plism-acquisition-probe-summary-v1'
            or source.get('status') != 'complete' or source.get('n_backbones_fixed') != 20
            or source.get('n_tissue_groups') != 46 or tuple(source['backbones']) != BACKBONES
            or source['methods'] != ['RAW','DELTA','PCA','Unpaired','Random']
            or source['tasks'] != ['scanner','stain_condition']):
        raise ValueError('Expected the completed fixed-20-FM/46-group acquisition analysis')
    absolutes={(r['task'],r['method']):r for r in source['absolute']}
    contrasts={(r['task'],r['comparator']):r for r in source['contrasts']}
    if len(absolutes)!=10 or len(contrasts)!=8 or len(source['contrasts'])!=8:
        raise ValueError('Incomplete probe comparison panel')
    rows=[]
    for ti,task in enumerate(source['tasks']):
        group_values=source['tissue_group_values'][ti]
        if len(group_values)!=5 or any(len(v)!=46 for v in group_values):
            raise ValueError('Incomplete paired tissue-group values')
        for mi,method in enumerate(source['methods']):
            vals=group_values[mi]
            if any(not math.isfinite(v) or v<0 or v>1 for v in vals):
                raise ValueError('Invalid balanced accuracy')
            if abs(math.fsum(vals)/46-absolutes[(task,method)]['mean'])>1e-12:
                raise ValueError('Absolute mean does not reproduce saved group scores')
            if method=='DELTA':continue
            result=contrasts[(task,method)]
            if result['reference']!='DELTA' or result['difference_direction']!='Comparator minus DELTA acquisition balanced accuracy':
                raise ValueError('Unexpected stored probe contrast direction')
            diff=result['difference'];lo,hi=diff['ci95']
            expected=math.fsum(v-d for v,d in zip(vals,group_values[1]))/46
            if (not all(math.isfinite(v) for v in (diff['mean'],lo,hi,result['p_holm_8']))
                    or abs(diff['mean']-expected)>1e-12 or not lo<=diff['mean']<=hi
                    or not 0<=result['p_holm_8']<=1):
                raise ValueError('Invalid paired probe estimate or interval')
            rows.append(dict(task=task,method=method,
                             absolute_accuracy_percent=100*absolutes[(task,method)]['mean'],
                             delta_accuracy_percent=100*absolutes[(task,'DELTA')]['mean'],
                             difference_pp=-100*diff['mean'],
                             ci95_pp=[-100*hi,-100*lo],p_holm_8=result['p_holm_8'],
                             p_bonferroni_8=result['p_bonferroni_8']))
    return rows


def validate_numeric_bundle_sources(bundle, summary):
    """Match the complete recorded input identity, not only a release marker."""
    receipts = bundle.get("source_receipts", {})
    if set(receipts) != set(BACKBONES):
        raise ValueError("Incomplete collected PLISM source receipts")
    for backbone in BACKBONES:
        source = summary["source_hashes"][backbone]
        receipt = receipts[backbone]
        if receipt.get("manifest_sha256") != source["manifest_sha256"]:
            raise ValueError("Collected condition manifest differs")
        for key in ("summary_sha256", "retrieval_sha256"):
            if receipt.get(key) != source[key]:
                raise ValueError("Collected numerical input differs")
        identity = collect_marker_identity(receipt)
        if identity != summary.get("input_identities", {}).get(backbone):
            raise ValueError("Collected feature/operator/index/source identity differs")
        for key, count in (("feature_sha256", 91), ("index_sha256", 13), ("operator_sha256", 8)):
            hashes = receipt.get(key, {})
            if len(hashes) != count or any(not isinstance(h, str) or len(h) != 64
                                           or any(c not in "0123456789abcdef" for c in h) for h in hashes.values()):
                raise ValueError("Incomplete collected feature/operator/index identity")
        if not isinstance(receipt.get("source_sha256"), str) or len(receipt["source_sha256"]) != 64:
            raise ValueError("Missing source implementation identity")


def collect_marker_identity(marker):
    result = {}
    for key, count in (("feature_sha256", 91), ("index_sha256", 13), ("operator_sha256", 8)):
        values = marker.get(key, {})
        if len(values) != count or any(not isinstance(value, str) or len(value) != 64
                                       or any(char not in "0123456789abcdef" for char in value)
                                       for value in values.values()):
            raise ValueError("Incomplete source feature/operator/index hashes")
        result[key] = values
    value = marker.get("source_sha256")
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("Missing source implementation hash")
    result["source_sha256"] = value
    return result


def attach_input_identities(summary, results_root):
    identities = {}
    for backbone in BACKBONES:
        path = results_root / backbone / "CONDITIONS_COMPLETE.json"
        if sha256(path) != summary["source_hashes"][backbone]["manifest_sha256"]:
            raise ValueError("Condition receipt changed during collection")
        identities[backbone] = collect_marker_identity(json.loads(path.read_text(encoding="utf-8")))
    return {**summary, "input_identities": identities}


def export_probe_tables(source, output_dir):
    rows = probe_table_rows(source)
    import pandas as pd
    frame = pd.DataFrame([{**row, "ci95_low_pp": row["ci95_pp"][0],
                            "ci95_high_pp": row["ci95_pp"][1]} for row in rows]).drop(columns="ci95_pp")
    frame.to_csv(output_dir / "PLISM_acquisition_probe_statistics.csv", index=False)
    with pd.ExcelWriter(output_dir / "PLISM_acquisition_probe_statistics.xlsx", engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Paired contrasts", index=False)
        pd.DataFrame(source["absolute"]).to_excel(writer, sheet_name="Absolute scores", index=False)
    return rows


def _new_directory(path):
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("Choose a new or empty output directory")
    path.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--results-root", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    tables = sub.add_parser("tables")
    tables.add_argument("--summary", type=Path, required=True)
    tables.add_argument("--bundle", type=Path, required=True)
    tables.add_argument("--output-dir", type=Path, required=True)
    probe = sub.add_parser("probe-tables")
    probe.add_argument("--summary", type=Path, required=True)
    probe.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "summarize":
        if args.output.exists():
            raise FileExistsError(args.output)
        value = attach_input_identities(summarize_tissue_groups(args.results_root), args.results_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temp = args.output.with_name(args.output.name + ".partial")
        temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temp, args.output)
        print(args.output)
        return
    supplied = json.loads(args.summary.read_text(encoding="utf-8"))
    if args.command == "probe-tables":
        probe_table_rows(supplied)
        _new_directory(args.output_dir)
        export_probe_tables(supplied, args.output_dir)
    else:
        analysis = analyze(supplied.get("source_summary", supplied), args.bundle)
        _new_directory(args.output_dir)
        write_tables(analysis, args.output_dir)
        analysis = bind_release_output({**analysis, "report_sha256": sha256(Path(__file__))})
        (args.output_dir / "PLISM_statistics.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("PLISM_NUMERIC_REPORT_COMPLETE")


if __name__ == "__main__":
    main()


