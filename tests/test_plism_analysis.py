from __future__ import annotations
import ast
import csv
import json
from pathlib import Path
import numpy as np
import pytest
from experiments.plism_analysis import report, confusion
from experiments.plism_analysis.evaluate import evaluate_corresponding_features
from experiments.plism_analysis import provenance

def test_figure5_uses_fixed_encoder_panel_and_46_tissue_groups() -> None:
    figure = report
    panel = np.arange(20)[:, None] + np.arange(46)[None, :]
    values = figure.fixed_panel_tissue_values(panel)
    assert len(values) == 46
    np.testing.assert_allclose(values, 9.5 + np.arange(46))
    with np.testing.assert_raises(ValueError):
        figure.fixed_panel_tissue_values(np.ones((46, 20)))

def test_figure5_exact_signed_rank_ties_and_zeros() -> None:
    import itertools
    figure = report
    # Tied average ranks (doubled), excluding one exact zero.
    differences = [1.0, -1.0, 2.0, -3.0, 0.0]
    ranks = [3, 3, 6, 8]
    observed = min(9, sum(ranks) - 9)
    totals = [sum(r for r, s in zip(ranks, signs) if s) for signs in itertools.product([0, 1], repeat=4)]
    expected = sum(min(v, sum(ranks) - v) <= observed for v in totals) / len(totals)
    statistic, p, n = figure.exact_signed_rank(differences)
    assert statistic == observed / 2
    assert p == expected
    assert n == 4
    assert figure.exact_signed_rank([0.0] * 46) == (0.0, 1.0, 0)
    assert figure.exact_signed_rank([1.0] * 46)[1] == 2 ** -45
    assert figure.exact_signed_rank([1e-16, -1e-16] * 23) == (0.0, 1.0, 0)
    assert figure.holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]

def test_figure5_bootstrap_preserves_pairing() -> None:
    figure = report
    indices = np.asarray([np.full(46, 0), np.full(46, 45), np.arange(46)])
    original = np.arange(46, dtype=float)
    corrected = original + 0.25
    paired = figure.tissue_bootstrap_description(corrected - original, indices)
    assert paired == {'mean': 0.25, 'ci95': [0.25, 0.25]}
    result = figure.tissue_bootstrap_description(original, indices)
    np.testing.assert_allclose(result['ci95'], np.quantile([0, 45, 22.5], [.025, .975]))

def acquisition_probe_figure_fixture(figure):
    methods=['RAW','DELTA','PCA','Unpaired','Random'];tasks=['scanner','stain_condition']
    source=dict(schema='plism-acquisition-probe-summary-v1',status='complete',
                n_backbones_fixed=20,n_tissue_groups=46,backbones=list(figure.BACKBONES),
                methods=methods,tasks=tasks,absolute=[],contrasts=[],tissue_group_values=[])
    for ti,task in enumerate(tasks):
        means=[.8-ti*.1,.79-ti*.1,.797-ti*.1,.795-ti*.1,.799-ti*.1]
        source['tissue_group_values'].append([[m]*46 for m in means])
        for method,mean in zip(methods,means):
            source['absolute'].append(dict(task=task,method=method,mean=mean,ci95=[mean-.02,mean+.02]))
            if method=='DELTA':continue
            effect=mean-means[1]
            source['contrasts'].append(dict(task=task,reference='DELTA',comparator=method,
                difference_direction='Comparator minus DELTA acquisition balanced accuracy',
                difference=dict(mean=effect,ci95=[effect-.001,effect+.001]),
                p_holm_8=.001,p_bonferroni_8=.002))
    return source

def test_plism_probe_figure_reverses_paired_ci_and_preserves_absolute_scores() -> None:
    figure=report
    rows=figure.probe_table_rows(acquisition_probe_figure_fixture(figure))
    assert len(rows)==8
    row=rows[0]
    assert (row['task'],row['method'])==('scanner','RAW')
    np.testing.assert_allclose(row['difference_pp'],-1.)
    np.testing.assert_allclose(row['ci95_pp'],[-1.1,-.9])
    np.testing.assert_allclose(row['absolute_accuracy_percent'],80.)
    np.testing.assert_allclose(row['delta_accuracy_percent'],79.)
    assert row['p_holm_8']==.001

def test_plism_probe_figure_rejects_mismatched_direction_and_unpaired_estimates() -> None:
    figure=report
    source=acquisition_probe_figure_fixture(figure)
    source['contrasts'][0]['difference_direction']='DELTA minus comparator'
    with np.testing.assert_raises(ValueError):figure.probe_table_rows(source)
    source=acquisition_probe_figure_fixture(figure)
    source['contrasts'][0]['difference']['mean']+=.001
    with np.testing.assert_raises(ValueError):figure.probe_table_rows(source)
    source=acquisition_probe_figure_fixture(figure)
    source['absolute'][0]['mean']+=.001
    with np.testing.assert_raises(ValueError):figure.probe_table_rows(source)

def test_confusion_numeric_export_preserves_all_directed_cells(tmp_path):
    matrices = np.asarray([[[[[.8,.2],[.1,.9]]],[[[.9,.1],[.15,.85]]]]])
    source = tmp_path / "confusion.npz"
    np.savez(source, matrices=matrices, tissue_labels=np.asarray(["a","b"]),
             backbones=np.asarray(["toy"]), pair_types=np.asarray(["scanner"]))
    output = tmp_path / "confusion.csv"
    confusion.export_csv(source, output)
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    np.testing.assert_array_equal([float(row["probability"]) for row in rows], matrices.ravel())
    assert rows[1]["query_tissue"] == "a" and rows[1]["neighbor_tissue"] == "b"
    assert rows[2]["query_tissue"] == "b" and rows[2]["neighbor_tissue"] == "a"
    with pytest.raises(FileExistsError):
        confusion.export_csv(source, output)

def synthetic_summary():
    method_map = dict(zip(report.METHODS, ("Original embedding", "DELTA-v1.3 p2", "Fitting-bank PCA", "Unpaired displacement", "Random subspace (five-basis mean)")))
    base = {"Original":.10, "DELTA":.14, "PCA":.11, "Unpaired":.12, "Random":.105}
    backbones = {b:{pair:{m:{metric:base[m] for metric in report.METRICS} for m in report.METHODS}
                    for pair in report.PAIR_TYPES} for b in report.BACKBONES}
    panel = {b:{pair:{m:{metric:[base[m]]*46 for metric in report.METRICS} for m in report.METHODS}
                for pair in report.PAIR_TYPES} for b in report.BACKBONES}
    strata = {pair:{"fixture":{m:{metric:[base[m]]*46 for metric in report.METRICS} for m in report.METHODS}}
              for pair in report.PAIR_TYPES}
    summary = {"n_backbones":20,"backbones":backbones,"source_hashes":{},
               "tissue_groups":{"n":46,"labels":[str(i) for i in range(46)],"backbones":panel,"strata":strata}}
    table=[]
    for pair,count in zip(report.PAIR_TYPES,(13,7,1)):
        for m in report.METHODS:
            for group in range(count):
                table.append({"pair_type":pair,"method":method_map[m],"group":group,
                              **{report.INPUT_FIELDS[metric]:base[m] for metric in report.METRICS}})
    return summary,{"multistain":{"table":table,"manifest_sha256":{}}}

def test_numeric_analysis_and_csv_excel_exports(tmp_path, monkeypatch):
    summary,bundle=synthetic_summary()
    path=tmp_path/"frozen.json"
    path.write_text(json.dumps(bundle))
    monkeypatch.setattr(report,"EXPECTED_BUNDLE_SHA256",report.sha256(path))
    monkeypatch.setattr(report,"BOOTSTRAP_REPLICATES",128)
    analysis=report.analyze(summary,path)
    assert len(analysis["absolute"])==45 and len(analysis["contrasts"])==36
    raw=next(row for row in analysis["contrasts"] if row["comparator"]=="Original")
    assert raw["difference"]["mean"]==pytest.approx(.04,abs=1e-12)
    np.testing.assert_allclose(raw["difference"]["ci95"],[.04,.04],rtol=0,atol=1e-12)
    assert raw["p_raw"]==2**-45
    assert raw["p_holm_36"]==36*2**-45
    load_workbook=pytest.importorskip("openpyxl").load_workbook
    output=tmp_path/"tables";output.mkdir()
    report.write_tables(analysis,output)
    with (output/"PLISM_Figure_5_paired_statistics.csv").open(encoding="utf-8-sig",newline="") as stream:
        rows=list(csv.reader(stream))
    assert len(rows)==37 and float(rows[1][3])==pytest.approx(4.)
    book=load_workbook(output/"PLISM_Figure_5_statistics.xlsx",data_only=True)
    assert book.sheetnames==["Scores","DELTA comparisons","Encoder scores"]
    assert book["DELTA comparisons"].max_row==39
    assert book["Encoder scores"].max_row==303
    assert book["DELTA comparisons"].cell(4,4).value==pytest.approx(4.)
    book.close()
    assert not any(p.suffix.lower() in {".png",".pdf",".svg"} for p in output.iterdir())

def marker_identity():
    return {"feature_sha256":{str(i):"a"*64 for i in range(91)},
            "index_sha256":{str(i):"b"*64 for i in range(13)},
            "operator_sha256":{str(i):"c"*64 for i in range(8)},"source_sha256":"d"*64}

def test_new_numeric_bundle_requires_complete_matching_source_identity():
    identity=marker_identity()
    receipts={b:{**identity,"manifest_sha256":"e"*64,"summary_sha256":"f"*64,"retrieval_sha256":"0"*64} for b in report.BACKBONES}
    summary={"source_hashes":{b:{k:receipts[b][k] for k in ("manifest_sha256","summary_sha256","retrieval_sha256")} for b in report.BACKBONES},
             "input_identities":{b:identity for b in report.BACKBONES}}
    bundle={"source_receipts":receipts}
    report.validate_numeric_bundle_sources(bundle,summary)
    changed=json.loads(json.dumps(bundle))
    changed["source_receipts"][report.BACKBONES[0]]["operator_sha256"]["0"]="1"*64
    with pytest.raises(ValueError,match="identity differs"):
        report.validate_numeric_bundle_sources(changed,summary)
    incomplete=marker_identity();incomplete["feature_sha256"].pop("0")
    with pytest.raises(ValueError,match="Incomplete source"):
        report.collect_marker_identity(incomplete)

def test_report_modules_have_no_drawing_imports_or_render_functions():
    for module in (report,confusion):
        source=Path(module.__file__).read_text()
        tree=ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                assert not any(alias.name.split('.')[0] in {"matplotlib","reportlab","PIL","pypdf"} for alias in node.names)
            if isinstance(node,ast.ImportFrom):
                assert not node.module or node.module.split('.')[0] not in {"matplotlib","reportlab","PIL","pypdf"}
        assert not any(isinstance(node,ast.FunctionDef) and (node.name.startswith("render") or node.name=="draw") for node in tree.body)


def test_precomputed_acquisition_features_preserve_distance_and_retrieval_endpoints():
    reference = np.asarray([[0., 0.], [2., 0.], [0., 2.]], dtype=np.float32)
    raw_query = reference + [1., 0.]
    adjusted_query = reference + [.5, 0.]
    result = evaluate_corresponding_features(reference, raw_query, reference, adjusted_query, ["a", "a", "b"])
    np.testing.assert_allclose(result.removed_fraction, .75)
    assert result.top1.all() and result.within_tissue_top1.all()
    np.testing.assert_array_equal(result.nonidentical_field_tissue_consistency, [True, True, False])
    zero = evaluate_corresponding_features(reference, reference, reference, reference, ["a", "a", "b"])
    assert zero.removed_fraction.isna().all()
    with pytest.raises(ValueError, match="corresponding"):
        evaluate_corresponding_features(reference, raw_query[:2], reference, adjusted_query, ["a", "a", "b"])


def test_plism_public_sources_do_not_import_private_operator_code():
    for path in Path(report.__file__).parent.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("delta")
            if isinstance(node, ast.FunctionDef):
                assert node.name not in {"control_bases", "select_rank_cap_free", "apply_delta"}


def test_public_manifest_validates_evaluation_bytes_and_rejects_private_scope(tmp_path):
    source = Path(report.__file__).resolve()
    manifest = tmp_path / "manifest.json"
    payload = {"hash_policy": "sha256-lf-canonical-bytes-v1", "files": {
        "experiments/plism_analysis/report.py": provenance.canonical_sha256(source)}}
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert provenance.verify_release_manifest(manifest) == payload
    payload["files"] = {"delta/rank.py": "0" * 64}
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source scope"):
        provenance.verify_release_manifest(manifest)


def test_external_historical_protocol_hashes_fail_closed(tmp_path):
    protocol = tmp_path / "current.json"
    protocol.write_text("{}", encoding="utf-8")
    assert provenance.validate_source_protocol({"protocol_sha256": "a" * 64}, protocol, ["a" * 64]) is False
    with pytest.raises(ValueError, match="unrecognized historical"):
        provenance.validate_source_protocol({"protocol_sha256": "b" * 64}, protocol, ["a" * 64])


def test_manifest_sources_ignore_external_resource_root(tmp_path, monkeypatch):
    external = tmp_path / "external-configurations"
    duplicate = external / "experiments" / "plism_analysis" / "report.py"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_text("# external resources cannot replace executed evaluation code\n", encoding="utf-8")
    monkeypatch.setenv("DELTA_REPOSITORY_ROOT", str(external))
    relative = "experiments/plism_analysis/report.py"
    payload = {"hash_policy": "sha256-lf-canonical-bytes-v1", "files": {
        relative: provenance.canonical_sha256(Path(report.__file__))}}
    manifest = tmp_path / "public-manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert provenance.verify_release_manifest(manifest) == payload
    payload["files"][relative] = provenance.canonical_sha256(duplicate)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="evaluation code changed"):
        provenance.verify_release_manifest(manifest)

