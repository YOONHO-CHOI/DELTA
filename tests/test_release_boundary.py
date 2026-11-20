"""The evaluation distribution must not include or import the withheld core."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest


ROOT = Path(__file__).resolve().parents[1]
WITHHELD_FUNCTIONS = {
    "fit_delta", "fit_raw_reinhard", "apply_delta", "apply_delta_torch",
    "select_rank_unsupervised", "select_rank_cap_free", "select_operator",
    "patch_disjoint_pairs", "humc_disjoint_pairs", "patch_stable_priority",
    "humc_stable_priority", "patch_counterfactual_batch", "control_bases",
    "build_bank_for_split", "select_v13_operator", "project_features",
    "write_projected_h5", "materialize_fold_features",
}


def sources():
    return list((ROOT / "experiments").rglob("*.py")) + list((ROOT / "tests").glob("*.py"))


def test_withheld_implementation_and_study_payloads_are_absent():
    for name in ("delta", "software", "src", "examples", "configs"):
        assert not (ROOT / name).exists(), name
    assert {p.relative_to(ROOT / "reference").as_posix()
            for p in (ROOT / "reference").rglob("*") if p.is_file()} == {
        "backbone_registry.csv",
    }
    forbidden = {
        "experiments/pathorob/bank_coverage.py",
        "experiments/patch_aggregation/run.py",
        "experiments/bank_coverage/run.py",
        "experiments/humc_normative/run.py",
        "experiments/wsi_survival/run.py",
        "experiments/plism_analysis/run.py",
        "experiments/plism_analysis/controls.py",
        "experiments/plism_analysis/acquisition_probe.py",
        "experiments/stpbench_molecular_prediction/run.py",
        "experiments/stpbench_molecular_prediction/external.py",
        "experiments/stpbench_molecular_prediction/SOURCE_FOLD_OPERATORS.md",
        "docs/METHOD.md",
    }
    assert not any((ROOT / name).exists() for name in forbidden)


def test_public_source_has_no_hidden_core_import_or_implementation():
    for path in sources():
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.module
                       else [item.name for item in node.names] if isinstance(node, ast.Import) else [])
            assert not any(name.split(".")[0] in {"delta", "software"} for name in modules), path
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.name not in WITHHELD_FUNCTIONS, (path, node.name)
            if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
                name = node.args[0].value
                if isinstance(name, str) and name.split(".")[0] == "delta":
                    assert not (isinstance(node.func, ast.Name) and node.func.id == "__import__"), path
                    assert not (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"), path


def test_retained_internal_import_paths_exist():
    for path in sources():
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        package = ".".join(path.relative_to(ROOT).with_suffix("").parts[:-1])
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                module = importlib.util.resolve_name("." * node.level + (node.module or ""), package)
                modules = [module]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            elif isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            else:
                continue
            for module in modules:
                if module.split(".")[0] == "experiments":
                    target = ROOT.joinpath(*module.split("."))
                    assert target.with_suffix(".py").is_file() or (target / "__init__.py").is_file(), (path, module)


def test_readme_states_the_partial_release_and_restricts_humc_data():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "core DELTA implementation will be released before publication" in text
    assert "End-to-end execution currently requires the unreleased DELTA module" in text
    assert "HUMC data cannot be transferred outside the institution" in text
    assert "delta-fit" not in text and "from delta import" not in text
    assert "10-2026-0205252" in text


def test_banner_and_figure_one_remain_unchanged():
    figure = ROOT / "docs/figure1.png"
    assert hashlib.sha256(figure.read_bytes()).hexdigest() == (
        "d66dea76f5a7f6ce98d95701698cf4ef84271586a6ca85e1ce316ff35204a6f6"
    )
    svg = ET.fromstring((ROOT / "docs/banner.svg").read_text(encoding="utf-8"))
    texts = svg.findall(".//{http://www.w3.org/2000/svg}text")
    assert len(texts) == 2
    assert texts[1].text == "Training-free, label-free correction of pathology foundation model embeddings"


def test_documentation_links_and_module_commands_are_real():
    documents = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    for path in documents:
        text = path.read_text(encoding="utf-8-sig")
        for link in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
            if link.startswith(("https://", "http://", "mailto:", "#")):
                continue
            assert (path.parent / link.split("#", 1)[0]).exists(), (path, link)
        for module in re.findall(r"python\s+-m\s+(experiments(?:\.[A-Za-z_]\w*)+)", text):
            assert ROOT.joinpath(*module.split(".")).with_suffix(".py").is_file(), module


@pytest.mark.parametrize("module", [
    "experiments.common.split_cli", "experiments.pathorob.run",
    "experiments.pathorob.report", "experiments.pathorob.summary",
    "experiments.pathorob.workbook", "experiments.patch_aggregation.report",
    "experiments.patch_aggregation.summary", "experiments.wsi_survival.report",
    "experiments.wsi_survival.merge", "experiments.wsi_survival.summary",
    "experiments.bank_coverage.report", "experiments.humc_normative.report",
    "experiments.humc_normative.summary", "experiments.plism_analysis.collect",
    "experiments.plism_analysis.report", "experiments.plism_analysis.evaluate",
    "experiments.stpbench_molecular_prediction.report",
    "experiments.stpbench_molecular_prediction.statistics",
    "experiments.stpbench_molecular_prediction.external_statistics",
    "experiments.stpbench_molecular_prediction.tables",
])
def test_retained_cli_help_runs_without_the_delta_core(module):
    result = subprocess.run([sys.executable, "-B", "-m", module, "--help"],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
