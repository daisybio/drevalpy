"""Policy tests for predictor-owned literature packages."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
PREDICTORS_ROOT = REPO_ROOT / "drevalpy" / "components" / "predictors"
LITERATURE_ROOT = PREDICTORS_ROOT / "literature"

FORBIDDEN_TOKENS = (
    "literature.impl",
    "LiteratureEngineBase",
    "LiteratureEngineMixin",
    "ENGINE_MODULES",
    "_engine_class_name",
    "raw_engine_adapter",
    "block_engine_adapter",
    "structured_engine_adapter",
)

SINGLE_DRUG_PACKAGES = frozenset({"molir", "superfeltr"})
FEATURE_DATASET_BLOCK_PACKAGES = frozenset(
    {
        "dipk",
        "sparsego",
        "pharmaformer",
        "precily",
        "srmf",
        "druggnn",
    }
)
LITERATURE_PACKAGES = (
    "dipk",
    "sparsego",
    "molir",
    "superfeltr",
    "pharmaformer",
    "precily",
    "srmf",
    "druggnn",
)

LIFECYCLE_METHODS = ("fit", "predict", "get_state", "set_state", "is_fitted")


def _python_files_under(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def test_literature_package_has_no_lazy_predictor_reexports() -> None:
    init_path = LITERATURE_ROOT / "__init__.py"
    text = init_path.read_text(encoding="utf-8")
    assert "_LAZY_EXPORTS" not in text
    assert "__getattr__" not in text


def test_literature_tree_has_no_forbidden_engine_indirection() -> None:
    offenders: list[str] = []
    for path in _python_files_under(LITERATURE_ROOT):
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_TOKENS:
            if token in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {token}")
    assert not offenders, "\n".join(offenders)


def _defined_lifecycle_methods(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    defined_methods: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    defined_methods.add(item.name)
    return defined_methods


@pytest.mark.parametrize("package", LITERATURE_PACKAGES)
def test_literature_predictor_modules_own_lifecycle(package: str) -> None:
    predictor_path = LITERATURE_ROOT / package / "predictor.py"
    assert predictor_path.is_file()
    defined_methods = _defined_lifecycle_methods(predictor_path)
    if package in SINGLE_DRUG_PACKAGES:
        shared_path = PREDICTORS_ROOT / "single_drug_block.py"
        defined_methods |= _defined_lifecycle_methods(shared_path)
    elif package in FEATURE_DATASET_BLOCK_PACKAGES:
        shared_path = PREDICTORS_ROOT / "feature_dataset_block.py"
        defined_methods |= _defined_lifecycle_methods(shared_path)
    missing = [name for name in LIFECYCLE_METHODS if name not in defined_methods]
    assert not missing, f"{predictor_path} missing lifecycle methods: {missing}"
