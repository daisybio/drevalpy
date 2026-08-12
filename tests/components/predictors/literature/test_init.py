"""Tests for the literature predictor package boundary and policy.

Mirrors :mod:`drevalpy.components.predictors.literature` (the package
``__init__``), which is the module that guarantees no lazy predictor
re-exports and no engine indirection anywhere in the literature tree.

This file also holds the package-level invariants that span every literature
predictor package rather than any single one: that each is reachable as a
native ``drevalpy.models`` facade under a matching zoo name, and that each
declares exactly one input interface.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.abstract.matrix import MatrixPredictor
from drevalpy.models import construct_model
from drevalpy.models.config import ModelConfig, from_spec, validate
from drevalpy.models.drp_model import DRPModel
from drevalpy.registry._builtins import ensure_predictor_registered, register_builtin_components
from drevalpy.registry.predictor import get as get_predictor

REPO_ROOT = Path(__file__).resolve().parents[4]
PREDICTORS_ROOT = REPO_ROOT / "drevalpy" / "components" / "predictors"
LITERATURE_ROOT = PREDICTORS_ROOT / "literature"

LITERATURE_FACTORY_NAMES = [
    "DrugGNN",
    "DIPK",
    "MOLIR",
    "SuperFELTR",
    "PharmaFormer",
    "Precily",
    "SRMF",
    "SimpleNeuralNetwork",
    "MultiViewNeuralNetwork",
    "SparseGO",
]

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

LIFECYCLE_METHODS = ("_fit", "_predict", "get_state", "set_state", "is_fitted")


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


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
    missing = [name for name in LIFECYCLE_METHODS if name not in defined_methods]
    assert not missing, f"{predictor_path} missing lifecycle methods: {missing}"


def test_literature_predictor_modules_avoid_removed_adapter_modules() -> None:
    for module_name in (
        "drevalpy.components.predictors.literature.precily.predictor",
        "drevalpy.components.predictors.literature.druggnn.predictor",
        "drevalpy.components.predictors.neural_network.predictor",
        "drevalpy.components.predictors.literature.dipk.predictor",
    ):
        module = importlib.import_module(module_name)
        source_path = module.__file__
        assert source_path is not None
        text = Path(source_path).read_text(encoding="utf-8")
        assert "public_models" not in text
        assert "literature.impl" not in text
        assert "LiteratureEngineMixin" not in text


def test_literature_predictor_lazy_package_import() -> None:
    ensure_predictor_registered("dipk")
    precily_module = "drevalpy.components.predictors.literature.precily.predictor"
    saved = sys.modules.pop(precily_module, None)
    try:
        cls = get_predictor("dipk")
        assert cls.__name__ == "DIPKPredictor"
        assert precily_module not in sys.modules
    finally:
        if saved is not None:
            sys.modules[precily_module] = saved


@pytest.mark.parametrize("name", LITERATURE_FACTORY_NAMES)
def test_literature_factory_entries_are_native_facades(name: str) -> None:
    cls = construct_model(name)
    assert issubclass(cls, DRPModel)
    assert cls.__module__ == "drevalpy.models"


@pytest.mark.parametrize("name", LITERATURE_FACTORY_NAMES)
def test_model_config_and_factory_share_zoo_name(name: str) -> None:
    config = from_spec(name)
    assert isinstance(config, ModelConfig)
    model_cls = construct_model(name)
    validate(config)
    assert model_cls.get_model_name() == name


@pytest.mark.parametrize(
    ("name", "interface"),
    [
        ("drugGNN", "block"),
        ("neuralNetwork", "matrix"),
        ("precily", "block"),
        ("srmf", "block"),
        ("molir", "block"),
        ("superfeltr", "block"),
        ("pharmaFormer", "block"),
        ("dipk", "block"),
        ("sparsego", "block"),
    ],
)
def test_literature_predictor_flags(name: str, interface: str) -> None:
    cls = get_predictor(name)
    if interface == "matrix":
        assert issubclass(cls, MatrixPredictor)
        assert not issubclass(cls, BlockPredictor)
    elif interface == "block":
        assert issubclass(cls, BlockPredictor)
        assert not issubclass(cls, MatrixPredictor)
