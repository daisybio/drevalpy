"""Tests for literature refactor: public factory and component parity."""

from __future__ import annotations

import importlib

import pytest

from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.models import MODEL_FACTORY
from drevalpy.models.config import ModelConfig

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
]


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


@pytest.mark.parametrize("name", LITERATURE_FACTORY_NAMES)
def test_model_factory_imports_component_public_models(name: str) -> None:
    cls = MODEL_FACTORY[name]
    module = cls.__module__
    assert module.startswith("drevalpy.components.predictors.literature")


@pytest.mark.parametrize("name", LITERATURE_FACTORY_NAMES)
def test_model_config_and_factory_share_zoo_name(name: str) -> None:
    config = ModelConfig.from_spec(name)
    model_cls = MODEL_FACTORY[name]
    config.validate()
    assert model_cls.get_model_name() == name


def test_legacy_model_import_paths_still_resolve() -> None:
    dipk = importlib.import_module("drevalpy.models.DIPK.dipk")
    assert dipk.DIPKModel.__module__.startswith("drevalpy.components.predictors.literature.impl")


def test_structured_predictors_do_not_import_models_package_implementations() -> None:
    module = importlib.import_module("drevalpy.components.predictors.literature.structured_predictors")
    source_path = module.__file__
    assert source_path is not None
    text = open(source_path, encoding="utf-8").read()
    assert "drevalpy.models.DIPK" not in text
    assert "drevalpy.models.Precily" not in text
    assert "drevalpy.components.predictors.literature.impl" in text


def test_literature_predictor_modules_avoid_legacy_adapter_modules() -> None:
    for module_name in (
        "drevalpy.components.predictors.literature.structured_predictors",
        "drevalpy.components.predictors.literature.druggnn",
        "drevalpy.components.predictors.literature.neural_network",
    ):
        module = importlib.import_module(module_name)
        source_path = module.__file__
        assert source_path is not None
        text = open(source_path, encoding="utf-8").read()
        assert "_legacy_structured" not in text
        assert "FeaturizerValidatedLegacyPredictor" not in text
        assert "legacy_stack" not in text
