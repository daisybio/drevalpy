"""Literature models are root facades backed by zoo configs."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from drevalpy.models import MODEL_FACTORY
from drevalpy.models._native_drp_model import NativeDRPModel
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
    "SparseGO",
]


@pytest.mark.parametrize("name", LITERATURE_FACTORY_NAMES)
def test_literature_factory_entries_are_native_facades(name: str) -> None:
    cls = MODEL_FACTORY[name]
    assert issubclass(cls, NativeDRPModel)
    assert cls.__module__ == "drevalpy.models"


@pytest.mark.parametrize("name", LITERATURE_FACTORY_NAMES)
def test_model_config_and_factory_share_zoo_name(name: str) -> None:
    config = ModelConfig.from_spec(name)
    model_cls = MODEL_FACTORY[name]
    config.validate()
    assert model_cls.get_model_name() == name


def test_structured_predictors_do_not_import_models_package_implementations() -> None:
    module = importlib.import_module("drevalpy.components.predictors.literature.structured_engine_adapter")
    source_path = module.__file__
    assert source_path is not None
    text = Path(source_path).read_text(encoding="utf-8")
    assert "drevalpy.models.DIPK" not in text
    assert "drevalpy.models.Precily" not in text
    assert "drevalpy.components.predictors.literature.impl" in text
    assert "set_build_context" not in text


def test_literature_predictor_modules_avoid_legacy_adapter_modules() -> None:
    for module_name in (
        "drevalpy.components.predictors.literature.structured_engine_adapter",
        "drevalpy.components.predictors.literature.precily_predictor",
        "drevalpy.components.predictors.literature.druggnn",
        "drevalpy.components.predictors.literature.neural_network",
    ):
        module = importlib.import_module(module_name)
        source_path = module.__file__
        assert source_path is not None
        text = Path(source_path).read_text(encoding="utf-8")
        assert "_legacy_structured" not in text
        assert "FeaturizerValidatedLegacyPredictor" not in text
        assert "legacy_stack" not in text
        assert "public_models" not in text
