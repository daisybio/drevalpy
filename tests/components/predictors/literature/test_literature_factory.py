"""Literature models are root facades backed by zoo configs."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from drevalpy.models import construct_model
from drevalpy.models.config import ModelConfig
from drevalpy.models.drp_model import DRPModel

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
    cls = construct_model(name)
    assert issubclass(cls, DRPModel)
    assert cls.__module__ == "drevalpy.models"


@pytest.mark.parametrize("name", LITERATURE_FACTORY_NAMES)
def test_model_config_and_factory_share_zoo_name(name: str) -> None:
    config = ModelConfig.from_spec(name)
    model_cls = construct_model(name)
    config.validate()
    assert model_cls.get_model_name() == name


def test_literature_predictor_modules_avoid_legacy_adapter_modules() -> None:
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
        assert "_legacy_structured" not in text
        assert "FeaturizerValidatedLegacyPredictor" not in text
        assert "legacy_stack" not in text
        assert "public_models" not in text
        assert "literature.impl" not in text
        assert "LiteratureEngineMixin" not in text
