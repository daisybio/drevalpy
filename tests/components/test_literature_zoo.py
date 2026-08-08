"""Tests for literature zoo recipes and validation."""

from __future__ import annotations

import pytest

from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.components.registry import get_predictor
from drevalpy.models import construct_model
from drevalpy.models.config import ModelConfig, from_spec, validate
from drevalpy.models.zoo import get_zoo_config, list_zoo_names

LITERATURE_ZOO_NAMES = [
    "DrugGNN",
    "PharmaFormer",
    "DIPK",
    "MOLIR",
    "SuperFELTR",
    "Precily",
    "SRMF",
    "SimpleNeuralNetwork",
    "MultiViewNeuralNetwork",
    "SparseGO",
]

BLOCK_ZOO_NAMES = {"DrugGNN", "PharmaFormer", "DIPK", "MOLIR", "SuperFELTR", "Precily", "SRMF", "SparseGO"}


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


@pytest.mark.parametrize("name", LITERATURE_ZOO_NAMES)
def test_literature_zoo_entries_validate(name: str) -> None:
    assert name in list_zoo_names(include_external=False)
    config = get_zoo_config(name)
    validate(config)
    assert config.cell_line_featurizer is not None
    if name in BLOCK_ZOO_NAMES:
        assert config.drug_featurizer is not None
        assert issubclass(get_predictor(config.predictor.name), BlockPredictor)


@pytest.mark.parametrize("name", LITERATURE_ZOO_NAMES)
def test_literature_zoo_entries_create_model(name: str) -> None:
    config = get_zoo_config(name)
    model = construct_model(name, config)()
    assert model is not None


@pytest.mark.parametrize("name", ["MOLIR", "SuperFELTR"])
def test_single_drug_zoo_entries_route_with_identity(name: str) -> None:
    config = get_zoo_config(name)
    assert config.drug_featurizer is not None
    assert config.drug_featurizer.name == "identity"
    validate(config)


def test_from_spec_resolves_literature_zoo() -> None:
    config = from_spec("DrugGNN")
    assert isinstance(config, ModelConfig)
    assert config.predictor.name == "drugGNN"
    assert config.cell_line_featurizer is not None
    assert config.drug_featurizer is not None
