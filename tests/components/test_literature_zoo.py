"""Tests for literature zoo recipes and validation."""

from __future__ import annotations

import pytest

from drevalpy.components.config import ModelConfig
from drevalpy.components.register_builtins import register_builtin_components
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
]


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components(include_legacy=False)


@pytest.mark.parametrize("name", LITERATURE_ZOO_NAMES)
def test_literature_zoo_entries_require_featurizers(name: str) -> None:
    assert name in list_zoo_names(include_external=False)
    config = get_zoo_config(name)
    assert config.cell_line_featurizer is not None
    config.validate()


def test_druggnn_rejects_fingerprint_drug_featurizer() -> None:
    from drevalpy.components.config import FeaturizerConfig, PredictorConfig

    config = ModelConfig(
        cell_line_featurizer=FeaturizerConfig(name="landmarkGenesReduced", registry="cell_line"),
        drug_featurizer=FeaturizerConfig(name="fingerprints", registry="drug"),
        predictor=PredictorConfig(type="drugGNN"),
    )
    with pytest.raises(ValueError, match="incompatible"):
        config.validate()


def test_molir_allows_missing_drug_featurizer() -> None:
    config = get_zoo_config("MOLIR")
    assert config.drug_featurizer is None
    config.validate()


@pytest.mark.parametrize("name", LITERATURE_ZOO_NAMES)
def test_literature_zoo_entries_create_model(name: str) -> None:
    config = get_zoo_config(name)
    model = config.create_model()
    assert model is not None
    assert config.cell_line_featurizer is not None


def test_superfeltr_allows_missing_drug_featurizer() -> None:
    config = get_zoo_config("SuperFELTR")
    assert config.drug_featurizer is None
    config.validate()


def test_simple_and_multiview_neural_network_share_predictor() -> None:
    simple = get_zoo_config("SimpleNeuralNetwork")
    multi = get_zoo_config("MultiViewNeuralNetwork")
    assert simple.predictor.type == "neuralNetwork"
    assert multi.predictor.type == "neuralNetwork"
    assert simple.cell_line_featurizer is not None
    assert multi.cell_line_featurizer is not None
    assert simple.cell_line_featurizer.name == "scaledGeneExpression"
    assert multi.cell_line_featurizer.name == "concatFeaturizers"


def test_molir_requires_cell_line_featurizer() -> None:
    from drevalpy.components.config import PredictorConfig

    config = ModelConfig(
        cell_line_featurizer=None,
        drug_featurizer=None,
        predictor=PredictorConfig(type="molir"),
    )
    with pytest.raises(ValueError, match="cell_line_featurizer"):
        config.validate()


def test_pharmaformer_rejects_graph_drug_featurizer() -> None:
    from drevalpy.components.config import FeaturizerConfig, PredictorConfig

    config = ModelConfig(
        cell_line_featurizer=FeaturizerConfig(name="landmarkGenes", registry="cell_line"),
        drug_featurizer=FeaturizerConfig(name="drugGraph", registry="drug"),
        predictor=PredictorConfig(type="pharmaFormer"),
    )
    with pytest.raises(ValueError, match="incompatible"):
        config.validate()


def test_model_config_from_spec_resolves_literature_zoo() -> None:
    config = ModelConfig.from_spec("DrugGNN")
    assert config.predictor.type == "drugGNN"
    assert config.cell_line_featurizer is not None
    assert config.drug_featurizer is not None
