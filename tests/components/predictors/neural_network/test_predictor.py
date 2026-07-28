"""Smoke tests for the neural_network predictor package."""

from __future__ import annotations

import numpy as np

import drevalpy.components.register_builtins as register_builtins
from drevalpy.components.predictors.neural_network.predictor import NeuralNetworkPredictor
from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.components.registry import get_predictor
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models import construct_model
from drevalpy.models.config import ModelConfig


def test_neural_network_predictor_registry_name() -> None:
    register_builtins.ensure_predictor_registered("neuralNetwork")
    assert get_predictor("neuralNetwork") is NeuralNetworkPredictor


def _synthetic_data() -> tuple[DrugResponseDataset, FeatureDataset, FeatureDataset]:
    response = DrugResponseDataset(
        response=np.array([1.0, 2.0, 3.0, 4.0]),
        cell_line_ids=np.array(["cl1", "cl1", "cl2", "cl2"]),
        drug_ids=np.array(["d1", "d2", "d1", "d2"]),
    )
    cell_line_input = FeatureDataset(
        features={
            "cl1": {"gene_expression": np.array([0.1, 0.2, 0.3, 0.4])},
            "cl2": {"gene_expression": np.array([0.5, 0.6, 0.7, 0.8])},
        }
    )
    drug_input = FeatureDataset(
        features={
            "d1": {"fingerprints": np.array([1.0, 0.0, 0.5, 0.2])},
            "d2": {"fingerprints": np.array([0.0, 1.0, 0.3, 0.7])},
        }
    )
    return response, cell_line_input, drug_input


def test_neural_network_zoo_trains_on_synthetic_data() -> None:
    register_builtin_components()
    response, cell_line_input, drug_input = _synthetic_data()
    config = ModelConfig.from_spec(
        "SimpleNeuralNetwork",
        hyperparameters={"max_epochs": 1, "batch_size": 2},
    )
    model = construct_model("SimpleNeuralNetwork", config)()
    model.train(response, cell_line_input, drug_input)
    preds = model.predict(
        response.cell_line_ids,
        response.drug_ids,
        cell_line_input,
        drug_input,
    )
    assert preds.shape == (4,)
    assert np.isfinite(preds).all()


def test_neural_network_configured_is_not_fitted_before_training() -> None:
    register_builtin_components()
    predictor = NeuralNetworkPredictor(
        hyperparameters={"max_epochs": 1, "units_per_layer": [4, 2]},
    )
    assert predictor._model is None
    assert predictor.is_fitted() is False
