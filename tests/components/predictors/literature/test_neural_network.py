"""Smoke tests for modular neural network predictor."""

from __future__ import annotations

import numpy as np

from drevalpy.components.predictors.literature.neural_network import NeuralNetworkPredictor
from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.config import ModelConfig


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
    model = config.create_model()
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
