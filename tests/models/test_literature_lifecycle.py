"""Smoke tests for literature models routed through the component bridge."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models import MODEL_FACTORY


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


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


@pytest.mark.parametrize(
    ("model_name", "hyperparameters"),
    [
        ("SimpleNeuralNetwork", {"units_per_layer": [2, 2], "max_epochs": 1}),
        ("NaiveDrugMeanPredictor", {}),
    ],
)
def test_literature_or_bridge_model_lifecycle(model_name: str, hyperparameters: dict) -> None:
    response, cell_line_input, drug_input = _synthetic_data()
    model = MODEL_FACTORY[model_name]()
    model.build_model(hyperparameters)
    model.train(response, cell_line_input, drug_input)
    preds = model.predict(
        response.cell_line_ids,
        response.drug_ids,
        cell_line_input,
        drug_input,
    )
    assert preds.shape == (len(response),)
    assert np.isfinite(preds).all()

    with tempfile.TemporaryDirectory() as directory:
        model.save(directory)
        loaded = type(model).load(directory)
        loaded_preds = loaded.predict(
            response.cell_line_ids,
            response.drug_ids,
            cell_line_input,
            drug_input,
        )
    assert np.allclose(preds, loaded_preds, rtol=1e-5, atol=1e-5)


def test_untrained_component_model_raises() -> None:
    from drevalpy.models import construct_model

    model_cls = construct_model("elasticNet", "geneExpression:fingerprints:elasticNet")
    model = model_cls()
    model.build_model({})
    response, cell_line_input, drug_input = _synthetic_data()
    with pytest.raises(RuntimeError, match="not been trained"):
        model.predict(
            response.cell_line_ids,
            response.drug_ids,
            cell_line_input,
            drug_input,
        )
