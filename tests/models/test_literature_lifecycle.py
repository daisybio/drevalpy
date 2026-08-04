"""Smoke tests for literature models routed through the native facade."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models import construct_model


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


def _multi_view_synthetic_data() -> tuple[DrugResponseDataset, FeatureDataset, FeatureDataset]:
    response = DrugResponseDataset(
        response=np.array([1.0, 2.0, 3.0, 4.0]),
        cell_line_ids=np.array(["cl1", "cl1", "cl2", "cl2"]),
        drug_ids=np.array(["d1", "d2", "d1", "d2"]),
    )
    cell_line_input = FeatureDataset(
        features={
            "cl1": {
                "gene_expression": np.array([0.1, 0.2, 0.3, 0.4]),
                "methylation": np.array([0.2, 0.3, 0.4, 0.5]),
                "mutations": np.array([0.0, 1.0, 0.0, 1.0]),
                "copy_number_variation_gistic": np.array([0.1, 0.1, 0.2, 0.2]),
            },
            "cl2": {
                "gene_expression": np.array([0.5, 0.6, 0.7, 0.8]),
                "methylation": np.array([0.6, 0.7, 0.8, 0.9]),
                "mutations": np.array([1.0, 0.0, 1.0, 0.0]),
                "copy_number_variation_gistic": np.array([0.3, 0.3, 0.4, 0.4]),
            },
        }
    )
    drug_input = FeatureDataset(
        features={
            "d1": {"fingerprints": np.array([1.0, 0.0, 0.5, 0.2])},
            "d2": {"fingerprints": np.array([0.0, 1.0, 0.3, 0.7])},
        }
    )
    return response, cell_line_input, drug_input


LITERATURE_MODEL_NAMES = (
    "DIPK",
    "DrugGNN",
    "MOLIR",
    "PharmaFormer",
    "Precily",
    "SRMF",
    "SimpleNeuralNetwork",
    "MultiViewNeuralNetwork",
    "SuperFELTR",
    "SparseGO",
)


@pytest.mark.parametrize("model_name", LITERATURE_MODEL_NAMES)
def test_literature_models_build_with_defaults(model_name: str) -> None:
    model_cls = construct_model(model_name)
    hyperparameters = dict(model_cls.get_hyperparameter_set()[0])
    if model_name == "DIPK":
        hyperparameters.update({"epochs": 1, "epochs_autoencoder": 1, "heads": 1})
    elif model_name in {"SimpleNeuralNetwork", "MultiViewNeuralNetwork"}:
        hyperparameters.update({"units_per_layer": [2, 2], "max_epochs": 1})
    elif model_name == "PharmaFormer":
        hyperparameters.update({"epochs": 1, "patience": 2})
    elif model_name == "Precily":
        hyperparameters.update({"epochs": 1, "batch_size": 32})
    elif model_name == "SparseGO":
        hyperparameters.update({"epochs": 1, "batch_size": 32})
    model_cls(hyperparameters)


@pytest.mark.parametrize(
    ("model_name", "hyperparameters", "data_factory"),
    [
        ("SimpleNeuralNetwork", {"units_per_layer": [2, 2], "max_epochs": 1}, "_synthetic_data"),
        ("SRMF", {"K": 2, "max_iter": 2, "n_features": 4}, "_synthetic_data"),
        (
            "MultiViewNeuralNetwork",
            {
                "units_per_layer": [2, 2],
                "max_epochs": 1,
                "methylation_pca_components": 2,
            },
            "_multi_view_synthetic_data",
        ),
        ("NaiveDrugMeanPredictor", {}, "_synthetic_data"),
    ],
)
def test_literature_model_lifecycle(
    model_name: str,
    hyperparameters: dict,
    data_factory: str,
) -> None:
    response, cell_line_input, drug_input = globals()[data_factory]()
    model = construct_model(model_name)(hyperparameters)
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
        checkpoint = f"{directory}/model"
        model.save(checkpoint)
        loaded = type(model).load(checkpoint)
        loaded_preds = loaded.predict(
            response.cell_line_ids,
            response.drug_ids,
            cell_line_input,
            drug_input,
        )
    assert np.allclose(preds, loaded_preds, rtol=1e-5, atol=1e-5)


def test_untrained_component_model_raises() -> None:
    from drevalpy.models import construct_model

    model_cls = construct_model("elasticNet", "raw[expression]:fingerprints:elasticNet")
    model = model_cls({})
    response, cell_line_input, drug_input = _synthetic_data()
    with pytest.raises(RuntimeError, match="not been trained"):
        model.predict(
            response.cell_line_ids,
            response.drug_ids,
            cell_line_input,
            drug_input,
        )
