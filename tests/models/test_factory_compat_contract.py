"""Public interface contract tests for MODEL_FACTORY entries."""

from __future__ import annotations

import importlib
import tempfile

import numpy as np
import pytest

from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models import MODEL_FACTORY

_RUNNABLE_BASELINES = (
    "ElasticNet",
    "NaivePredictor",
    "NaiveDrugMeanPredictor",
    "RandomForest",
    "SingleDrugElasticNet",
)


def _minimal_hyperparameters(model_name: str) -> dict:
    model = MODEL_FACTORY[model_name]()
    hyperparameters = model.get_hyperparameter_set()[0]
    if model_name == "DIPK":
        return {**hyperparameters, "epochs": 1, "epochs_autoencoder": 1, "heads": 1}
    if model_name in {"SimpleNeuralNetwork", "MultiViewNeuralNetwork"}:
        return {**hyperparameters, "units_per_layer": [2, 2], "max_epochs": 1}
    if model_name == "PharmaFormer":
        return {**hyperparameters, "epochs": 1, "patience": 2}
    if model_name == "Precily":
        return {**hyperparameters, "epochs": 1, "batch_size": 32}
    return hyperparameters


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


@pytest.mark.parametrize("model_name", sorted(MODEL_FACTORY))
def test_factory_entry_instantiates_and_builds(model_name: str) -> None:
    model = MODEL_FACTORY[model_name]()
    assert model.get_model_name() == model_name
    hyperparameters = _minimal_hyperparameters(model_name)
    try:
        model.build_model(hyperparameters)
    except ImportError as exc:
        if model_name in {"MultiViewXGBoost", "MultiViewLightGBM"}:
            pytest.skip(str(exc))
        raise


@pytest.mark.parametrize("model_name", _RUNNABLE_BASELINES)
def test_factory_entry_train_predict_save_load(model_name: str) -> None:
    response = DrugResponseDataset(
        response=np.array([1.0, 2.0, 3.0, 4.0]),
        cell_line_ids=np.array(["cl1", "cl1", "cl2", "cl2"]),
        drug_ids=np.array(["d1", "d2", "d1", "d2"]),
    )
    cell_line_input = FeatureDataset(
        features={
            "cl1": {"gene_expression": np.array([0.1, 0.2, 0.3])},
            "cl2": {"gene_expression": np.array([0.4, 0.5, 0.6])},
        }
    )
    drug_input: FeatureDataset | None = FeatureDataset(
        features={
            "d1": {"fingerprints": np.array([1.0, 0.0])},
            "d2": {"fingerprints": np.array([0.0, 1.0])},
        }
    )
    model = MODEL_FACTORY[model_name]()
    hyperparameters = _minimal_hyperparameters(model_name)
    model.build_model(hyperparameters)
    if model_name.startswith("SingleDrug"):
        drug_input = None
    model.train(response, cell_line_input, drug_input)
    preds = model.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    assert preds.shape == (4,)
    with tempfile.TemporaryDirectory() as tmp:
        model.save(tmp)
        loaded = MODEL_FACTORY[model_name].load(tmp)
        loaded_preds = loaded.predict(response.cell_line_ids, response.drug_ids, cell_line_input, drug_input)
    assert np.allclose(preds, loaded_preds, equal_nan=True)


def test_legacy_naive_import_path() -> None:
    module = importlib.import_module("drevalpy.models.baselines.naive_pred")
    assert hasattr(module, "NaivePredictor")
