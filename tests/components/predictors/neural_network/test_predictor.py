"""Smoke tests for the neural_network predictor package."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.contracts.training_context import TrainingContext
from drevalpy.components.predictors.neural_network.predictor import NeuralNetworkPredictor
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.models import construct_model
from drevalpy.models.config import from_spec
from drevalpy.registry._builtins import ensure_predictor_registered, register_builtin_components
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.data.batch.response_batch import ResponseBatch
from tests.components.predictors._helpers import neural_batch


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


def test_neural_network_predictor_registry_name() -> None:
    ensure_predictor_registered("neuralNetwork")
    assert get_predictor("neuralNetwork") is NeuralNetworkPredictor


def test_neural_network_requires_numeric_contracts() -> None:
    cls = get_predictor("neuralNetwork")
    assert cls.cell_line_contract.format == FeatureFormat.NUMERIC_MATRIX
    assert cls.drug_contract.format == FeatureFormat.NUMERIC_MATRIX


def test_neural_network_zoo_trains_on_synthetic_data() -> None:
    register_builtin_components()
    import anndata as ad
    import mudata as md
    import pandas as pd

    from drevalpy.types import SplitMask, SplitMasks
    from drevalpy.types.data.dataset import Dataset

    cl_ids = np.array(["cl1", "cl2"])
    drug_ids = np.array(["d1", "d2"])
    response_matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    response_ad = ad.AnnData(
        X=response_matrix,
        obs=pd.DataFrame({"cell_line_name": cl_ids, "tissue": ["Lung", "Blood"]}, index=cl_ids),
        var=pd.DataFrame(index=drug_ids),
    )
    ge_matrix = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float32)
    gene_expression_ad = ad.AnnData(
        X=ge_matrix,
        obs=pd.DataFrame(index=cl_ids),
        var=pd.DataFrame(index=[f"gene{i}" for i in range(4)]),
    )
    response_ad.varm["morgan_fingerprint"] = np.array([[1.0, 0.0, 0.5, 0.2], [0.0, 1.0, 0.3, 0.7]], dtype=np.float32)
    mdata = md.MuData({"response": response_ad, "gene_expression": gene_expression_ad})
    mudataset = Dataset(mdata, name="test")
    split = SplitMasks(
        train=SplitMask(np.array([[True, True], [False, False]])),
        test=SplitMask(np.array([[False, False], [True, True]])),
        val=SplitMask(np.zeros((2, 2), dtype=bool)),
    )

    config = from_spec(
        "SimpleNeuralNetwork",
        hyperparameters={"max_epochs": 1, "batch_size": 2},
    )
    model = construct_model("SimpleNeuralNetwork", config)()
    model.train(mudataset, split)
    preds = model.predict(mudataset, split)
    assert preds.shape[0] > 0
    assert np.isfinite(preds).all()


def test_neural_network_configured_is_not_fitted_before_training() -> None:
    register_builtin_components()
    predictor = NeuralNetworkPredictor(
        hyperparameters={"max_epochs": 1, "units_per_layer": [4, 2]},
    )
    assert predictor._model is None
    assert predictor.is_fitted() is False


def test_neural_network_configured_is_not_fitted() -> None:
    predictor = NeuralNetworkPredictor(
        hyperparameters={"max_epochs": 1, "units_per_layer": [4, 2]},
    )
    assert predictor._model is None
    assert predictor.is_fitted() is False


def _matrix_batch() -> ModelInputBatch:
    response = ResponseBatch(
        response=np.array([1.0, 2.0, 3.0, 4.0]),
        cell_line_ids=np.array(["cl1", "cl1", "cl2", "cl2"]),
        drug_ids=np.array(["d1", "d2", "d1", "d2"]),
    )
    cell_line_features = np.vstack(
        [
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.array([0.5, 0.6, 0.7, 0.8]),
        ]
    )
    drug_features = np.vstack(
        [
            np.array([1.0, 0.0, 0.5, 0.2]),
            np.array([0.0, 1.0, 0.3, 0.7]),
        ]
    )
    return ModelInputBatch.from_response(
        response,
        cell_line_entity_ids=np.array(["cl1", "cl2"]),
        drug_entity_ids=np.array(["d1", "d2"]),
        cell_line_features=cell_line_features,
        drug_features=drug_features,
        cell_line_pair_idx=np.array([0, 0, 1, 1]),
        drug_pair_idx=np.array([0, 1, 0, 1]),
        training_context=TrainingContext(),
    )


def test_neural_network_state_round_trip() -> None:
    predictor = NeuralNetworkPredictor(hyperparameters={"max_epochs": 1, "units_per_layer": [4, 2]})
    predictor.fit(_matrix_batch())
    restored = NeuralNetworkPredictor(hyperparameters={"max_epochs": 1, "units_per_layer": [4, 2]})
    restored.set_state(predictor.get_state())
    assert restored.is_fitted()
    assert restored._input_dim == predictor._input_dim


def test_neural_network_set_state_rejects_invalid_checkpoint() -> None:
    predictor = NeuralNetworkPredictor()
    with pytest.raises(PredictorStateError):
        predictor.set_state({"checkpoint": b"invalid"})


def test_neural_network_early_stopping_wires_validation_loader() -> None:
    predictor = NeuralNetworkPredictor(
        hyperparameters={"max_epochs": 1, "batch_size": 2, "units_per_layer": [4, 2]},
    )
    batch = neural_batch(with_early_stopping=True)
    captured: dict[str, object] = {}

    def _capture_fit(self, model, train_dataloaders, val_dataloaders=None):
        captured["val_loader"] = val_dataloaders
        return None

    with patch("pytorch_lightning.Trainer.fit", _capture_fit):
        predictor.fit(batch)
    assert captured["val_loader"] is not None


def test_neural_network_round_trip_state() -> None:
    predictor = NeuralNetworkPredictor(
        hyperparameters={"max_epochs": 1, "batch_size": 2, "units_per_layer": [4, 2]},
    )
    predictor.fit(neural_batch())
    preds = predictor.predict(neural_batch())
    assert preds.shape == (4,)
    assert np.isfinite(preds).all()

    restored = NeuralNetworkPredictor()
    restored.set_state(predictor.get_state())
    assert restored.is_fitted()
    restored_preds = restored.predict(neural_batch())
    assert np.allclose(preds, restored_preds)


def test_neural_network_set_state_raises_on_invalid_payload() -> None:
    predictor = NeuralNetworkPredictor()
    with pytest.raises(PredictorStateError):
        predictor.set_state({"checkpoint": b"not-a-torch-checkpoint"})
