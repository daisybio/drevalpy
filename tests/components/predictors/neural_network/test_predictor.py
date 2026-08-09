"""Smoke tests for the neural_network predictor package."""

from __future__ import annotations

import numpy as np
import pytest

import drevalpy.components.register_builtins as register_builtins
from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.neural_network.predictor import NeuralNetworkPredictor
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.components.registry import get_predictor
from drevalpy.components.training_context import TrainingContext
from drevalpy.data.structures.response_batch import ResponseBatch
from drevalpy.models import construct_model
from drevalpy.models.config import from_spec


def test_neural_network_predictor_registry_name() -> None:
    register_builtins.ensure_predictor_registered("neuralNetwork")
    assert get_predictor("neuralNetwork") is NeuralNetworkPredictor


def test_neural_network_zoo_trains_on_synthetic_data() -> None:
    register_builtin_components()
    import anndata as ad
    import pandas as pd

    import mudata as md
    from drevalpy.data.structures import SplitMasks
    from drevalpy.data.structures.mudataset import MuDataset

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
    response_ad.varm["fingerprints"] = np.array([[1.0, 0.0, 0.5, 0.2], [0.0, 1.0, 0.3, 0.7]], dtype=np.float32)
    mdata = md.MuData({"response": response_ad, "gene_expression": gene_expression_ad})
    mudataset = MuDataset(mdata)
    split = SplitMasks(
        train=np.array([[0, 0], [0, 1]]),
        test=np.array([[1, 0], [1, 1]]),
        val=np.empty((0, 2), dtype=np.intp),
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
