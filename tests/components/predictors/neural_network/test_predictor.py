"""Tests for the neural network predictor."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.predictors.neural_network.predictor import NeuralNetworkPredictor
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.models import DRPModel, ModelSpec
from drevalpy.models.config import from_spec
from drevalpy.registry._builtins import register_builtin_components
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.utils.torch_io import save_trusted_mapping
from tests.components.predictors._helpers import neural_batch


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


def test_registry_name_contracts_and_defaults() -> None:
    cls = get_predictor("neuralNetwork")
    assert cls is NeuralNetworkPredictor
    assert cls.sample_contract.format == FeatureFormat.NUMERIC_MATRIX
    assert cls.drug_contract.format == FeatureFormat.NUMERIC_MATRIX
    assert NeuralNetworkPredictor.get_default_hyperparameters() == {
        "units_per_layer": [32, 16, 8, 4],
        "dropout_prob": 0.3,
        "max_epochs": 100,
        "batch_size": 16,
        "patience": 5,
    }
    spec = NeuralNetworkPredictor.get_hyperparameter_space()["units_per_layer"]
    assert spec["type"] == "categorical"
    assert spec["default"] in spec["choices"]
    for widths in ([32, 16, 8, 4], [128, 64, 32], [64, 64, 32], [16, 8, 4]):
        assert widths in spec["choices"]
    predictor = cls(hyperparameters={"max_epochs": 1, "units_per_layer": [4, 2]})
    assert predictor._model is None
    assert predictor.is_fitted() is False


def test_neural_network_zoo_trains_on_synthetic_data() -> None:
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

    model = DRPModel(
        ModelSpec("SimpleNeuralNetwork", from_spec("scaledGeneExpression:fingerprints:neuralNetwork")),
        {"max_epochs": 1, "batch_size": 2},
    )
    model.train(mudataset, split.train)
    preds = model.predict(mudataset, split.test)
    assert preds.shape[0] > 0
    assert np.isfinite(preds).all()


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
    assert restored._input_dim == predictor._input_dim
    assert restored._hyperparameters == predictor._hyperparameters
    assert np.allclose(preds, restored.predict(neural_batch()))


@pytest.mark.parametrize(
    "state",
    [
        pytest.param({}, id="missing-checkpoint"),
        pytest.param({"checkpoint": "not-bytes"}, id="not-bytes"),
        pytest.param({"checkpoint": b"not-a-torch-checkpoint"}, id="undecodable"),
        pytest.param(
            {"checkpoint": save_trusted_mapping({"input_dim": 3, "state_dict": {}})}, id="no-predictor-hyperparameters"
        ),
    ],
)
def test_neural_network_set_state_rejects_invalid_checkpoint(state: dict[str, object]) -> None:
    with pytest.raises(PredictorStateError):
        NeuralNetworkPredictor().set_state(state)


@pytest.mark.parametrize("missing", ["input_dim", "state_dict"])
def test_neural_network_set_state_requires_input_dim_and_state_dict(missing: str) -> None:
    payload: dict[str, object] = {"predictor_hyperparameters": {}, "input_dim": 3, "state_dict": {}}
    del payload[missing]
    with pytest.raises(PredictorStateError, match=missing):
        NeuralNetworkPredictor().set_state({"checkpoint": save_trusted_mapping(payload)})
