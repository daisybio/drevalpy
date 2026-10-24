"""Tests for the dense feed-forward network and the neural network predictor built on it."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.predictors.neural_network.network import FeedForwardNetwork
from drevalpy.components.predictors.neural_network.predictor import NeuralNetworkPredictor
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.models import DRPModel, ModelSpec
from drevalpy.models.config import from_spec
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.utils.torch_io import save_trusted_mapping
from tests.components.predictors._helpers import neural_batch
from tests.models.synthetic_fixtures import lso_split_masks, synthetic_mudataset

HPAMS = {"units_per_layer": [8, 4, 2], "dropout_prob": 0.0}


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
    mudataset = synthetic_mudataset()
    split = lso_split_masks()

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


def _network(input_dim: int = 6, **overrides: object) -> FeedForwardNetwork:
    hyperparameters = {**HPAMS, **overrides}
    network = FeedForwardNetwork(hyperparameters, input_dim)
    network.eval()
    return network


@pytest.mark.parametrize(
    ("hyperparameters", "message"),
    [
        ({"units_per_layer": 8, "dropout_prob": 0.1}, "units_per_layer must be a list of integers"),
        ({"units_per_layer": [8, 4.5], "dropout_prob": 0.1}, "units_per_layer must be a list of integers"),
        ({"units_per_layer": [8, 4], "dropout_prob": 0}, "dropout_prob must be a float"),
    ],
)
def test_network_rejects_invalid_hyperparameters(hyperparameters: dict, message: str) -> None:
    with pytest.raises(TypeError, match=message):
        FeedForwardNetwork(hyperparameters, 4)


def test_network_chains_the_requested_layer_widths_and_predicts_one_scalar_per_row() -> None:
    network = _network(input_dim=6, dropout_prob=0.35)

    widths = [(layer.in_features, layer.out_features) for layer in network.fully_connected_layers]
    assert widths == [(6, 8), (8, 4), (4, 2), (2, 1)]
    assert len(network.batch_norm_layers) == 3
    assert all(isinstance(layer, nn.BatchNorm1d) for layer in network.batch_norm_layers)
    assert network.dropout_layer is not None
    assert network.dropout_layer.p == pytest.approx(0.35)
    with torch.no_grad():
        output = network(torch.randn(5, 6))
        single_hidden = _network(input_dim=4, units_per_layer=[3])(torch.randn(2, 4))
    assert output.shape == (5,)
    assert torch.isfinite(output).all()
    assert single_hidden.shape == (2,)


def test_unpack_batch_concatenates_all_but_the_last_tensor() -> None:
    response = torch.arange(3, dtype=torch.float32)

    features, unpacked_response = FeedForwardNetwork._unpack_batch((torch.zeros(3, 2), torch.ones(3, 4), response))
    sample_only, _ = FeedForwardNetwork._unpack_batch((torch.zeros(2, 5), torch.ones(2)))

    assert features.shape == (3, 6)
    assert torch.allclose(unpacked_response, response)
    assert sample_only.shape == (2, 5)


def test_steps_return_a_scalar_mse_loss_and_the_optimizer_is_adam_over_all_parameters() -> None:
    network = FeedForwardNetwork(HPAMS, 6)
    batch = (torch.randn(4, 4), torch.randn(4, 2), torch.randn(4))

    loss = network.training_step(batch, 0)

    assert loss.ndim == 0
    assert loss.item() >= 0.0
    assert network.validation_step(batch, 0).ndim == 0
    optimizer = network.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam)
    assert sum(len(group["params"]) for group in optimizer.param_groups) == len(list(network.parameters()))
