"""Tests for the dense feed-forward Lightning network."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from drevalpy.components.predictors.neural_network.network import FeedForwardNetwork

HPAMS = {"units_per_layer": [8, 4, 2], "dropout_prob": 0.0}


def _network(input_dim: int = 6, **overrides: object) -> FeedForwardNetwork:
    hyperparameters = {**HPAMS, **overrides}
    network = FeedForwardNetwork(hyperparameters, input_dim)
    network.eval()
    return network


def test_network_rejects_non_list_units_per_layer() -> None:
    with pytest.raises(TypeError, match="units_per_layer must be a list of integers"):
        FeedForwardNetwork({"units_per_layer": 8, "dropout_prob": 0.1}, 4)


def test_network_rejects_non_integer_layer_widths() -> None:
    with pytest.raises(TypeError, match="units_per_layer must be a list of integers"):
        FeedForwardNetwork({"units_per_layer": [8, 4.5], "dropout_prob": 0.1}, 4)


def test_network_rejects_an_integer_dropout_probability() -> None:
    with pytest.raises(TypeError, match="dropout_prob must be a float"):
        FeedForwardNetwork({"units_per_layer": [8, 4], "dropout_prob": 0}, 4)


def test_network_builds_one_linear_per_layer_plus_an_output_head() -> None:
    network = _network(input_dim=6)

    assert len(network.fully_connected_layers) == 4
    assert network.fully_connected_layers[0].in_features == 6
    assert network.fully_connected_layers[-1].out_features == 1


def test_network_chains_the_requested_layer_widths() -> None:
    network = _network(input_dim=6)

    widths = [(layer.in_features, layer.out_features) for layer in network.fully_connected_layers]

    assert widths == [(6, 8), (8, 4), (4, 2), (2, 1)]


def test_network_builds_one_batch_norm_per_hidden_layer() -> None:
    network = _network()

    assert len(network.batch_norm_layers) == 3
    assert all(isinstance(layer, nn.BatchNorm1d) for layer in network.batch_norm_layers)


def test_network_uses_the_requested_dropout_probability() -> None:
    network = _network(dropout_prob=0.35)

    assert network.dropout_layer is not None
    assert network.dropout_layer.p == pytest.approx(0.35)


def test_network_predicts_one_flat_scalar_per_row() -> None:
    network = _network(input_dim=6)

    with torch.no_grad():
        output = network(torch.randn(5, 6))

    assert output.shape == (5,)
    assert torch.isfinite(output).all()


def test_network_supports_a_single_hidden_layer() -> None:
    network = _network(input_dim=4, units_per_layer=[3])

    with torch.no_grad():
        output = network(torch.randn(2, 4))

    assert output.shape == (2,)


def test_unpack_batch_concatenates_all_but_the_last_tensor() -> None:
    cell = torch.zeros(3, 2)
    drug = torch.ones(3, 4)
    response = torch.arange(3, dtype=torch.float32)

    features, unpacked_response = FeedForwardNetwork._unpack_batch((cell, drug, response))

    assert features.shape == (3, 6)
    assert torch.allclose(unpacked_response, response)


def test_unpack_batch_handles_a_cell_line_only_batch() -> None:
    cell = torch.zeros(2, 5)
    response = torch.ones(2)

    features, _ = FeedForwardNetwork._unpack_batch((cell, response))

    assert features.shape == (2, 5)


def test_training_step_returns_a_scalar_mse_loss() -> None:
    network = FeedForwardNetwork(HPAMS, 6)
    batch = (torch.randn(4, 4), torch.randn(4, 2), torch.randn(4))

    loss = network.training_step(batch, 0)

    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_validation_step_returns_a_scalar_mse_loss() -> None:
    network = FeedForwardNetwork(HPAMS, 6)
    batch = (torch.randn(4, 4), torch.randn(4, 2), torch.randn(4))

    loss = network.validation_step(batch, 0)

    assert loss.ndim == 0


def test_configure_optimizers_returns_adam_over_all_parameters() -> None:
    network = _network()

    optimizer = network.configure_optimizers()

    assert isinstance(optimizer, torch.optim.Adam)
    assert sum(len(group["params"]) for group in optimizer.param_groups) == len(list(network.parameters()))
