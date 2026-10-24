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
