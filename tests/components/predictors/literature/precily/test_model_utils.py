"""Tests for the Precily feed-forward regressor."""

from __future__ import annotations

import torch
from torch import nn

from drevalpy.components.predictors.literature.precily.model_utils import PrecilyNetwork


def test_precily_network_predicts_one_scalar_per_row() -> None:
    network = PrecilyNetwork(input_dim=12)

    output = network(torch.randn(5, 12))

    assert output.shape == (5,)


def test_precily_network_squeezes_a_single_row_batch_to_one_element() -> None:
    network = PrecilyNetwork(input_dim=6)
    network.eval()

    output = network(torch.randn(1, 6))

    assert output.shape == (1,)


def test_precily_network_reproduces_the_reference_layer_widths() -> None:
    network = PrecilyNetwork(input_dim=7)

    linear_shapes = [(layer.in_features, layer.out_features) for layer in network.net if isinstance(layer, nn.Linear)]

    assert linear_shapes == [(7, 1429), (1429, 512), (512, 140), (140, 200), (200, 1)]


def test_precily_network_uses_the_requested_dropout_probability() -> None:
    network = PrecilyNetwork(input_dim=4, dropout=0.42)

    probabilities = {layer.p for layer in network.net if isinstance(layer, nn.Dropout)}

    assert probabilities == {0.42}


def test_precily_network_defaults_to_dropout_of_one_tenth() -> None:
    network = PrecilyNetwork(input_dim=4)

    probabilities = {layer.p for layer in network.net if isinstance(layer, nn.Dropout)}

    assert probabilities == {0.1}


def test_precily_network_is_deterministic_in_eval_mode() -> None:
    network = PrecilyNetwork(input_dim=8)
    network.eval()
    features = torch.randn(3, 8)

    with torch.no_grad():
        first = network(features)
        second = network(features)

    assert torch.allclose(first, second)


def test_precily_network_output_is_differentiable() -> None:
    network = PrecilyNetwork(input_dim=5)

    network(torch.randn(4, 5)).sum().backward()

    assert network.net[0].weight.grad is not None
