"""Tests for the Precily feed-forward regressor."""

from __future__ import annotations

import torch
from torch import nn

from drevalpy.components.predictors.literature.precily.model_utils import PrecilyNetwork


def test_network_predicts_one_deterministic_differentiable_scalar_per_row() -> None:
    network = PrecilyNetwork(input_dim=8)
    features = torch.randn(3, 8)

    network.eval()
    with torch.no_grad():
        first = network(features)
        second = network(features)
        single = network(features[:1])
    assert first.shape == (3,)
    assert torch.allclose(first, second)
    assert single.shape == (1,)

    network.train()
    network(features).sum().backward()
    assert network.net[0].weight.grad is not None


def test_network_reproduces_the_reference_layer_widths_and_dropout() -> None:
    network = PrecilyNetwork(input_dim=7)
    linear_shapes = [(layer.in_features, layer.out_features) for layer in network.net if isinstance(layer, nn.Linear)]

    assert linear_shapes == [(7, 1429), (1429, 512), (512, 140), (140, 200), (200, 1)]
    assert {layer.p for layer in network.net if isinstance(layer, nn.Dropout)} == {0.1}
    custom = PrecilyNetwork(input_dim=4, dropout=0.42)
    assert {layer.p for layer in custom.net if isinstance(layer, nn.Dropout)} == {0.42}
