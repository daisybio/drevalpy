"""Tests for the DIPK attention, dense, and combined predictor modules."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from drevalpy.components.predictors.literature.dipk.model_utils import (
    DEVICE,
    AttentionLayer,
    DenseLayers,
    Predictor,
    features_dim_bionic,
    features_dim_gene,
)

FC_LAYER_DIM = [16, 8, 4, 4, 4, 4]
HIDDEN_DIM = 768


def _molgnet(batch_size: int, seq_len: int) -> torch.Tensor:
    return torch.randn(batch_size, seq_len, HIDDEN_DIM)


def test_device_and_feature_dims() -> None:
    assert DEVICE.type == ("cuda" if torch.cuda.is_available() else "cpu")
    assert features_dim_gene == features_dim_bionic == 512


def test_attention_layer_projects_to_the_hidden_dim_and_squeezes_a_single_row() -> None:
    layer = AttentionLayer(heads=1)
    layer.eval()

    def run(batch_size: int, seq_len: int) -> torch.Tensor:
        return layer(
            _molgnet(batch_size, seq_len),
            torch.ones(batch_size, seq_len),
            torch.randn(batch_size, features_dim_gene),
            torch.randn(batch_size, features_dim_bionic),
        )

    assert run(2, 5).shape == (2, HIDDEN_DIM)
    assert run(1, 3).shape == (HIDDEN_DIM,)
    with pytest.raises(ValueError, match="divisible by the number of heads"):
        AttentionLayer(heads=5)


def test_dense_layers_reduce_to_one_scalar_per_row_and_size_their_head() -> None:
    dense = DenseLayers(fc_layer_num=3, fc_layer_dim=FC_LAYER_DIM, dropout_rate=0.0)
    dense.eval()
    gene, bionic = torch.randn(4, features_dim_gene), torch.randn(4, features_dim_bionic)

    assert dense(torch.randn(4, HIDDEN_DIM), gene, bionic).shape == (4, 1)
    assert dense(torch.randn(HIDDEN_DIM), gene[:1], bionic[:1]).shape == (1, 1)
    assert isinstance(dense.fc_output, nn.Linear)
    assert (dense.fc_output.in_features, dense.fc_output.out_features) == (FC_LAYER_DIM[1], 1)
    dropout = DenseLayers(fc_layer_num=4, fc_layer_dim=FC_LAYER_DIM, dropout_rate=0.25)
    assert {layer.p for layer in dropout.dropout_layers} == {0.25}
    assert len(dropout.dropout_layers) == 4


def test_predictor_scores_one_response_per_pair_and_trains_the_attention_stage() -> None:
    predictor = Predictor(heads=1, fc_layer_num=3, fc_layer_dim=FC_LAYER_DIM, dropout_rate=0.0)
    assert isinstance(predictor.attention_layer, AttentionLayer)
    assert isinstance(predictor.dense_layers, DenseLayers)

    def run(seq_len: int) -> torch.Tensor:
        return predictor(
            _molgnet(2, seq_len),
            torch.randn(2, features_dim_gene),
            torch.randn(2, features_dim_bionic),
            torch.ones(2, seq_len),
        )

    predictor.eval()
    with torch.no_grad():
        output = run(4)
    assert output.shape == (2, 1)
    assert torch.isfinite(output).all()

    predictor.train()
    run(3).sum().backward()
    assert predictor.attention_layer.fc_layer_0.weight.grad is not None
