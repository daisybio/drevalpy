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


def test_device_follows_cuda_availability() -> None:
    expected = "cuda" if torch.cuda.is_available() else "cpu"

    assert DEVICE.type == expected


def test_module_level_feature_dims_match_the_bionic_and_gene_encoders() -> None:
    assert features_dim_gene == 512
    assert features_dim_bionic == 512


def test_attention_layer_projects_molgnet_features_to_the_hidden_dim() -> None:
    layer = AttentionLayer(heads=1)
    layer.eval()

    output = layer(
        _molgnet(2, 5),
        torch.ones(2, 5),
        torch.randn(2, features_dim_gene),
        torch.randn(2, features_dim_bionic),
    )

    assert output.shape == (2, HIDDEN_DIM)


def test_attention_layer_squeezes_a_single_row_batch_to_one_dimension() -> None:
    layer = AttentionLayer(heads=1)
    layer.eval()

    output = layer(
        _molgnet(1, 3),
        torch.ones(1, 3),
        torch.randn(1, features_dim_gene),
        torch.randn(1, features_dim_bionic),
    )

    assert output.shape == (HIDDEN_DIM,)


def test_attention_layer_rejects_a_head_count_that_does_not_divide_the_hidden_dim() -> None:
    with pytest.raises(ValueError, match="divisible by the number of heads"):
        AttentionLayer(heads=5)


def test_dense_layers_reduce_attention_output_to_one_scalar_per_row() -> None:
    dense = DenseLayers(fc_layer_num=3, fc_layer_dim=FC_LAYER_DIM, dropout_rate=0.0)
    dense.eval()

    output = dense(
        torch.randn(4, HIDDEN_DIM),
        torch.randn(4, features_dim_gene),
        torch.randn(4, features_dim_bionic),
    )

    assert output.shape == (4, 1)


def test_dense_layers_add_a_batch_axis_to_a_one_dimensional_input() -> None:
    dense = DenseLayers(fc_layer_num=3, fc_layer_dim=FC_LAYER_DIM, dropout_rate=0.0)
    dense.eval()

    output = dense(
        torch.randn(HIDDEN_DIM),
        torch.randn(1, features_dim_gene),
        torch.randn(1, features_dim_bionic),
    )

    assert output.shape == (1, 1)


def test_dense_layers_build_one_dropout_per_requested_layer() -> None:
    dense = DenseLayers(fc_layer_num=4, fc_layer_dim=FC_LAYER_DIM, dropout_rate=0.25)

    assert len(dense.dropout_layers) == 4
    assert {layer.p for layer in dense.dropout_layers} == {0.25}


def test_dense_layers_size_the_output_head_from_the_penultimate_layer_dim() -> None:
    dense = DenseLayers(fc_layer_num=3, fc_layer_dim=FC_LAYER_DIM, dropout_rate=0.0)

    assert isinstance(dense.fc_output, nn.Linear)
    assert dense.fc_output.in_features == FC_LAYER_DIM[1]
    assert dense.fc_output.out_features == 1


def test_predictor_scores_one_response_per_pair() -> None:
    predictor = Predictor(heads=1, fc_layer_num=3, fc_layer_dim=FC_LAYER_DIM, dropout_rate=0.0)
    predictor.eval()

    with torch.no_grad():
        output = predictor(
            _molgnet(2, 4),
            torch.randn(2, features_dim_gene),
            torch.randn(2, features_dim_bionic),
            torch.ones(2, 4),
        )

    assert output.shape == (2, 1)
    assert torch.isfinite(output).all()


def test_predictor_composes_the_attention_and_dense_stages() -> None:
    predictor = Predictor(heads=1, fc_layer_num=3, fc_layer_dim=FC_LAYER_DIM, dropout_rate=0.1)

    assert isinstance(predictor.attention_layer, AttentionLayer)
    assert isinstance(predictor.dense_layers, DenseLayers)


def test_predictor_gradients_reach_the_attention_stage() -> None:
    predictor = Predictor(heads=1, fc_layer_num=3, fc_layer_dim=FC_LAYER_DIM, dropout_rate=0.0)

    predictor(
        _molgnet(2, 3),
        torch.randn(2, features_dim_gene),
        torch.randn(2, features_dim_bionic),
        torch.ones(2, 3),
    ).sum().backward()

    assert predictor.attention_layer.fc_layer_0.weight.grad is not None
