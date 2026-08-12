"""Tests for the DIPK multi-head attention layer."""

from __future__ import annotations

import pytest
import torch

from drevalpy.components.predictors.literature.dipk.attention_utils import MultiHeadAttentionLayer


def _layer(*, hid_dim: int = 8, n_heads: int = 2, dropout: float = 0.0) -> MultiHeadAttentionLayer:
    layer = MultiHeadAttentionLayer(hid_dim=hid_dim, n_heads=n_heads, dropout=dropout, device="cpu")
    layer.eval()
    return layer


def test_attention_layer_rejects_a_head_count_that_does_not_divide_the_hidden_dim() -> None:
    with pytest.raises(ValueError, match="divisible by the number of heads"):
        MultiHeadAttentionLayer(hid_dim=10, n_heads=4, dropout=0.0, device="cpu")


def test_attention_layer_splits_the_hidden_dim_evenly_across_heads() -> None:
    layer = _layer(hid_dim=12, n_heads=3)

    assert layer.head_dim == 4


def test_attention_layer_scale_is_the_square_root_of_the_head_dim() -> None:
    layer = _layer(hid_dim=16, n_heads=4)

    assert layer.scale.item() == pytest.approx(2.0)


def test_attention_layer_output_keeps_the_query_sequence_length() -> None:
    layer = _layer(hid_dim=8, n_heads=2)
    query = torch.randn(3, 1, 8)
    key = torch.randn(3, 5, 8)

    output, attention = layer(query, key, key)

    assert output.shape == (3, 1, 8)
    assert attention.shape == (3, 2, 1, 5)


def test_attention_weights_sum_to_one_over_the_key_axis() -> None:
    layer = _layer(hid_dim=8, n_heads=2)
    query = torch.randn(2, 3, 8)
    key = torch.randn(2, 3, 8)

    _, attention = layer(query, key, key)

    torch.testing.assert_close(attention.sum(dim=-1), torch.ones(2, 2, 3))


def test_attention_mask_removes_masked_keys_from_the_weights() -> None:
    layer = _layer(hid_dim=8, n_heads=2)
    query = torch.randn(1, 1, 8)
    key = torch.randn(1, 4, 8)
    mask = torch.tensor([[[[1, 1, 0, 0]]]])

    _, attention = layer(query, key, key, mask)

    assert attention[..., 2:].abs().max().item() == pytest.approx(0.0)
    torch.testing.assert_close(attention[..., :2].sum(dim=-1), torch.ones(1, 2, 1))


def test_attention_layer_is_differentiable() -> None:
    layer = _layer(hid_dim=8, n_heads=2)
    features = torch.randn(2, 3, 8)

    layer(features, features, features)[0].sum().backward()

    assert layer.fc_q.weight.grad is not None


def test_attention_layer_with_one_head_matches_its_hidden_dim() -> None:
    layer = _layer(hid_dim=6, n_heads=1)

    output, attention = layer(torch.randn(2, 1, 6), torch.randn(2, 2, 6), torch.randn(2, 2, 6))

    assert layer.head_dim == 6
    assert output.shape == (2, 1, 6)
    assert attention.shape == (2, 1, 1, 2)
