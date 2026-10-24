"""Tests for the DIPK multi-head attention layer."""

from __future__ import annotations

import pytest
import torch

from drevalpy.components.predictors.literature.dipk.attention_utils import MultiHeadAttentionLayer


def _layer(*, hid_dim: int = 8, n_heads: int = 2, dropout: float = 0.0) -> MultiHeadAttentionLayer:
    layer = MultiHeadAttentionLayer(hid_dim=hid_dim, n_heads=n_heads, dropout=dropout, device="cpu")
    layer.eval()
    return layer


def test_layer_rejects_indivisible_heads_and_splits_the_hidden_dim() -> None:
    with pytest.raises(ValueError, match="divisible by the number of heads"):
        MultiHeadAttentionLayer(hid_dim=10, n_heads=4, dropout=0.0, device="cpu")
    layer = _layer(hid_dim=16, n_heads=4)
    assert layer.head_dim == 4
    assert layer.scale.item() == pytest.approx(2.0)
    assert _layer(hid_dim=6, n_heads=1).head_dim == 6


def test_output_keeps_the_query_length_and_weights_sum_to_one_and_are_differentiable() -> None:
    layer = _layer(hid_dim=8, n_heads=2)
    query = torch.randn(3, 1, 8)
    key = torch.randn(3, 5, 8)

    output, attention = layer(query, key, key)

    assert output.shape == (3, 1, 8)
    assert attention.shape == (3, 2, 1, 5)
    torch.testing.assert_close(attention.sum(dim=-1), torch.ones(3, 2, 1))
    output.sum().backward()
    assert layer.fc_q.weight.grad is not None


def test_attention_mask_removes_masked_keys_from_the_weights() -> None:
    layer = _layer(hid_dim=8, n_heads=2)
    key = torch.randn(1, 4, 8)
    mask = torch.tensor([[[[1, 1, 0, 0]]]])

    _, attention = layer(torch.randn(1, 1, 8), key, key, mask)

    assert attention[..., 2:].abs().max().item() == pytest.approx(0.0)
    torch.testing.assert_close(attention[..., :2].sum(dim=-1), torch.ones(1, 2, 1))
