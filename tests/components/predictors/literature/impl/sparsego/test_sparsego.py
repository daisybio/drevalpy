"""Tests for SparseGO layer modules."""

from __future__ import annotations

import pytest
import torch

from drevalpy.components.predictors.literature.impl.sparsego.sparsego import SparseLinearNew


def test_sparse_linear_new_with_explicit_connectivity() -> None:
    connectivity = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    layer = SparseLinearNew(in_features=2, out_features=2, connectivity=connectivity)
    output = layer(torch.tensor([[1.0, 2.0]]))
    assert output.shape == (1, 2)


def test_sparse_linear_new_rejects_invalid_connectivity_shape() -> None:
    bad = torch.tensor([[0, 1, 2]], dtype=torch.long)
    with pytest.raises(ValueError, match="connectivity should be"):
        SparseLinearNew(in_features=2, out_features=2, connectivity=bad)
