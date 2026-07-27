"""Tests for the cell-line constant featurizer."""

from __future__ import annotations

import numpy as np

from drevalpy.components.featurizers.cell_line.constant import CellLineConstantFeaturizer
from drevalpy.components.registry import get_cell_line_featurizer, get_cell_line_featurizer_metadata
from drevalpy.datasets.dataset import FeatureDataset


def test_cell_line_constant_is_ones_column() -> None:
    featurizer = CellLineConstantFeaturizer()
    features = FeatureDataset(features={"cl1": {"gene_expression": np.array([1.0, 2.0])}})
    entity_ids = np.array(["cl1", "cl2", "cl3"], dtype=str)
    featurizer.fit(features, entity_ids=entity_ids)
    matrix = featurizer.transform(features, entity_ids)
    assert matrix.shape == (3, 1)
    assert matrix.dtype == np.float32
    assert np.allclose(matrix, 1.0)
    assert featurizer.output_dim == 1
    blocks = featurizer.transform_blocks(features, entity_ids)
    assert list(blocks) == ["constant"]
    assert np.allclose(blocks["constant"], 1.0)


def test_cell_line_constant_registered() -> None:
    assert get_cell_line_featurizer("constant") is CellLineConstantFeaturizer
    meta = get_cell_line_featurizer_metadata("constant")
    assert "intercept" in meta["description"].lower() or "constant" in meta["description"].lower()
