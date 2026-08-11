"""Tests for typed featurizer block payloads."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pytest

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.core.batch.feature_block import (
    FeatureBlock,
    graph_feature_block,
    merge_feature_blocks,
    metadata_feature_block,
    numeric_feature_block,
    ragged_feature_block,
)


def test_numeric_feature_block_stores_values_and_format() -> None:
    values = np.array([[1.0, 2.0]], dtype=np.float64)
    block = numeric_feature_block(values, feature_names=("a", "b"))
    assert block.format is FeatureFormat.NUMERIC_MATRIX
    assert block.feature_names == ("a", "b")
    assert block.entity_aligned is True
    np.testing.assert_array_equal(block.values, values)


def test_metadata_feature_block_is_not_entity_aligned() -> None:
    block = metadata_feature_block(np.array(["d1", "d2"], dtype=str))
    assert block.entity_aligned is False


def test_feature_block_metadata_is_immutable() -> None:
    block = numeric_feature_block(np.ones((1, 1)), metadata={"dim": 4})
    assert isinstance(block.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        cast(Any, block.metadata)["dim"] = 8


def test_graph_and_ragged_blocks_preserve_object_dtype() -> None:
    graph_payload = object()
    graph = graph_feature_block(np.array([graph_payload], dtype=object))
    ragged = ragged_feature_block(np.array([np.array([1, 2, 3])], dtype=object))
    assert graph.format is FeatureFormat.GRAPH
    assert ragged.format is FeatureFormat.RAGGED_SEQUENCE
    assert graph.values.dtype == object
    assert ragged.values.dtype == object


def test_merge_feature_blocks_rejects_duplicate_names() -> None:
    left = {"gene_expression": numeric_feature_block(np.ones((2, 1)))}
    right = {"gene_expression": numeric_feature_block(np.zeros((2, 1)))}
    with pytest.raises(ValueError, match="Duplicate featurizer block name 'gene_expression'"):
        merge_feature_blocks(left, right)


def test_merge_feature_blocks_combines_distinct_names() -> None:
    merged = merge_feature_blocks(
        {"gene_expression": numeric_feature_block(np.ones((2, 1)))},
        {"fingerprints": numeric_feature_block(np.zeros((2, 3)))},
    )
    assert set(merged) == {"gene_expression", "fingerprints"}
    assert all(isinstance(block, FeatureBlock) for block in merged.values())
