"""Feature blocks, block specs, and the featurizer result: projection and row lookup."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pytest

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.types.data.batch.feature_block import (
    BlockSpec,
    FeatureBlock,
    FeaturizerResult,
    global_feature_block,
    graph_feature_block,
    merge_feature_blocks,
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


def test_global_feature_block_is_not_entity_aligned() -> None:
    block = global_feature_block(np.array(["d1", "d2"], dtype=str))
    assert block.entity_aligned is False


def test_block_spec_is_entity_aligned_unless_declared_global() -> None:
    assert BlockSpec("view", FeatureFormat.NUMERIC_MATRIX).entity_aligned is True
    assert BlockSpec("view", FeatureFormat.NUMERIC_MATRIX, entity_aligned=False).entity_aligned is False


def test_feature_block_annotations_are_immutable() -> None:
    block = numeric_feature_block(np.ones((1, 1)), annotations={"dim": 4})
    assert isinstance(block.annotations, MappingProxyType)
    with pytest.raises(TypeError):
        cast(Any, block.annotations)["dim"] = 8


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


def test_projection_rules_cover_empty_one_and_many_numeric_blocks() -> None:
    ids = np.array(["a", "b"])
    first = numeric_feature_block(np.array([[1.0], [2.0]]))
    second = numeric_feature_block(np.array([[3.0], [4.0]]))
    assert FeaturizerResult(ids, {"a": first}, ()).matrix.shape == (2, 0)
    assert FeaturizerResult(ids, {"a": first}, ("a",)).matrix is first.values
    np.testing.assert_allclose(FeaturizerResult(ids, {"a": first, "b": second}, ("a", "b")).matrix, [[1, 3], [2, 4]])


def test_projection_rejects_missing_duplicate_or_non_numeric_multi_block() -> None:
    ids = np.array(["a"])
    numeric = numeric_feature_block(np.array([[1.0]]))
    object_block = FeatureBlock(np.array([object()], dtype=object), numeric.format)
    with pytest.raises(ValueError, match="missing block"):
        FeaturizerResult(ids, {"a": numeric}, ("missing",))
    with pytest.raises(ValueError, match="cannot repeat"):
        FeaturizerResult(ids, {"a": numeric}, ("a", "a"))
    with pytest.raises(ValueError, match="Multiple matrix blocks"):
        FeaturizerResult(ids, {"a": numeric, "b": object_block}, ("a", "b"))


def test_projection_rejects_global_blocks() -> None:
    ids = np.array(["a"])
    with pytest.raises(ValueError, match="cannot select global block"):
        FeaturizerResult(ids, {"g": global_feature_block(np.ones(3))}, ("g",))


def test_from_blocks_projects_only_entity_aligned_numeric_blocks() -> None:
    ids = np.array(["a", "b"])
    blocks = {
        "numeric": numeric_feature_block(np.ones((2, 1))),
        "global": global_feature_block(np.ones(5)),
        "graph": graph_feature_block(np.array([object(), object()], dtype=object)),
    }
    assert FeaturizerResult.from_blocks(blocks, ids).projection == ("numeric",)


def test_result_rejects_mismatched_rows_and_scalars() -> None:
    ids = np.array(["cl1"])
    with pytest.raises(ValueError, match="rows must match entity_ids"):
        FeaturizerResult.from_blocks({"view": numeric_feature_block(np.ones((2, 1)))}, ids)
    with pytest.raises(ValueError, match="rows must match entity_ids"):
        FeaturizerResult.from_blocks({"view": numeric_feature_block(np.array(1.0))}, ids)


def _result(ids: list[str]) -> FeaturizerResult:
    return FeaturizerResult(np.asarray(ids), {}, ())


class TestRowsFor:
    def test_returns_the_row_of_each_identifier_in_the_requested_order(self) -> None:
        rows = _result(["cl1", "cl2", "cl3"]).rows_for(np.array(["cl3", "cl1", "cl3"]))
        assert rows.tolist() == [2, 0, 2]
        assert np.issubdtype(rows.dtype, np.integer)

    def test_no_identifiers_yield_no_rows(self) -> None:
        assert _result(["cl1"]).rows_for(np.array([], dtype=str)).size == 0

    def test_a_missing_identifier_raises(self) -> None:
        with pytest.raises(ValueError, match=r"Missing identifiers in featurizer rows: 'nope'"):
            _result(["cl1"]).rows_for(np.array(["cl1", "nope"]))

    def test_the_error_previews_five_missing_identifiers(self) -> None:
        with pytest.raises(ValueError, match=r"'m0', 'm1', 'm2', 'm3', 'm4' \(\+2 more\)"):
            _result(["cl1"]).rows_for(np.array([f"m{i}" for i in range(7)]))

    def test_duplicate_entity_identifiers_raise(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            _result(["cl1", "cl1"]).rows_for(np.array(["cl1"]))
