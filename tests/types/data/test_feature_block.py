"""Feature blocks, block specs, and the featurizer result: matrix materialization and row lookup."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.types.data.feature_block import FeatureBlock, FeaturizerOutput
from drevalpy.types.data.feature_format import BlockSpec, FeatureFormat


def test_feature_block_defaults_to_numeric_entity_aligned() -> None:
    values = np.array([[1.0, 2.0]], dtype=np.float64)
    block = FeatureBlock(values, feature_names=("a", "b"))
    assert block.format is FeatureFormat.NUMERIC_MATRIX
    assert block.feature_names == ("a", "b")
    assert block.entity_aligned is True
    np.testing.assert_array_equal(block.values, values)


def test_block_spec_defaults_to_numeric_matrix() -> None:
    assert BlockSpec("gene_expression") == BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX)


def test_global_feature_block_is_not_entity_aligned() -> None:
    block = FeatureBlock(np.array(["d1", "d2"], dtype=str), entity_aligned=False)
    assert block.entity_aligned is False


def test_graph_and_ragged_blocks_preserve_object_dtype() -> None:
    graph_payload = object()
    graph = FeatureBlock(np.array([graph_payload], dtype=object), FeatureFormat.GRAPH)
    ragged = FeatureBlock(np.array([np.array([1, 2, 3])], dtype=object), FeatureFormat.RAGGED_SEQUENCE)
    assert graph.format is FeatureFormat.GRAPH
    assert ragged.format is FeatureFormat.RAGGED_SEQUENCE
    assert graph.values.dtype == object
    assert ragged.values.dtype == object


def test_matrix_covers_empty_one_and_many_numeric_blocks() -> None:
    ids = np.array(["a", "b"])
    first = FeatureBlock(np.array([[1.0], [2.0]]))
    second = FeatureBlock(np.array([[3.0], [4.0]]))
    assert FeaturizerOutput(ids, {}).matrix.shape == (2, 0)
    assert FeaturizerOutput(ids, {"a": first}).matrix is first.values
    np.testing.assert_allclose(FeaturizerOutput(ids, {"a": first, "b": second}).matrix, [[1, 3], [2, 4]])


def test_matrix_skips_global_and_non_numeric_blocks_and_rejects_malformed_numeric() -> None:
    ids = np.array(["a", "b"])
    numeric = FeatureBlock(np.ones((2, 1)))
    blocks = {
        "numeric": numeric,
        "global": FeatureBlock(np.ones(5), entity_aligned=False),
        "graph": FeatureBlock(np.array([object(), object()], dtype=object), FeatureFormat.GRAPH),
    }
    assert FeaturizerOutput(ids, blocks).matrix is numeric.values

    bad_numeric = FeatureBlock(np.array([object()], dtype=object), FeatureFormat.NUMERIC_MATRIX)
    with pytest.raises(ValueError, match="two-dimensional numeric arrays"):
        _ = FeaturizerOutput(np.array(["a"]), {"bad": bad_numeric}).matrix


def test_result_rejects_mismatched_rows_and_scalars() -> None:
    ids = np.array(["cl1"])
    with pytest.raises(ValueError, match="rows must match entity_ids"):
        FeaturizerOutput(ids, {"view": FeatureBlock(np.ones((2, 1)))})
    with pytest.raises(ValueError, match="rows must match entity_ids"):
        FeaturizerOutput(ids, {"view": FeatureBlock(np.array(1.0))})


def _result(ids: list[str]) -> FeaturizerOutput:
    return FeaturizerOutput(np.asarray(ids), {})


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
