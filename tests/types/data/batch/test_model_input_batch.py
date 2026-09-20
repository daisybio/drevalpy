"""Tests for pair indexing and explicit feature-result projections."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.types.data.batch.feature_block import FeatureBlock, FeaturizerResult, numeric_feature_block
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch, pair_cell_line_indices, pair_drug_indices
from drevalpy.types.data.batch.response_batch import ResponseBatch


def _result(ids, values, projection=("view",)) -> FeaturizerResult:
    return FeaturizerResult(np.asarray(ids), {"view": numeric_feature_block(np.asarray(values))}, projection)


def _batch(*, drug=True, stopping=None) -> ModelInputBatch:
    cell = _result(["cl1", "cl2"], [[0.1, 0.2], [0.3, 0.4]])
    drug_result = _result(["d1", "d2"], [[1.0, 0.0], [0.0, 1.0]]) if drug else None
    return ModelInputBatch(
        np.array(["cl1", "cl2"]),
        np.array(["d1", "d2"]),
        np.array([1.0, 2.0]),
        cell,
        drug_result,
        np.array([0, 1]),
        None if not drug else np.array([0, 1]),
        stopping,
    )


def test_to_feature_matrix_combines_canonical_projections() -> None:
    matrix = _batch().to_feature_matrix()
    assert matrix.shape == (2, 4)
    np.testing.assert_allclose(matrix[0], [0.1, 0.2, 1.0, 0.0])


def test_to_feature_matrix_handles_one_side_and_featureless_result() -> None:
    assert _batch(drug=False).to_feature_matrix().shape == (2, 2)
    empty = FeaturizerResult(np.array(["cl1"]), {}, ())
    batch = ModelInputBatch(np.array(["cl1"]), np.array(["d1"]), np.array([1.0]), empty, None, np.array([0]), None)
    assert batch.to_feature_matrix().shape == (1, 0)


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


def test_subset_pairs_filters_stopping_by_surviving_drugs() -> None:
    stopping = ResponseBatch(np.array([0.5, 0.6, 0.7]), np.array(["cl1", "cl2", "cl1"]), np.array(["d1", "d2", "d3"]))
    batch = _batch(stopping=stopping)
    subset = batch.subset_pairs(np.array([True, False]))
    assert subset.n_pairs == 1
    assert subset.early_stopping_response is not None
    assert subset.early_stopping_response.drug_ids.tolist() == ["d1"]


def test_pair_index_helpers_map_and_report_missing_ids() -> None:
    assert pair_cell_line_indices(np.array(["cl2", "cl1"]), {"cl1": 0, "cl2": 1}).tolist() == [1, 0]
    with pytest.raises(ValueError, match="Missing drug identifiers"):
        pair_drug_indices(np.array(["missing"]), {"d1": 0})


def test_batch_rejects_pair_indices_that_do_not_match_pair_ids() -> None:
    cell = _result(["cl1", "cl2"], [[1.0], [2.0]])
    with pytest.raises(ValueError, match="cell_line pair identifiers"):
        ModelInputBatch(
            np.array(["cl1"]),
            np.array(["d1"]),
            np.array([1.0]),
            cell,
            None,
            np.array([1]),
            None,
        )


def test_target_free_numeric_batch_projects_requested_pairs() -> None:
    cell = _result(["cl1", "cl2"], [[1.0], [2.0]])
    batch = ModelInputBatch(
        np.array(["cl2", "cl1"]),
        np.array(["d1", "d1"]),
        None,
        cell,
        None,
        np.array([1, 0]),
        None,
    )
    np.testing.assert_allclose(batch.to_feature_matrix(), [[2.0], [1.0]])
