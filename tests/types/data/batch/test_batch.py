"""Model input batches: feature blocks, pair indexing and batch building, and the response triple."""

from __future__ import annotations

import dataclasses
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pytest

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.types.data.batch.feature_block import (
    FeatureBlock,
    FeaturizerResult,
    graph_feature_block,
    merge_feature_blocks,
    metadata_feature_block,
    numeric_feature_block,
    ragged_feature_block,
)
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch, pair_drug_indices, pair_sample_indices
from drevalpy.types.data.batch.model_input_build import build_model_input_batch
from drevalpy.types.data.batch.response_batch import ResponseBatch


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
    assert pair_sample_indices(np.array(["cl2", "cl1"]), {"cl1": 0, "cl2": 1}).tolist() == [1, 0]
    with pytest.raises(ValueError, match="Missing drug identifiers"):
        pair_drug_indices(np.array(["missing"]), {"d1": 0})


def test_batch_rejects_pair_indices_that_do_not_match_pair_ids() -> None:
    cell = _result(["cl1", "cl2"], [[1.0], [2.0]])
    with pytest.raises(ValueError, match="sample pair identifiers"):
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


def _response(cell=("cl1", "cl2"), drug=("d1", "d2")) -> ResponseBatch:
    return ResponseBatch(np.arange(len(cell), dtype=float), np.asarray(cell), np.asarray(drug))


def _entity_result(ids=("cl1", "cl2"), values=None) -> FeaturizerResult:
    matrix = np.arange(len(ids) * 2, dtype=np.float32).reshape(len(ids), 2) if values is None else values
    return FeaturizerResult.from_blocks({"view": numeric_feature_block(matrix)}, np.asarray(ids))


def test_build_indexes_entities_in_result_order() -> None:
    batch = build_model_input_batch(
        _response(("cl2", "cl1", "cl2"), ("d2", "d1", "d2")),
        sample=_entity_result(),
        drug=_entity_result(("d1", "d2")),
    )
    assert batch.sample_pair_idx.tolist() == [1, 0, 1]
    assert batch.drug_pair_idx.tolist() == [1, 0, 1]
    assert batch.sample.entity_ids.tolist() == ["cl1", "cl2"]


def test_drug_is_optional() -> None:
    batch = build_model_input_batch(_response(), sample=_entity_result())
    assert batch.drug is None
    assert batch.drug_pair_idx is None


def test_featureless_entity_table_retains_and_indexes_entity_ids() -> None:
    featureless = FeaturizerResult(np.array(["cl1", "cl2"]), {}, ())
    batch = build_model_input_batch(_response(), sample=featureless)
    assert batch.sample_pair_idx.tolist() == [0, 1]


def test_build_rejects_duplicate_entity_ids() -> None:
    duplicate = FeaturizerResult(np.array(["cl1", "cl1"]), {}, ())
    with pytest.raises(ValueError, match="unique"):
        build_model_input_batch(_response(), sample=duplicate)


def test_zero_pairs_yield_empty_indices() -> None:
    empty = FeaturizerResult(np.array([], dtype=str), {}, ())
    batch = build_model_input_batch(_response((), ()), sample=empty, drug=empty)
    assert batch.sample_pair_idx.size == 0
    assert batch.drug_pair_idx is not None and batch.drug_pair_idx.size == 0


def test_result_rejects_mismatched_rows_and_scalars() -> None:
    with pytest.raises(ValueError, match="rows must match entity_ids"):
        _entity_result(("cl1",), np.ones((2, 1)))
    with pytest.raises(ValueError, match="rows must match entity_ids"):
        _entity_result(("cl1",), np.array(1.0))


def test_build_rejects_missing_pair_ids() -> None:
    with pytest.raises(ValueError, match="Missing sample identifiers"):
        build_model_input_batch(
            _response(("missing",), ("d1",)), sample=_entity_result(), drug=_entity_result(("d1", "d2"))
        )
    with pytest.raises(ValueError, match="Missing drug identifiers"):
        build_model_input_batch(
            _response(("cl1",), ("missing",)), sample=_entity_result(), drug=_entity_result(("d1", "d2"))
        )


@pytest.fixture()
def batch() -> ResponseBatch:
    """Two measured pairs across two samples and two drugs."""
    return ResponseBatch(
        response=np.array([1.0, 2.0]),
        sample_ids=np.array(["cl1", "cl2"]),
        drug_ids=np.array(["d1", "d2"]),
    )


class TestConstruction:
    def test_fields_are_stored_verbatim(self, batch: ResponseBatch):
        np.testing.assert_array_equal(batch.response, np.array([1.0, 2.0]))
        assert batch.sample_ids.tolist() == ["cl1", "cl2"]
        assert batch.drug_ids.tolist() == ["d1", "d2"]

    def test_arrays_are_not_copied(self):
        response = np.array([1.0])

        batch = ResponseBatch(response=response, sample_ids=np.array(["cl1"]), drug_ids=np.array(["d1"]))

        assert batch.response is response

    def test_keyword_construction_requires_all_three_fields(self):
        with pytest.raises(TypeError):
            ResponseBatch(response=np.array([1.0]))  # type: ignore[call-arg]


class TestLength:
    def test_len_counts_response_pairs(self, batch: ResponseBatch):
        assert len(batch) == 2

    def test_an_empty_batch_has_length_zero(self):
        empty = ResponseBatch(
            response=np.array([]),
            sample_ids=np.array([]),
            drug_ids=np.array([]),
        )

        assert len(empty) == 0

    def test_nan_responses_still_count_as_pairs(self):
        """NaN marks an unmeasured pair; filtering happens in the predictors, not here."""
        with_nan = ResponseBatch(
            response=np.array([1.0, np.nan]),
            sample_ids=np.array(["cl1", "cl2"]),
            drug_ids=np.array(["d1", "d2"]),
        )

        assert len(with_nan) == 2


class TestImmutability:
    def test_fields_cannot_be_reassigned(self, batch: ResponseBatch):
        with pytest.raises(dataclasses.FrozenInstanceError):
            batch.response = np.array([9.0])  # type: ignore[misc]

    def test_slots_replace_the_instance_dict(self, batch: ResponseBatch):
        """``slots=True`` keeps these per-pair containers cheap to allocate."""
        assert ResponseBatch.__slots__ == ("response", "sample_ids", "drug_ids")
        assert not hasattr(batch, "__dict__")
