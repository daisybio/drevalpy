"""Model input batches: construction from a response, the design matrix, and pair subsetting."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.types.data.batch.feature_block import FeaturizerResult, numeric_feature_block
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.data.batch.response_batch import ResponseBatch


def _response(sample=("cl1", "cl2"), drug=("d1", "d2")) -> ResponseBatch:
    return ResponseBatch(np.arange(len(sample), dtype=float), np.asarray(sample), np.asarray(drug))


def _entity_result(ids=("cl1", "cl2"), values=None) -> FeaturizerResult:
    matrix = np.arange(len(ids) * 2, dtype=np.float32).reshape(len(ids), 2) if values is None else values
    return FeaturizerResult.from_blocks({"view": numeric_feature_block(matrix)}, np.asarray(ids))


def _batch(*, drug=True, stopping=None) -> ModelInputBatch:
    sample_result = FeaturizerResult(
        np.array(["cl1", "cl2"]), {"view": numeric_feature_block(np.array([[0.1, 0.2], [0.3, 0.4]]))}, ("view",)
    )
    drug_result = (
        FeaturizerResult(
            np.array(["d1", "d2"]), {"view": numeric_feature_block(np.array([[1.0, 0.0], [0.0, 1.0]]))}, ("view",)
        )
        if drug
        else None
    )
    return ModelInputBatch(
        np.array(["cl1", "cl2"]),
        np.array(["d1", "d2"]),
        np.array([1.0, 2.0]),
        sample_result,
        drug_result,
        np.array([0, 1]),
        np.array([0, 1]) if drug else None,
        stopping,
    )


class TestFromResponse:
    def test_indexes_entities_in_result_order(self) -> None:
        batch = ModelInputBatch.from_response(
            _response(("cl2", "cl1", "cl2"), ("d2", "d1", "d2")),
            sample=_entity_result(),
            drug=_entity_result(("d1", "d2")),
        )
        assert batch.sample_pair_idx.tolist() == [1, 0, 1]
        assert batch.drug_pair_idx.tolist() == [1, 0, 1]
        assert batch.sample.entity_ids.tolist() == ["cl1", "cl2"]

    def test_keeps_the_response_pairs_and_targets(self) -> None:
        response = _response(("cl2", "cl1"), ("d1", "d1"))
        batch = ModelInputBatch.from_response(response, sample=_entity_result(), drug=_entity_result(("d1", "d2")))
        assert batch.sample_ids.tolist() == ["cl2", "cl1"]
        assert batch.drug_ids.tolist() == ["d1", "d1"]
        np.testing.assert_array_equal(batch.response, response.response)

    def test_drug_is_optional(self) -> None:
        batch = ModelInputBatch.from_response(_response(), sample=_entity_result())
        assert batch.drug is None
        assert batch.drug_pair_idx is None

    def test_featureless_entity_table_retains_and_indexes_entity_ids(self) -> None:
        featureless = FeaturizerResult(np.array(["cl1", "cl2"]), {}, ())
        batch = ModelInputBatch.from_response(_response(), sample=featureless)
        assert batch.sample_pair_idx.tolist() == [0, 1]

    def test_rejects_duplicate_entity_ids(self) -> None:
        duplicate = FeaturizerResult(np.array(["cl1", "cl1"]), {}, ())
        with pytest.raises(ValueError, match="unique"):
            ModelInputBatch.from_response(_response(), sample=duplicate)

    def test_zero_pairs_yield_empty_indices(self) -> None:
        empty = FeaturizerResult(np.array([], dtype=str), {}, ())
        batch = ModelInputBatch.from_response(_response((), ()), sample=empty, drug=empty)
        assert batch.sample_pair_idx.size == 0
        assert batch.drug_pair_idx is not None and batch.drug_pair_idx.size == 0

    def test_rejects_missing_pair_ids(self) -> None:
        with pytest.raises(ValueError, match="Missing identifiers"):
            ModelInputBatch.from_response(
                _response(("missing",), ("d1",)), sample=_entity_result(), drug=_entity_result(("d1", "d2"))
            )
        with pytest.raises(ValueError, match="Missing identifiers"):
            ModelInputBatch.from_response(
                _response(("cl1",), ("missing",)), sample=_entity_result(), drug=_entity_result(("d1", "d2"))
            )


class TestConstruction:
    def test_rejects_pair_indices_that_do_not_match_pair_ids(self) -> None:
        with pytest.raises(ValueError, match="sample pair identifiers"):
            ModelInputBatch(
                np.array(["cl1"]), np.array(["d1"]), np.array([1.0]), _entity_result(), None, np.array([1]), None
            )

    def test_rejects_indices_outside_the_entity_table(self) -> None:
        with pytest.raises(ValueError, match="outside the entity table"):
            ModelInputBatch(
                np.array(["cl1"]), np.array(["d1"]), np.array([1.0]), _entity_result(), None, np.array([5]), None
            )

    def test_rejects_non_integer_indices(self) -> None:
        with pytest.raises(ValueError, match="must be integers"):
            ModelInputBatch(
                np.array(["cl1"]), np.array(["d1"]), np.array([1.0]), _entity_result(), None, np.array([0.0]), None
            )

    def test_a_drug_result_requires_pair_indices(self) -> None:
        with pytest.raises(ValueError, match="drug feature result requires pair indices"):
            ModelInputBatch(
                np.array(["cl1"]),
                np.array(["d1"]),
                np.array([1.0]),
                _entity_result(),
                _entity_result(("d1",)),
                np.array([0]),
                None,
            )


class TestToFeatureMatrix:
    def test_combines_canonical_projections(self) -> None:
        matrix = _batch().to_feature_matrix()
        assert matrix.shape == (2, 4)
        np.testing.assert_allclose(matrix[0], [0.1, 0.2, 1.0, 0.0])

    def test_handles_one_side(self) -> None:
        assert _batch(drug=False).to_feature_matrix().shape == (2, 2)

    def test_featureless_result_has_no_columns(self) -> None:
        empty = FeaturizerResult(np.array(["cl1"]), {}, ())
        batch = ModelInputBatch(np.array(["cl1"]), np.array(["d1"]), np.array([1.0]), empty, None, np.array([0]), None)
        assert batch.to_feature_matrix().shape == (1, 0)

    def test_target_free_batch_projects_requested_pairs(self) -> None:
        batch = ModelInputBatch(
            np.array(["cl2", "cl1"]),
            np.array(["d1", "d1"]),
            None,
            _entity_result(values=np.array([[1.0], [2.0]])),
            None,
            np.array([1, 0]),
            None,
        )
        np.testing.assert_allclose(batch.to_feature_matrix(), [[2.0], [1.0]])

    def test_rejects_a_non_numeric_projection(self) -> None:
        text = FeaturizerResult(np.array(["cl1"]), {"view": numeric_feature_block(np.array([["a"]]))}, ("view",))
        batch = ModelInputBatch(np.array(["cl1"]), np.array(["d1"]), None, text, None, np.array([0]), None)
        with pytest.raises(ValueError, match="two-dimensional numeric"):
            batch.to_feature_matrix()


class TestSubsetPairs:
    def test_filters_stopping_by_surviving_drugs(self) -> None:
        stopping = ResponseBatch(
            np.array([0.5, 0.6, 0.7]), np.array(["cl1", "cl2", "cl1"]), np.array(["d1", "d2", "d3"])
        )
        subset = _batch(stopping=stopping).subset_pairs(np.array([True, False]))
        assert subset.n_pairs == 1
        assert subset.early_stopping_response is not None
        assert subset.early_stopping_response.drug_ids.tolist() == ["d1"]

    def test_drops_stopping_when_no_drug_survives(self) -> None:
        stopping = ResponseBatch(np.array([0.5]), np.array(["cl1"]), np.array(["d3"]))
        assert _batch(stopping=stopping).subset_pairs(np.array([True, False])).early_stopping_response is None

    def test_rejects_a_mask_of_the_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="matching n_pairs"):
            _batch().subset_pairs(np.array([True]))
