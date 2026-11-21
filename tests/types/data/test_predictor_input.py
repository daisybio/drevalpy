"""Model inputs: construction from response pairs, the design matrix, and pair subsetting."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from drevalpy.types.data.feature_block import FeatureBlock, FeaturizerOutput
from drevalpy.types.data.predictor_input import PredictorInput
from drevalpy.types.data.response_table import ResponseTable


def _response(sample=("cl1", "cl2"), drug=("d1", "d2")) -> ResponseTable:
    return ResponseTable(np.arange(len(sample), dtype=float), np.asarray(sample), np.asarray(drug))


def _entity_result(ids=("cl1", "cl2"), values=None) -> FeaturizerOutput:
    matrix = np.arange(len(ids) * 2, dtype=np.float32).reshape(len(ids), 2) if values is None else values
    return FeaturizerOutput(np.asarray(ids), {"view": FeatureBlock(matrix)})


def _model_input(*, drug=True, stopping=None) -> PredictorInput:
    sample_result = FeaturizerOutput(
        np.array(["cl1", "cl2"]), {"view": FeatureBlock(np.array([[0.1, 0.2], [0.3, 0.4]]))}
    )
    drug_result = (
        FeaturizerOutput(np.array(["d1", "d2"]), {"view": FeatureBlock(np.array([[1.0, 0.0], [0.0, 1.0]]))})
        if drug
        else None
    )
    return PredictorInput(
        ResponseTable(np.array([1.0, 2.0]), np.array(["cl1", "cl2"]), np.array(["d1", "d2"])),
        sample_result,
        drug_result,
        stopping,
    )


class TestConstruction:
    def test_indexes_entities_in_result_order(self) -> None:
        data = PredictorInput(
            _response(("cl2", "cl1", "cl2"), ("d2", "d1", "d2")),
            sample=_entity_result(),
            drug=_entity_result(("d1", "d2")),
        )
        assert data.sample_pair_idx.tolist() == [1, 0, 1]
        assert data.drug_pair_idx.tolist() == [1, 0, 1]
        assert data.sample.entity_ids.tolist() == ["cl1", "cl2"]

    def test_keeps_the_response_pairs_and_targets(self) -> None:
        response = _response(("cl2", "cl1"), ("d1", "d1"))
        data = PredictorInput(response, sample=_entity_result(), drug=_entity_result(("d1", "d2")))
        assert data.sample_ids.tolist() == ["cl2", "cl1"]
        assert data.drug_ids.tolist() == ["d1", "d1"]
        np.testing.assert_array_equal(data.response, response.response)

    def test_drug_is_optional(self) -> None:
        data = PredictorInput(_response(), sample=_entity_result())
        assert data.drug is None
        assert data.drug_pair_idx is None

    def test_featureless_entity_table_retains_and_indexes_entity_ids(self) -> None:
        featureless = FeaturizerOutput(np.array(["cl1", "cl2"]), {})
        data = PredictorInput(_response(), sample=featureless)
        assert data.sample_pair_idx.tolist() == [0, 1]

    def test_rejects_duplicate_entity_ids(self) -> None:
        duplicate = FeaturizerOutput(np.array(["cl1", "cl1"]), {})
        with pytest.raises(ValueError, match="unique"):
            PredictorInput(_response(), sample=duplicate)

    def test_zero_pairs_yield_empty_indices(self) -> None:
        empty = FeaturizerOutput(np.array([], dtype=str), {})
        data = PredictorInput(_response((), ()), sample=empty, drug=empty)
        assert data.sample_pair_idx.size == 0
        assert data.drug_pair_idx is not None and data.drug_pair_idx.size == 0

    def test_rejects_missing_pair_ids(self) -> None:
        with pytest.raises(ValueError, match="Missing identifiers"):
            PredictorInput(_response(("missing",), ("d1",)), sample=_entity_result(), drug=_entity_result(("d1", "d2")))
        with pytest.raises(ValueError, match="Missing identifiers"):
            PredictorInput(
                _response(("cl1",), ("missing",)), sample=_entity_result(), drug=_entity_result(("d1", "d2"))
            )

    def test_is_immutable(self) -> None:
        data = _model_input()
        with pytest.raises(FrozenInstanceError):
            data.early_stopping = None  # type: ignore[misc]


class TestToFeatureMatrix:
    def test_combines_side_matrices(self) -> None:
        matrix = _model_input().to_feature_matrix()
        assert matrix.shape == (2, 4)
        np.testing.assert_allclose(matrix[0], [0.1, 0.2, 1.0, 0.0])

    def test_handles_one_side(self) -> None:
        assert _model_input(drug=False).to_feature_matrix().shape == (2, 2)

    def test_featureless_result_has_no_columns(self) -> None:
        empty = FeaturizerOutput(np.array(["cl1"]), {})
        data = PredictorInput(_response(("cl1",), ("d1",)), empty)
        assert data.to_feature_matrix().shape == (1, 0)

    def test_target_free_input_projects_requested_pairs(self) -> None:
        data = PredictorInput(
            ResponseTable(np.full(2, np.nan), np.array(["cl2", "cl1"]), np.array(["d1", "d1"])),
            _entity_result(values=np.array([[1.0], [2.0]])),
        )
        np.testing.assert_allclose(data.to_feature_matrix(), [[2.0], [1.0]])

    def test_rejects_a_non_numeric_block(self) -> None:
        text = FeaturizerOutput(np.array(["cl1"]), {"view": FeatureBlock(np.array([["a"]]))})
        data = PredictorInput(_response(("cl1",), ("d1",)), text)
        with pytest.raises(ValueError, match="two-dimensional numeric"):
            data.to_feature_matrix()


class TestIndexing:
    def test_filters_stopping_by_surviving_drugs(self) -> None:
        stopping = ResponseTable(
            np.array([0.5, 0.6, 0.7]), np.array(["cl1", "cl2", "cl1"]), np.array(["d1", "d2", "d3"])
        )
        subset = _model_input(stopping=stopping)[np.array([True, False])]
        assert len(subset) == 1
        assert subset.early_stopping is not None
        assert subset.early_stopping.drug_ids.tolist() == ["d1"]

    def test_drops_stopping_when_no_drug_survives_or_mask_is_empty(self) -> None:
        stopping = ResponseTable(np.array([0.5]), np.array(["cl1"]), np.array(["d3"]))
        assert _model_input(stopping=stopping)[np.array([True, False])].early_stopping is None

        matching_stopping = ResponseTable(np.array([0.5]), np.array(["cl1"]), np.array(["d1"]))
        empty = _model_input(stopping=matching_stopping)[np.array([False, False])]
        assert len(empty) == 0
        assert empty.early_stopping is None

    def test_supports_slices_and_integer_indices(self) -> None:
        data = _model_input()
        assert data[:1].sample_ids.tolist() == ["cl1"]
        assert data[np.array([1, 0])].sample_ids.tolist() == ["cl2", "cl1"]

    def test_rejects_a_mask_of_the_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="matching len"):
            _ = _model_input()[np.array([True])]


class TestResponseLayers:
    def test_carries_the_response_layers(self) -> None:
        response = ResponseTable(
            np.array([1.0, 2.0]),
            np.array(["cl1", "cl2"]),
            np.array(["d1", "d2"]),
            {"regulation": np.array([-1.0, 0.0])},
        )

        data = PredictorInput(response, sample=_entity_result())

        assert data.response_layers["regulation"].tolist() == [-1.0, 0.0]

    def test_an_input_has_no_response_layers_by_default(self) -> None:
        assert dict(_model_input().response_layers) == {}

    def test_indexing_keeps_the_response_layers_aligned(self) -> None:
        data = PredictorInput(
            ResponseTable(
                np.array([1.0, 2.0]),
                np.array(["cl1", "cl2"]),
                np.array(["d1", "d2"]),
                {"regulation": np.array([-1.0, 1.0])},
            ),
            sample=_entity_result(),
        )

        assert data[np.array([False, True])].response_layers["regulation"].tolist() == [1.0]
