"""Tests for ModelInputBatch construction and matrix views."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.types.data.batch.model_input_batch import (
    ModelInputBatch,
    pair_cell_line_indices,
    pair_drug_indices,
)
from drevalpy.types.data.batch.model_input_build import build_model_input_batch
from drevalpy.types.data.batch.response_batch import ResponseBatch


def test_build_model_input_batch_indexes_entities() -> None:
    response = ResponseBatch(
        response=np.array([1.0, 2.0]),
        cell_line_ids=np.array(["cl1", "cl2"]),
        drug_ids=np.array(["d1", "d2"]),
    )
    early_stopping = ResponseBatch(
        response=np.array([1.0, 2.0]),
        cell_line_ids=np.array(["cl1", "cl2"]),
        drug_ids=np.array(["d1", "d2"]),
    )
    batch = build_model_input_batch(
        response,
        cell_line_entity_ids=np.array(["cl1", "cl2"]),
        drug_entity_ids=np.array(["d1", "d2"]),
        cell_line_features=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        drug_features=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        early_stopping_response=early_stopping,
    )
    assert batch.cell_line_pair_idx.tolist() == [0, 1]
    assert batch.drug_pair_idx is not None
    assert batch.drug_pair_idx.tolist() == [0, 1]
    assert batch.early_stopping_response is early_stopping


def test_to_feature_matrix_combined_cell_line_and_drug() -> None:
    batch = ModelInputBatch(
        cell_line_ids=np.array(["cl1", "cl2"]),
        drug_ids=np.array(["d1", "d2"]),
        response=np.array([1.0, 2.0]),
        cell_line_entity_ids=np.array(["cl1", "cl2"]),
        drug_entity_ids=np.array(["d1", "d2"]),
        cell_line_features=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        drug_features=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        cell_line_pair_idx=np.array([0, 1]),
        drug_pair_idx=np.array([0, 1]),
    )
    matrix = batch.to_feature_matrix()
    assert matrix.shape == (2, 4)
    np.testing.assert_allclose(matrix[0], np.array([0.1, 0.2, 1.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(matrix[1], np.array([0.3, 0.4, 0.0, 1.0], dtype=np.float32))


def test_to_feature_matrix_cell_line_only() -> None:
    batch = ModelInputBatch(
        cell_line_ids=np.array(["cl1", "cl2"]),
        drug_ids=np.array(["d1", "d2"]),
        response=np.array([1.0, 2.0]),
        cell_line_entity_ids=np.array(["cl1", "cl2"]),
        drug_entity_ids=None,
        cell_line_features=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        drug_features=None,
        cell_line_pair_idx=np.array([0, 1]),
        drug_pair_idx=None,
    )
    matrix = batch.to_feature_matrix()
    assert matrix.shape == (2, 2)


def test_to_feature_matrix_drug_only() -> None:
    batch = ModelInputBatch(
        cell_line_ids=np.array(["cl1", "cl2"]),
        drug_ids=np.array(["d1", "d2"]),
        response=np.array([1.0, 2.0]),
        cell_line_entity_ids=np.array([]),
        drug_entity_ids=np.array(["d1", "d2"]),
        cell_line_features=np.empty((0, 0), dtype=np.float32),
        drug_features=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        cell_line_pair_idx=np.array([0, 1]),
        drug_pair_idx=np.array([0, 1]),
    )
    matrix = batch.to_feature_matrix()
    assert matrix.shape == (2, 2)


def test_to_feature_matrix_empty_baseline() -> None:
    batch = ModelInputBatch(
        cell_line_ids=np.array(["cl1", "cl2"]),
        drug_ids=np.array(["d1", "d2"]),
        response=np.array([1.0, 2.0]),
        cell_line_entity_ids=np.array([]),
        drug_entity_ids=None,
        cell_line_features=np.empty((0, 0), dtype=np.float32),
        drug_features=None,
        cell_line_pair_idx=np.array([0, 0]),
        drug_pair_idx=None,
    )
    matrix = batch.to_feature_matrix()
    assert matrix.shape == (2, 0)


def test_build_model_input_batch_rejects_mismatched_entity_rows() -> None:
    response = ResponseBatch(
        response=np.array([1.0]),
        cell_line_ids=np.array(["cl1"]),
        drug_ids=np.array(["d1"]),
    )
    with pytest.raises(ValueError, match="cell_line_entity_ids length"):
        build_model_input_batch(
            response,
            cell_line_entity_ids=np.array(["cl1"]),
            drug_entity_ids=np.array(["d1"]),
            cell_line_features=np.array([[0.1], [0.2]], dtype=np.float32),
            drug_features=np.array([[1.0]], dtype=np.float32),
        )


def test_build_model_input_batch_rejects_missing_pair_ids() -> None:
    response = ResponseBatch(
        response=np.array([1.0]),
        cell_line_ids=np.array(["missing"]),
        drug_ids=np.array(["d1"]),
    )
    with pytest.raises(ValueError, match="Missing cell-line identifiers"):
        build_model_input_batch(
            response,
            cell_line_entity_ids=np.array(["cl1"]),
            drug_entity_ids=np.array(["d1"]),
            cell_line_features=np.array([[0.1]], dtype=np.float32),
            drug_features=np.array([[1.0]], dtype=np.float32),
        )


def test_to_feature_matrix_builds_drug_indices_from_entity_maps() -> None:
    batch = ModelInputBatch(
        cell_line_ids=np.array(["cl1"]),
        drug_ids=np.array(["d1"]),
        response=np.array([1.0]),
        cell_line_entity_ids=np.array(["cl1"]),
        drug_entity_ids=np.array(["d1"]),
        cell_line_features=np.array([[0.1]], dtype=np.float32),
        drug_features=np.array([[1.0]], dtype=np.float32),
        cell_line_pair_idx=np.array([0]),
        drug_pair_idx=None,
    )
    matrix = batch.to_feature_matrix()
    assert matrix.shape == (1, 2)
    np.testing.assert_allclose(matrix, np.array([[0.1, 1.0]], dtype=np.float32))


def test_subset_pairs_filters_pairs_and_early_stopping_by_drug() -> None:
    response = ResponseBatch(
        response=np.array([1.0, 2.0, 3.0, 4.0]),
        cell_line_ids=np.array(["cl1", "cl2", "cl1", "cl2"]),
        drug_ids=np.array(["d1", "d1", "d2", "d2"]),
    )
    early_stopping = ResponseBatch(
        response=np.array([0.5, 0.6, 0.7]),
        cell_line_ids=np.array(["cl1", "cl2", "cl1"]),
        drug_ids=np.array(["d1", "d1", "d2"]),
    )
    batch = ModelInputBatch(
        cell_line_ids=response.cell_line_ids,
        drug_ids=response.drug_ids,
        response=np.asarray(response.response, dtype=np.float64),
        cell_line_entity_ids=np.array(["cl1", "cl2"]),
        drug_entity_ids=np.array(["d1", "d2"]),
        cell_line_features=np.array([[0.1], [0.2]], dtype=np.float32),
        drug_features=None,
        cell_line_pair_idx=np.array([0, 1, 0, 1]),
        drug_pair_idx=np.array([0, 0, 1, 1]),
        early_stopping_response=early_stopping,
    )
    subset = batch.subset_pairs(np.array([True, True, False, False]))
    assert subset.n_pairs == 2
    assert subset.drug_ids.tolist() == ["d1", "d1"]
    assert subset.early_stopping_response is not None
    assert subset.early_stopping_response.drug_ids.tolist() == ["d1", "d1"]


def test_subset_pairs_keeps_early_stopping_for_every_surviving_drug() -> None:
    """A multi-drug subset keeps validation pairs for each surviving drug.

    ``PredictorBase.fit`` filters NaN pairs across the whole batch, so the mask it
    passes routinely spans several drugs. Narrowing early stopping to one drug -- or
    rejecting the mask outright -- would break every early-stopping predictor.
    """
    response = ResponseBatch(
        response=np.array([1.0, 2.0, 3.0, 4.0]),
        cell_line_ids=np.array(["cl1", "cl2", "cl1", "cl2"]),
        drug_ids=np.array(["d1", "d1", "d2", "d2"]),
    )
    early_stopping = ResponseBatch(
        response=np.array([0.5, 0.6, 0.7]),
        cell_line_ids=np.array(["cl1", "cl2", "cl1"]),
        drug_ids=np.array(["d1", "d2", "d3"]),
    )
    batch = ModelInputBatch(
        cell_line_ids=response.cell_line_ids,
        drug_ids=response.drug_ids,
        response=np.asarray(response.response, dtype=np.float64),
        cell_line_entity_ids=np.array(["cl1", "cl2"]),
        drug_entity_ids=np.array(["d1", "d2"]),
        cell_line_features=np.array([[0.1], [0.2]], dtype=np.float32),
        drug_features=None,
        cell_line_pair_idx=np.array([0, 1, 0, 1]),
        drug_pair_idx=np.array([0, 0, 1, 1]),
        early_stopping_response=early_stopping,
    )
    # Drops one pair per drug, so the surviving mask still spans d1 and d2.
    subset = batch.subset_pairs(np.array([True, False, True, False]))
    assert subset.n_pairs == 2
    assert subset.drug_ids.tolist() == ["d1", "d2"]
    assert subset.early_stopping_response is not None
    # d1 and d2 are retained; d3 has no surviving response pair and is dropped.
    assert subset.early_stopping_response.drug_ids.tolist() == ["d1", "d2"]


def test_pair_cell_line_indices_maps_ids() -> None:
    indices = pair_cell_line_indices(
        np.array(["cl2", "cl1", "cl2"]),
        {"cl1": 0, "cl2": 1},
    )
    assert indices.tolist() == [1, 0, 1]


def test_pair_drug_indices_raises_for_missing_ids() -> None:
    with pytest.raises(ValueError, match="Missing drug identifiers"):
        pair_drug_indices(np.array(["d1", "missing"]), {"d1": 0})
