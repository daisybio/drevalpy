"""Tests for ModelInputBatch construction and matrix views."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.model_input_build import build_model_input_batch
from drevalpy.datasets.dataset import DrugResponseDataset


def test_build_model_input_batch_indexes_entities() -> None:
    response = DrugResponseDataset(
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
    )
    assert batch.cell_line_pair_idx.tolist() == [0, 1]
    assert batch.drug_pair_idx is not None
    assert batch.drug_pair_idx.tolist() == [0, 1]


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


def test_to_feature_matrix_requires_drug_pair_idx() -> None:
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
    with pytest.raises(ValueError, match="drug_pair_idx is required"):
        batch.to_feature_matrix()
