"""Tests for :func:`build_model_input_batch`.

Carved out of ``test_model_input_batch.py``, which now covers only the
``ModelInputBatch`` dataclass and its pair-index helpers.
"""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.contracts.training_context import TrainingContext
from drevalpy.types.data.batch.model_input_build import build_model_input_batch
from drevalpy.types.data.batch.response_batch import ResponseBatch


def _pairs(n: int = 2) -> ResponseBatch:
    """Build *n* pairs named ``cl1..cln`` / ``d1..dn``."""
    return ResponseBatch(
        response=np.arange(1.0, n + 1.0),
        cell_line_ids=np.array([f"cl{i}" for i in range(1, n + 1)]),
        drug_ids=np.array([f"d{i}" for i in range(1, n + 1)]),
    )


class TestIndexing:
    def test_build_model_input_batch_indexes_entities(self) -> None:
        response = _pairs()
        early_stopping = _pairs()

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

    def test_pair_indices_follow_entity_row_order_not_pair_order(self) -> None:
        response = ResponseBatch(
            response=np.array([1.0, 2.0]),
            cell_line_ids=np.array(["cl2", "cl1"]),
            drug_ids=np.array(["d2", "d1"]),
        )

        batch = build_model_input_batch(
            response,
            cell_line_entity_ids=np.array(["cl1", "cl2"]),
            drug_entity_ids=np.array(["d1", "d2"]),
            cell_line_features=np.array([[0.1], [0.2]], dtype=np.float32),
            drug_features=np.array([[1.0], [2.0]], dtype=np.float32),
        )

        assert batch.cell_line_pair_idx.tolist() == [1, 0]
        assert batch.drug_pair_idx is not None
        assert batch.drug_pair_idx.tolist() == [1, 0]

    def test_drug_pair_idx_is_none_without_drug_entity_ids(self) -> None:
        batch = build_model_input_batch(
            _pairs(),
            cell_line_entity_ids=np.array(["cl1", "cl2"]),
            drug_entity_ids=None,
            cell_line_features=np.array([[0.1], [0.2]], dtype=np.float32),
            drug_features=None,
        )

        assert batch.drug_pair_idx is None

    def test_drug_pair_idx_is_none_without_drug_features(self) -> None:
        batch = build_model_input_batch(
            _pairs(),
            cell_line_entity_ids=np.array(["cl1", "cl2"]),
            drug_entity_ids=np.array(["d1", "d2"]),
            cell_line_features=np.array([[0.1], [0.2]], dtype=np.float32),
            drug_features=None,
        )

        assert batch.drug_pair_idx is None

    def test_blocks_and_training_context_are_forwarded(self) -> None:
        context = TrainingContext()

        batch = build_model_input_batch(
            _pairs(),
            cell_line_entity_ids=np.array(["cl1", "cl2"]),
            drug_entity_ids=None,
            cell_line_features=np.array([[0.1], [0.2]], dtype=np.float32),
            drug_features=None,
            training_context=context,
        )

        assert batch.training_context is context
        assert batch.cell_line_blocks == {}
        assert batch.drug_blocks == {}


class TestBaselineWithoutEntityFeatures:
    """Naive baselines pass empty entity ids, which must index to row zero."""

    def test_empty_cell_line_entity_ids_index_to_row_zero(self) -> None:
        batch = build_model_input_batch(
            _pairs(),
            cell_line_entity_ids=np.array([]),
            drug_entity_ids=None,
            cell_line_features=np.empty((0, 0), dtype=np.float32),
            drug_features=None,
        )

        assert batch.cell_line_pair_idx.tolist() == [0, 0]

    def test_empty_drug_entity_ids_index_to_row_zero(self) -> None:
        batch = build_model_input_batch(
            _pairs(),
            cell_line_entity_ids=np.array(["cl1", "cl2"]),
            drug_entity_ids=np.array([]),
            cell_line_features=np.array([[0.1], [0.2]], dtype=np.float32),
            drug_features=np.empty((0, 0), dtype=np.float32),
        )

        assert batch.drug_pair_idx is not None
        assert batch.drug_pair_idx.tolist() == [0, 0]

    def test_zero_pairs_yield_empty_index_arrays(self) -> None:
        response = ResponseBatch(
            response=np.array([]),
            cell_line_ids=np.array([]),
            drug_ids=np.array([]),
        )

        batch = build_model_input_batch(
            response,
            cell_line_entity_ids=np.array([]),
            drug_entity_ids=np.array([]),
            cell_line_features=np.empty((0, 0), dtype=np.float32),
            drug_features=np.empty((0, 0), dtype=np.float32),
        )

        assert batch.n_pairs == 0
        assert batch.cell_line_pair_idx.tolist() == []


class TestValidation:
    def test_build_model_input_batch_rejects_mismatched_entity_rows(self) -> None:
        response = _pairs(1)

        with pytest.raises(ValueError, match="cell_line_entity_ids length"):
            build_model_input_batch(
                response,
                cell_line_entity_ids=np.array(["cl1"]),
                drug_entity_ids=np.array(["d1"]),
                cell_line_features=np.array([[0.1], [0.2]], dtype=np.float32),
                drug_features=np.array([[1.0]], dtype=np.float32),
            )

    def test_build_model_input_batch_rejects_mismatched_drug_rows(self) -> None:
        with pytest.raises(ValueError, match="drug_entity_ids length"):
            build_model_input_batch(
                _pairs(1),
                cell_line_entity_ids=np.array(["cl1"]),
                drug_entity_ids=np.array(["d1"]),
                cell_line_features=np.array([[0.1]], dtype=np.float32),
                drug_features=np.array([[1.0], [2.0]], dtype=np.float32),
            )

    def test_zero_dimensional_features_are_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"cell_line_features rows \(0\)"):
            build_model_input_batch(
                _pairs(1),
                cell_line_entity_ids=np.array(["cl1"]),
                drug_entity_ids=None,
                cell_line_features=np.array(1.0, dtype=np.float32),
                drug_features=None,
            )

    def test_empty_cell_line_entity_ids_with_features_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="cell_line_entity_ids must be non-empty"):
            build_model_input_batch(
                _pairs(1),
                cell_line_entity_ids=np.array([]),
                drug_entity_ids=None,
                cell_line_features=np.array([[0.1]], dtype=np.float32),
                drug_features=None,
            )

    def test_empty_drug_entity_ids_with_features_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="drug_entity_ids must be non-empty"):
            build_model_input_batch(
                _pairs(1),
                cell_line_entity_ids=np.array(["cl1"]),
                drug_entity_ids=np.array([]),
                cell_line_features=np.array([[0.1]], dtype=np.float32),
                drug_features=np.array([[1.0]], dtype=np.float32),
            )

    def test_build_model_input_batch_rejects_missing_pair_ids(self) -> None:
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

    def test_build_model_input_batch_rejects_missing_drug_pair_ids(self) -> None:
        response = ResponseBatch(
            response=np.array([1.0]),
            cell_line_ids=np.array(["cl1"]),
            drug_ids=np.array(["missing"]),
        )

        with pytest.raises(ValueError, match="Missing drug identifiers"):
            build_model_input_batch(
                response,
                cell_line_entity_ids=np.array(["cl1"]),
                drug_entity_ids=np.array(["d1"]),
                cell_line_features=np.array([[0.1]], dtype=np.float32),
                drug_features=np.array([[1.0]], dtype=np.float32),
            )
