"""Tests for ModelInputBatch structural validation."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.model_input_batch import ModelInputBatch


def _ids(n: int) -> np.ndarray:
    return np.array([f"id_{i}" for i in range(n)])


def test_batch_rejects_response_length_mismatch() -> None:
    with pytest.raises(ValueError, match="response length"):
        ModelInputBatch(
            cell_line_ids=_ids(3),
            drug_ids=_ids(3),
            response=np.ones(2),
            cell_line_entity_ids=np.empty(0),
            drug_entity_ids=None,
            cell_line_features=np.empty((0, 0)),
            drug_features=None,
            cell_line_pair_idx=np.zeros(3, dtype=np.int64),
            drug_pair_idx=None,
        )


def test_batch_rejects_pair_idx_length_mismatch() -> None:
    with pytest.raises(ValueError, match="cell_line_pair_idx length"):
        ModelInputBatch(
            cell_line_ids=_ids(3),
            drug_ids=_ids(3),
            response=np.ones(3),
            cell_line_entity_ids=np.empty(0),
            drug_entity_ids=None,
            cell_line_features=np.empty((0, 0)),
            drug_features=None,
            cell_line_pair_idx=np.zeros(2, dtype=np.int64),
            drug_pair_idx=None,
        )
