"""Tests for the leave-cell-line-out splitter."""

from __future__ import annotations

import numpy as np

from drevalpy.registry.splitter import splitter_registry
from tests.data.splitters._helpers import MockMuDataset


class TestLeaveCellLineOut:
    def test_produces_folds(self, mock_dataset: MockMuDataset) -> None:
        splitter = splitter_registry.get("LCO")
        folds = splitter(mock_dataset, n_splits=3)
        assert len(folds) == 3

    def test_no_cell_line_in_both_train_and_test(self, mock_dataset: MockMuDataset) -> None:
        splitter = splitter_registry.get("LCO")
        folds = splitter(mock_dataset, n_splits=3)
        for fold in folds:
            train_rows = set(np.where(fold.train.mask.any(axis=1))[0].tolist())
            test_rows = set(np.where(fold.test.mask.any(axis=1))[0].tolist())
            assert train_rows & test_rows == set()

    def test_all_indices_within_bounds(self, mock_dataset: MockMuDataset) -> None:
        splitter = splitter_registry.get("LCO")
        folds = splitter(mock_dataset, n_splits=3)
        shape = mock_dataset.response_matrix.shape
        for fold in folds:
            assert fold.train.shape == shape
            assert fold.test.shape == shape
            assert fold.val.shape == shape
