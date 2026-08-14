"""Tests for the leave-tissue-out splitter."""

from __future__ import annotations

import numpy as np

from drevalpy.registry.splitter import splitter_registry
from tests.data.splitters._helpers import MockMuDataset, covered_pairs, first_measured_pairs


class TestLeaveTissueOut:
    def test_produces_folds(self, mock_dataset: MockMuDataset) -> None:
        splitter = splitter_registry.get("LTO")
        folds = splitter(mock_dataset, n_splits=3)
        assert len(folds) == 3

    def test_no_tissue_in_both_train_and_test(self, mock_dataset: MockMuDataset) -> None:
        splitter = splitter_registry.get("LTO")
        folds = splitter(mock_dataset, n_splits=3)
        tissues = mock_dataset.get_tissue(mock_dataset.cell_line_ids)
        for fold in folds:
            train_rows = np.where(fold.train.mask.any(axis=1))[0]
            test_rows = np.where(fold.test.mask.any(axis=1))[0]
            train_tissues = set(tissues[train_rows].tolist())
            test_tissues = set(tissues[test_rows].tolist())
            assert train_tissues & test_tissues == set()

    def test_low_quality_pairs_appear_in_no_fold(self, mock_dataset: MockMuDataset) -> None:
        """Measured pairs whose curve fails the thresholds are never split into."""
        failing = first_measured_pairs(mock_dataset, 3)
        splitter = splitter_registry.get("LTO")

        without_filtering = covered_pairs(splitter(mock_dataset, n_splits=3))
        with_filtering = covered_pairs(splitter(MockMuDataset(failing_pairs=failing), n_splits=3))

        for row, column in failing:
            assert without_filtering[row, column]
            assert not with_filtering[row, column]
