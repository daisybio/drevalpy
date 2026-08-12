"""Tests for the leave-tissue-out splitter."""

from __future__ import annotations

import numpy as np

from drevalpy.registry.splitter import splitter_registry
from tests.data.splitters._helpers import MockMuDataset


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
