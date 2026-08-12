"""Tests for the leave-drug-out splitter."""

from __future__ import annotations

import numpy as np

from drevalpy.registry.splitter import splitter_registry
from tests.data.splitters._helpers import MockMuDataset


class TestLeaveDrugOut:
    def test_produces_folds(self, mock_dataset: MockMuDataset) -> None:
        splitter = splitter_registry.get("LDO")
        folds = splitter(mock_dataset, n_splits=3)
        assert len(folds) == 3

    def test_no_drug_in_both_train_and_test(self, mock_dataset: MockMuDataset) -> None:
        splitter = splitter_registry.get("LDO")
        folds = splitter(mock_dataset, n_splits=3)
        for fold in folds:
            train_cols = set(np.where(fold.train.mask.any(axis=0))[0].tolist())
            test_cols = set(np.where(fold.test.mask.any(axis=0))[0].tolist())
            assert train_cols & test_cols == set()
