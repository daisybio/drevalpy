"""Tests for the leave-tissue-out splitter."""

from __future__ import annotations

import numpy as np

from drevalpy.data.quality import CurveQuality
from drevalpy.registry.splitter import splitter_registry
from tests.data.splitters._helpers import MockMuDataset, covered_pairs, first_measured_pairs

QUALITY = CurveQuality.default()


def test_no_tissue_in_both_train_and_test(mock_dataset: MockMuDataset) -> None:
    folds = splitter_registry.get("LTO")(mock_dataset, n_splits=3, quality=QUALITY)
    assert len(folds) == 3
    tissues = mock_dataset.get_tissue(mock_dataset.sample_ids)
    for fold in folds:
        train_rows = np.where(fold.train.mask.any(axis=1))[0]
        test_rows = np.where(fold.test.mask.any(axis=1))[0]
        assert set(tissues[train_rows].tolist()) & set(tissues[test_rows].tolist()) == set()


def test_low_quality_pairs_appear_in_no_fold(mock_dataset: MockMuDataset) -> None:
    failing = first_measured_pairs(mock_dataset, 3)
    splitter = splitter_registry.get("LTO")

    without_filtering = covered_pairs(splitter(mock_dataset, n_splits=3, quality=QUALITY))
    with_filtering = covered_pairs(splitter(MockMuDataset(failing_pairs=failing), n_splits=3, quality=QUALITY))

    for row, column in failing:
        assert without_filtering[row, column]
        assert not with_filtering[row, column]
