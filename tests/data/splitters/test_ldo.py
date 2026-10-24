"""Tests for the leave-drug-out splitter."""

from __future__ import annotations

import numpy as np

from drevalpy.data.quality import CurveQuality
from drevalpy.registry.splitter import splitter_registry
from tests.data.splitters._helpers import MockMuDataset, covered_pairs, first_measured_pairs

QUALITY = CurveQuality.default()


def test_no_drug_in_both_train_and_test(mock_dataset: MockMuDataset) -> None:
    folds = splitter_registry.get("LDO")(mock_dataset, n_splits=3, quality=QUALITY)
    assert len(folds) == 3
    for fold in folds:
        train_cols = set(np.where(fold.train.mask.any(axis=0))[0].tolist())
        test_cols = set(np.where(fold.test.mask.any(axis=0))[0].tolist())
        assert train_cols & test_cols == set()


def test_low_quality_pairs_appear_in_no_fold(mock_dataset: MockMuDataset) -> None:
    failing = first_measured_pairs(mock_dataset, 3)
    splitter = splitter_registry.get("LDO")

    without_filtering = covered_pairs(splitter(mock_dataset, n_splits=3, quality=QUALITY))
    with_filtering = covered_pairs(splitter(MockMuDataset(failing_pairs=failing), n_splits=3, quality=QUALITY))

    for row, column in failing:
        assert without_filtering[row, column]
        assert not with_filtering[row, column]
