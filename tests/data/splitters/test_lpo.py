"""Tests for the leave-pair-out splitter."""

from __future__ import annotations

from drevalpy.data.quality import CurveQuality
from drevalpy.registry.splitter import splitter_registry
from tests.data.splitters._helpers import MockMuDataset, covered_pairs, first_measured_pairs

QUALITY = CurveQuality.default()


def test_folds_are_disjoint_boolean_masks_with_metadata(mock_dataset: MockMuDataset) -> None:
    folds = splitter_registry.get("LPO")(mock_dataset, n_splits=3, quality=QUALITY)
    assert len(folds) == 3
    for i, fold in enumerate(folds):
        for part in (fold.train, fold.val, fold.test):
            assert part.shape == mock_dataset.response_matrix.shape
            assert part.mask.dtype == bool
        assert not (fold.train & fold.test).any()
        assert fold.metadata["mode"] == "LPO"
        assert fold.metadata["fold_index"] == i
        assert fold.metadata["n_splits"] == 3


def test_low_quality_pairs_appear_in_no_fold(mock_dataset: MockMuDataset) -> None:
    failing = first_measured_pairs(mock_dataset, 3)
    splitter = splitter_registry.get("LPO")

    without_filtering = covered_pairs(splitter(mock_dataset, n_splits=3, quality=QUALITY))
    with_filtering = covered_pairs(splitter(MockMuDataset(failing_pairs=failing), n_splits=3, quality=QUALITY))

    for row, column in failing:
        assert without_filtering[row, column]
        assert not with_filtering[row, column]
