"""Tests for the leave-pair-out splitter."""

from __future__ import annotations

from drevalpy.registry.splitter import splitter_registry
from tests.data.splitters._helpers import MockMuDataset, covered_pairs, first_measured_pairs


class TestLeavePairOut:
    def test_produces_correct_number_of_folds(self, mock_dataset: MockMuDataset) -> None:
        splitter = splitter_registry.get("LPO")
        folds = splitter(mock_dataset, n_splits=3)
        assert len(folds) == 3

    def test_all_folds_are_2d_bool(self, mock_dataset: MockMuDataset) -> None:
        splitter = splitter_registry.get("LPO")
        folds = splitter(mock_dataset, n_splits=3)
        shape = mock_dataset.response_matrix.shape
        for fold in folds:
            assert fold.train.shape == shape
            assert fold.test.shape == shape
            assert fold.val.shape == shape
            assert fold.train.mask.dtype == bool
            assert fold.test.mask.dtype == bool
            assert fold.val.mask.dtype == bool

    def test_no_pair_in_both_train_and_test(self, mock_dataset: MockMuDataset) -> None:
        splitter = splitter_registry.get("LPO")
        folds = splitter(mock_dataset, n_splits=3)
        for fold in folds:
            assert not (fold.train & fold.test).any()

    def test_metadata_injected(self, mock_dataset: MockMuDataset) -> None:
        splitter = splitter_registry.get("LPO")
        folds = splitter(mock_dataset, n_splits=3)
        for i, fold in enumerate(folds):
            assert fold.metadata["mode"] == "LPO"
            assert fold.metadata["fold_index"] == i
            assert fold.metadata["n_splits"] == 3

    def test_low_quality_pairs_appear_in_no_fold(self, mock_dataset: MockMuDataset) -> None:
        """Measured pairs whose curve fails the thresholds are never split into."""
        failing = first_measured_pairs(mock_dataset, 3)
        splitter = splitter_registry.get("LPO")

        without_filtering = covered_pairs(splitter(mock_dataset, n_splits=3))
        with_filtering = covered_pairs(splitter(MockMuDataset(failing_pairs=failing), n_splits=3))

        for row, column in failing:
            # Present when every curve passes, gone once it does not: the
            # absence is the filter's doing, not an artefact of the fixture.
            assert without_filtering[row, column]
            assert not with_filtering[row, column]
