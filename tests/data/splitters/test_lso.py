"""Tests for the leave-sample-out splitter."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.registry.splitter import splitter_registry
from tests.data.splitters._helpers import MockMuDataset, covered_pairs, first_measured_pairs


class TestLeaveSampleOut:
    def test_produces_folds(self, mock_dataset: MockMuDataset) -> None:
        splitter = splitter_registry.get("LSO")
        folds = splitter(mock_dataset, n_splits=3)
        assert len(folds) == 3

    def test_no_sample_in_both_train_and_test(self, mock_dataset: MockMuDataset) -> None:
        splitter = splitter_registry.get("LSO")
        folds = splitter(mock_dataset, n_splits=3)
        for fold in folds:
            train_rows = set(np.where(fold.train.mask.any(axis=1))[0].tolist())
            test_rows = set(np.where(fold.test.mask.any(axis=1))[0].tolist())
            assert train_rows & test_rows == set()

    def test_all_indices_within_bounds(self, mock_dataset: MockMuDataset) -> None:
        splitter = splitter_registry.get("LSO")
        folds = splitter(mock_dataset, n_splits=3)
        shape = mock_dataset.response_matrix.shape
        for fold in folds:
            assert fold.train.shape == shape
            assert fold.test.shape == shape
            assert fold.val.shape == shape

    def test_low_quality_pairs_appear_in_no_fold(self, mock_dataset: MockMuDataset) -> None:
        """Measured pairs whose curve fails the thresholds are never split into."""
        failing = first_measured_pairs(mock_dataset, 3)
        splitter = splitter_registry.get("LSO")

        without_filtering = covered_pairs(splitter(mock_dataset, n_splits=3))
        with_filtering = covered_pairs(splitter(MockMuDataset(failing_pairs=failing), n_splits=3))

        for row, column in failing:
            assert without_filtering[row, column]
            assert not with_filtering[row, column]


class TestLcoAlias:
    """``LCO`` is the pre-rename name of ``LSO`` and must keep working."""

    def test_the_alias_returns_the_same_splitter(self) -> None:
        assert splitter_registry.get("LCO") is splitter_registry.get("LSO")

    def test_the_alias_gives_the_same_folds(self, mock_dataset: MockMuDataset) -> None:
        via_alias = splitter_registry.get("LCO")(mock_dataset, n_splits=3)
        via_mode = splitter_registry.get("LSO")(mock_dataset, n_splits=3)

        assert len(via_alias) == len(via_mode)
        for old, new in zip(via_alias, via_mode, strict=True):
            for name in ("train", "val", "test"):
                np.testing.assert_array_equal(getattr(old, name).mask, getattr(new, name).mask)

    def test_the_alias_is_not_a_separate_mode(self) -> None:
        assert "LCO" not in splitter_registry.modes
        assert splitter_registry.get_metadata("LCO")["aliases"] == ["LCO"]

    def test_folds_report_the_canonical_mode(self, mock_dataset: MockMuDataset) -> None:
        folds = splitter_registry.get("LCO")(mock_dataset, n_splits=3)

        assert {fold.metadata["mode"] for fold in folds} == {"LSO"}

    def test_re_registering_the_alias_as_a_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="conflicts with an alias"):
            splitter_registry.register("LCO", "clash", validation="LSO")(lambda *args, **kwargs: [])
