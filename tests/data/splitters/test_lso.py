"""Tests for the leave-sample-out splitter."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.data.quality import CurveQuality
from drevalpy.registry.splitter import splitter_registry
from tests.data.splitters._helpers import MockMuDataset, covered_pairs, first_measured_pairs

QUALITY = CurveQuality.default()


def test_no_sample_in_both_train_and_test(mock_dataset: MockMuDataset) -> None:
    folds = splitter_registry.get("LSO")(mock_dataset, n_splits=3, quality=QUALITY)
    assert len(folds) == 3
    for fold in folds:
        train_rows = set(np.where(fold.train.mask.any(axis=1))[0].tolist())
        test_rows = set(np.where(fold.test.mask.any(axis=1))[0].tolist())
        assert train_rows & test_rows == set()
        for part in (fold.train, fold.val, fold.test):
            assert part.shape == mock_dataset.response_matrix.shape


def test_low_quality_pairs_appear_in_no_fold(mock_dataset: MockMuDataset) -> None:
    failing = first_measured_pairs(mock_dataset, 3)
    splitter = splitter_registry.get("LSO")

    without_filtering = covered_pairs(splitter(mock_dataset, n_splits=3, quality=QUALITY))
    with_filtering = covered_pairs(splitter(MockMuDataset(failing_pairs=failing), n_splits=3, quality=QUALITY))

    for row, column in failing:
        assert without_filtering[row, column]
        assert not with_filtering[row, column]


class TestLcoAlias:
    """``LCO`` is the pre-rename name of ``LSO`` and must keep working."""

    def test_the_alias_returns_the_same_splitter(self) -> None:
        assert splitter_registry.get("LCO") is splitter_registry.get("LSO")

    def test_the_alias_gives_the_same_folds_under_the_canonical_mode(self, mock_dataset: MockMuDataset) -> None:
        via_alias = splitter_registry.get("LCO")(mock_dataset, n_splits=3, quality=QUALITY)
        via_mode = splitter_registry.get("LSO")(mock_dataset, n_splits=3, quality=QUALITY)

        assert len(via_alias) == len(via_mode)
        for old, new in zip(via_alias, via_mode, strict=True):
            for name in ("train", "val", "test"):
                np.testing.assert_array_equal(getattr(old, name).mask, getattr(new, name).mask)
        assert {fold.metadata["mode"] for fold in via_alias} == {"LSO"}

    def test_the_alias_is_not_a_separate_mode(self) -> None:
        assert "LCO" not in splitter_registry.modes
        assert splitter_registry.get_metadata("LCO")["aliases"] == ["LCO"]

    def test_re_registering_the_alias_as_a_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="conflicts with an alias"):
            splitter_registry.register("LCO", "clash", validation="LSO")(lambda *args, **kwargs: [])
