"""Tests for SplitMask and SplitMasks."""

from __future__ import annotations

import tempfile

import numpy as np

from drevalpy.types import SplitMask, SplitMasks


def _mask(shape: tuple[int, int], *positions: tuple[int, int]) -> SplitMask:
    """Helper to build a SplitMask with True at given positions."""
    m = np.zeros(shape, dtype=bool)
    for r, c in positions:
        m[r, c] = True
    return SplitMask(m)


class TestSplitMasks:
    def test_creation(self):
        shape = (4, 3)
        masks = SplitMasks(
            train=_mask(shape, (0, 0), (1, 1)),
            test=_mask(shape, (2, 0)),
            val=_mask(shape, (3, 1)),
        )
        assert masks.train.shape == shape
        assert masks.test.shape == shape
        assert masks.val.shape == shape
        assert len(masks.train) == 2
        assert len(masks.test) == 1
        assert len(masks.val) == 1

    def test_metadata_default_empty(self):
        shape = (2, 2)
        masks = SplitMasks(
            train=_mask(shape, (0, 0)),
            test=_mask(shape, (1, 0)),
            val=SplitMask(np.zeros(shape, dtype=bool)),
        )
        assert masks.metadata == {}

    def test_metadata_mutable(self):
        shape = (2, 2)
        masks = SplitMasks(
            train=_mask(shape, (0, 0)),
            test=_mask(shape, (1, 0)),
            val=SplitMask(np.zeros(shape, dtype=bool)),
        )
        masks.metadata["key"] = "value"
        assert masks.metadata["key"] == "value"

    def test_save_load_roundtrip(self):
        shape = (6, 3)
        masks = SplitMasks(
            train=_mask(shape, (0, 0), (1, 1), (2, 2)),
            test=_mask(shape, (3, 0), (4, 1)),
            val=_mask(shape, (5, 2)),
            metadata={"mode": "LCO", "fold_index": 0, "custom": 42},
        )
        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            masks.save(f.name)
            loaded = SplitMasks.load(f.name)

        np.testing.assert_array_equal(loaded.train.mask, masks.train.mask)
        np.testing.assert_array_equal(loaded.test.mask, masks.test.mask)
        np.testing.assert_array_equal(loaded.val.mask, masks.val.mask)
        assert loaded.metadata == masks.metadata

    def test_save_load_empty_val(self):
        shape = (2, 2)
        masks = SplitMasks(
            train=_mask(shape, (0, 0)),
            test=_mask(shape, (1, 0)),
            val=SplitMask(np.zeros(shape, dtype=bool)),
        )
        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            masks.save(f.name)
            loaded = SplitMasks.load(f.name)

        assert not loaded.val.any()
        assert loaded.val.shape == shape

    def test_save_load_no_metadata(self):
        shape = (2, 2)
        masks = SplitMasks(
            train=_mask(shape, (0, 0)),
            test=_mask(shape, (1, 0)),
            val=SplitMask(np.zeros(shape, dtype=bool)),
        )
        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            masks.save(f.name)
            loaded = SplitMasks.load(f.name)

        assert loaded.metadata == {}


class TestSplitMask:
    def test_creation_from_mask(self):
        mask = np.array([[True, False], [False, True]])
        scope = SplitMask(mask=mask)
        assert scope.pairs.shape == (2, 2)
        assert len(scope) == 2

    def test_from_pairs(self):
        pairs = np.array([[0, 0], [1, 1]])
        scope = SplitMask.from_pairs(pairs, shape=(2, 2))
        assert scope.mask[0, 0]
        assert scope.mask[1, 1]
        assert not scope.mask[0, 1]
        assert len(scope) == 2

    def test_pairs_property_matches_mask(self):
        mask = np.array([[True, False, True], [False, True, False]])
        scope = SplitMask(mask=mask)
        expected = np.argwhere(mask)
        np.testing.assert_array_equal(scope.pairs, expected)
