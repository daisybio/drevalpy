"""Tests for the three-way ``SplitMasks`` container.

The single-mask behaviour of :class:`SplitMask` lives in ``test_split_mask.py``.
"""

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
            metadata={"mode": "LSO", "fold_index": 0, "custom": 42},
        )
        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            masks.save(f.name)
            loaded = SplitMasks.load(f.name)

        np.testing.assert_array_equal(loaded.train.mask, masks.train.mask)
        np.testing.assert_array_equal(loaded.test.mask, masks.test.mask)
        np.testing.assert_array_equal(loaded.val.mask, masks.val.mask)
        assert loaded.metadata == masks.metadata

    def test_save_load_preserves_pair_order_seeds(self):
        shape = (3, 2)
        masks = SplitMasks(
            train=_mask(shape, (0, 0)).shuffled(1),
            test=_mask(shape, (1, 0)).shuffled(2),
            val=_mask(shape, (2, 0)).shuffled(3),
        )
        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            masks.save(f.name)
            loaded = SplitMasks.load(f.name)

        assert [loaded.train._pair_seed, loaded.test._pair_seed, loaded.val._pair_seed] == [1, 2, 3]

    def test_loads_legacy_file_without_pair_order_seeds(self, tmp_path):
        path = tmp_path / "legacy.npz"
        arrays = {name: np.zeros((2, 2), dtype=bool) for name in ("train", "test", "val")}
        np.savez_compressed(path, **arrays)

        loaded = SplitMasks.load(path)

        assert [loaded.train._pair_seed, loaded.test._pair_seed, loaded.val._pair_seed] == [None, None, None]

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

    def test_early_stopping_masks_keep_the_validation_seed(self):
        shape = (2, 2)
        masks = SplitMasks(
            train=SplitMask(np.zeros(shape, dtype=bool)),
            test=SplitMask(np.zeros(shape, dtype=bool)),
            val=SplitMask(np.ones(shape, dtype=bool)).shuffled(5),
        )

        early_stopping, remaining = masks.early_stopping_mask()

        assert early_stopping._pair_seed == remaining._pair_seed == 5

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
