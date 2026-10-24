"""Tests for the three-way ``SplitMasks`` container."""

from __future__ import annotations

import numpy as np

from drevalpy.types import SplitMask, SplitMasks


def _mask(shape: tuple[int, int], *positions: tuple[int, int]) -> SplitMask:
    m = np.zeros(shape, dtype=bool)
    for r, c in positions:
        m[r, c] = True
    return SplitMask(m)


def _empty(shape: tuple[int, int]) -> SplitMask:
    return SplitMask(np.zeros(shape, dtype=bool))


def _seeds(masks: SplitMasks) -> list:
    return [masks.train._pair_seed, masks.test._pair_seed, masks.val._pair_seed]


class TestSplitMasks:
    def test_creation_and_metadata(self):
        shape = (4, 3)
        masks = SplitMasks(train=_mask(shape, (0, 0), (1, 1)), test=_mask(shape, (2, 0)), val=_mask(shape, (3, 1)))

        assert masks.train.shape == masks.test.shape == masks.val.shape == shape
        assert [len(masks.train), len(masks.test), len(masks.val)] == [2, 1, 1]
        assert masks.metadata == {}
        masks.metadata["key"] = "value"
        assert masks.metadata["key"] == "value"

    def test_save_load_roundtrip(self, tmp_path):
        shape = (6, 3)
        masks = SplitMasks(
            train=_mask(shape, (0, 0), (1, 1), (2, 2)),
            test=_mask(shape, (3, 0), (4, 1)),
            val=_mask(shape, (5, 2)),
            metadata={"mode": "LSO", "fold_index": 0, "custom": 42},
        )
        masks.save(tmp_path / "masks.npz")

        loaded = SplitMasks.load(tmp_path / "masks.npz")

        for name in ("train", "test", "val"):
            np.testing.assert_array_equal(getattr(loaded, name).mask, getattr(masks, name).mask)
        assert loaded.metadata == masks.metadata

    def test_save_load_empty_val_and_no_metadata(self, tmp_path):
        masks = SplitMasks(train=_mask((2, 2), (0, 0)), test=_mask((2, 2), (1, 0)), val=_empty((2, 2)))
        masks.save(tmp_path / "masks.npz")

        loaded = SplitMasks.load(tmp_path / "masks.npz")

        assert not loaded.val.any()
        assert loaded.val.shape == (2, 2)
        assert loaded.metadata == {}

    def test_pair_order_seeds_round_trip_and_legacy_files_have_none(self, tmp_path):
        shape = (3, 2)
        masks = SplitMasks(
            train=_mask(shape, (0, 0)).shuffled(1),
            test=_mask(shape, (1, 0)).shuffled(2),
            val=_mask(shape, (2, 0)).shuffled(3),
        )
        masks.save(tmp_path / "masks.npz")
        assert _seeds(SplitMasks.load(tmp_path / "masks.npz")) == [1, 2, 3]

        np.savez_compressed(
            tmp_path / "legacy.npz", **{n: np.zeros((2, 2), dtype=bool) for n in ("train", "test", "val")}
        )
        assert _seeds(SplitMasks.load(tmp_path / "legacy.npz")) == [None, None, None]

    def test_early_stopping_masks_keep_the_validation_seed(self):
        masks = SplitMasks(
            train=_empty((2, 2)), test=_empty((2, 2)), val=SplitMask(np.ones((2, 2), dtype=bool)).shuffled(5)
        )

        early_stopping, remaining = masks.early_stopping_mask()

        assert early_stopping._pair_seed == remaining._pair_seed == 5
