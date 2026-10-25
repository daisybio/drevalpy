"""Split masks: the single boolean SplitMask and the train/test/val SplitMasks container."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.types import SplitMask, SplitMasks


def _mask(*rows: list[bool]) -> SplitMask:
    return SplitMask(np.array(rows))


class TestConstruction:
    def test_creation_and_coercion(self):
        scope = SplitMask(mask=np.array([[True, False], [False, True]]))
        assert scope.pairs.shape == (2, 2)
        assert len(scope) == 2
        assert scope.shape == (2, 2)

        coerced = SplitMask(mask=np.array([[1, 0], [0, 2]]))
        assert coerced.mask.dtype == np.bool_
        assert len(coerced) == 2

    def test_from_pairs(self):
        scope = SplitMask.from_pairs(np.array([[0, 0], [1, 1]]), shape=(2, 2))
        assert scope.mask.tolist() == [[True, False], [False, True]]

        empty = SplitMask.from_pairs(np.empty((0, 2), dtype=int), shape=(3, 2))
        assert empty.shape == (3, 2)
        assert len(empty) == 0


class TestImmutability:
    def test_the_mask_is_read_only_and_does_not_alias_a_writable_input(self):
        source = np.ones((2, 2), dtype=bool)
        scope = SplitMask(source)

        source[0, 0] = False

        assert scope.mask[0, 0]
        with pytest.raises(ValueError, match="read-only"):
            scope.mask[0, 0] = False

    def test_an_already_read_only_mask_is_shared(self):
        scope = SplitMask(np.ones((2, 2), dtype=bool))

        assert SplitMask(scope.mask).mask is scope.mask

    def test_from_pairs_carries_the_pair_seed(self):
        scope = SplitMask.from_pairs(np.array([[0, 0], [1, 1]]), shape=(2, 2), pair_seed=4)

        assert scope.pair_seed == 4


class TestPairs:
    def test_pairs_match_the_mask_in_row_major_order(self):
        mask = np.array([[True, False, True], [False, True, False]])
        np.testing.assert_array_equal(SplitMask(mask=mask).pairs, np.argwhere(mask))
        assert SplitMask(np.ones((2, 2), dtype=bool)).pairs.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]

    def test_shuffled_is_a_reproducible_permutation_of_the_same_pairs(self):
        scope = SplitMask(np.ones((4, 4), dtype=bool))

        shuffled = scope.shuffled(seed=0)

        assert sorted(map(tuple, shuffled.pairs)) == sorted(map(tuple, scope.pairs))
        assert shuffled.pairs.tolist() != scope.pairs.tolist()
        np.testing.assert_array_equal(scope.shuffled(seed=7).pairs, scope.shuffled(seed=7).pairs)
        np.testing.assert_array_equal(shuffled.mask, scope.mask)


class TestSetOperations:
    def test_or_and_invert(self):
        assert (_mask([True, False], [False, False]) | _mask([False, True], [False, False])).pairs.tolist() == [
            [0, 0],
            [0, 1],
        ]
        assert (_mask([True, True], [False, False]) & _mask([False, True], [True, False])).pairs.tolist() == [[0, 1]]
        train = _mask([True, False], [False, False])
        assert (~train).pairs.tolist() == [[0, 1], [1, 0], [1, 1]]
        assert len(train & ~train) == 0

    def test_pair_order_seed_is_preserved_or_conflicts(self):
        left = SplitMask(np.ones((2, 2), dtype=bool)).shuffled(seed=3)

        assert (left | SplitMask(np.zeros((2, 2), dtype=bool))).pair_seed == 3
        assert (~left).pair_seed == 3
        with pytest.raises(ValueError, match="conflicting pair-order seeds"):
            left & SplitMask(np.ones((2, 2), dtype=bool)).shuffled(seed=4)

    def test_set_operations_reject_mismatched_shapes(self):
        with pytest.raises(ValueError, match="Cannot combine mask shapes"):
            SplitMask(np.ones((2, 2), dtype=bool)) | SplitMask(np.ones((3, 2), dtype=bool))


class TestSum:
    def test_sum_counts_selected_pairs_as_a_python_int(self):
        scope = _mask([True, True], [False, True])

        assert scope.sum() == len(scope) == 3
        assert type(scope.sum()) is int


class TestValueSemantics:
    def test_equality_ignores_the_pair_order(self):
        mask = np.array([[True, False], [False, True]])
        ones, zeros = np.ones((2, 2), dtype=bool), np.zeros((2, 2), dtype=bool)

        assert SplitMask(mask) == SplitMask(mask.copy())
        assert SplitMask(ones) != SplitMask(zeros)
        assert SplitMask(ones) == SplitMask(ones).shuffled(seed=3)
        assert SplitMask(ones) != "not a mask"


def _pairs_mask(shape: tuple[int, int], *positions: tuple[int, int]) -> SplitMask:
    m = np.zeros(shape, dtype=bool)
    for r, c in positions:
        m[r, c] = True
    return SplitMask(m)


def _empty(shape: tuple[int, int]) -> SplitMask:
    return SplitMask(np.zeros(shape, dtype=bool))


def _seeds(masks: SplitMasks) -> list:
    return [masks.train.pair_seed, masks.test.pair_seed, masks.val.pair_seed]


class TestSplitMasks:
    def test_creation_and_metadata(self):
        shape = (4, 3)
        masks = SplitMasks(
            train=_pairs_mask(shape, (0, 0), (1, 1)), test=_pairs_mask(shape, (2, 0)), val=_pairs_mask(shape, (3, 1))
        )

        assert masks.train.shape == masks.test.shape == masks.val.shape == shape
        assert [len(masks.train), len(masks.test), len(masks.val)] == [2, 1, 1]
        assert masks.metadata == {}
        masks.metadata["key"] = "value"
        assert masks.metadata["key"] == "value"

    def test_save_load_roundtrip(self, tmp_path):
        shape = (6, 3)
        masks = SplitMasks(
            train=_pairs_mask(shape, (0, 0), (1, 1), (2, 2)),
            test=_pairs_mask(shape, (3, 0), (4, 1)),
            val=_pairs_mask(shape, (5, 2)),
            metadata={"split_mode": "LSO", "fold_index": 0, "custom": 42},
        )
        masks.save(tmp_path / "masks.npz")

        loaded = SplitMasks.load(tmp_path / "masks.npz")

        for name in ("train", "test", "val"):
            np.testing.assert_array_equal(getattr(loaded, name).mask, getattr(masks, name).mask)
        assert loaded.metadata == masks.metadata

    def test_save_creates_the_parent_directory_and_load_releases_the_file(self, tmp_path):
        masks = SplitMasks(train=_pairs_mask((2, 2), (0, 0)), test=_pairs_mask((2, 2), (1, 0)), val=_empty((2, 2)))
        path = tmp_path / "folds" / "fold_0" / "masks.npz"
        masks.save(path)

        SplitMasks.load(path)
        path.unlink()

        assert not path.exists()

    def test_save_load_empty_val_and_no_metadata(self, tmp_path):
        masks = SplitMasks(train=_pairs_mask((2, 2), (0, 0)), test=_pairs_mask((2, 2), (1, 0)), val=_empty((2, 2)))
        masks.save(tmp_path / "masks.npz")

        loaded = SplitMasks.load(tmp_path / "masks.npz")

        assert len(loaded.val) == 0
        assert loaded.val.shape == (2, 2)
        assert loaded.metadata == {}

    def test_pair_order_seeds_round_trip_and_legacy_files_have_none(self, tmp_path):
        shape = (3, 2)
        masks = SplitMasks(
            train=_pairs_mask(shape, (0, 0)).shuffled(1),
            test=_pairs_mask(shape, (1, 0)).shuffled(2),
            val=_pairs_mask(shape, (2, 0)).shuffled(3),
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

        assert early_stopping.pair_seed == remaining.pair_seed == 5
        assert len(early_stopping) == 1
        assert not (early_stopping & remaining).mask.any()
        assert (early_stopping | remaining) == masks.val
