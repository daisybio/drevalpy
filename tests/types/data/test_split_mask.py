"""Tests for the single 2-D boolean :class:`SplitMask`."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.types import SplitMask


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
        assert not empty.any()


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
        assert not (train & ~train).any()

    def test_pair_order_seed_is_preserved_or_conflicts(self):
        left = SplitMask(np.ones((2, 2), dtype=bool)).shuffled(seed=3)

        assert (left | SplitMask(np.zeros((2, 2), dtype=bool)))._pair_seed == 3
        assert (~left)._pair_seed == 3
        with pytest.raises(ValueError, match="conflicting pair-order seeds"):
            left & SplitMask(np.ones((2, 2), dtype=bool)).shuffled(seed=4)

    def test_set_operations_reject_mismatched_shapes(self):
        with pytest.raises(ValueError, match="Cannot combine mask shapes"):
            SplitMask(np.ones((2, 2), dtype=bool)) | SplitMask(np.ones((3, 2), dtype=bool))


class TestPredicates:
    @pytest.mark.parametrize(
        ("mask", "expected"),
        [(np.zeros((2, 2), dtype=bool), False), (np.array([[False, True], [False, False]]), True)],
        ids=["all-false", "single-true"],
    )
    def test_any(self, mask, expected):
        assert SplitMask(mask).any() is expected

    def test_sum_counts_selected_pairs_as_a_python_int(self):
        scope = _mask([True, True], [False, True])

        assert scope.sum() == len(scope) == 3
        assert type(scope.sum()) is int


class TestValueSemantics:
    def test_equality_and_hash(self):
        mask = np.array([[True, False], [False, True]])
        ones, zeros = np.ones((2, 2), dtype=bool), np.zeros((2, 2), dtype=bool)

        assert SplitMask(mask) == SplitMask(mask.copy())
        assert hash(SplitMask(mask)) == hash(SplitMask(mask.copy()))
        assert SplitMask(ones) != SplitMask(zeros)
        assert SplitMask(ones) == SplitMask(ones).shuffled(seed=3)
        assert SplitMask(ones) != "not a mask"
        assert len({SplitMask(ones), SplitMask(ones.copy()), SplitMask(zeros)}) == 2
