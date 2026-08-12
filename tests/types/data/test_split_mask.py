"""Tests for the single 2-D boolean :class:`SplitMask`.

Carved out of ``test_split_masks.py``, which now covers only the three-way
``SplitMasks`` container it mirrors.
"""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.types import SplitMask


class TestConstruction:
    def test_creation_from_mask(self):
        mask = np.array([[True, False], [False, True]])

        scope = SplitMask(mask=mask)

        assert scope.pairs.shape == (2, 2)
        assert len(scope) == 2

    def test_non_boolean_input_is_coerced(self):
        scope = SplitMask(mask=np.array([[1, 0], [0, 2]]))

        assert scope.mask.dtype == np.bool_
        assert len(scope) == 2

    def test_from_pairs(self):
        pairs = np.array([[0, 0], [1, 1]])

        scope = SplitMask.from_pairs(pairs, shape=(2, 2))

        assert scope.mask[0, 0]
        assert scope.mask[1, 1]
        assert not scope.mask[0, 1]
        assert len(scope) == 2

    def test_from_empty_pairs_yields_an_all_false_mask(self):
        scope = SplitMask.from_pairs(np.empty((0, 2), dtype=int), shape=(3, 2))

        assert scope.shape == (3, 2)
        assert not scope.any()

    def test_shape_reports_the_underlying_matrix_shape(self):
        scope = SplitMask(np.zeros((4, 7), dtype=bool))

        assert scope.shape == (4, 7)


class TestPairs:
    def test_pairs_property_matches_mask(self):
        mask = np.array([[True, False, True], [False, True, False]])
        scope = SplitMask(mask=mask)

        np.testing.assert_array_equal(scope.pairs, np.argwhere(mask))

    def test_pairs_are_row_major_without_a_seed(self):
        scope = SplitMask(np.ones((2, 2), dtype=bool))

        assert scope.pairs.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]

    def test_shuffled_preserves_the_pair_set(self):
        scope = SplitMask(np.ones((3, 3), dtype=bool))

        shuffled = scope.shuffled(seed=0)

        assert sorted(map(tuple, shuffled.pairs)) == sorted(map(tuple, scope.pairs))

    def test_shuffled_reorders_the_pairs(self):
        scope = SplitMask(np.ones((4, 4), dtype=bool))

        shuffled = scope.shuffled(seed=0)

        assert shuffled.pairs.tolist() != scope.pairs.tolist()

    def test_shuffled_is_reproducible_for_one_seed(self):
        scope = SplitMask(np.ones((4, 4), dtype=bool))

        first = scope.shuffled(seed=7).pairs
        second = scope.shuffled(seed=7).pairs

        np.testing.assert_array_equal(first, second)

    def test_shuffled_leaves_the_mask_untouched(self):
        scope = SplitMask(np.array([[True, False], [False, True]]))

        shuffled = scope.shuffled(seed=1)

        np.testing.assert_array_equal(shuffled.mask, scope.mask)


class TestSetOperations:
    def test_or_unions_the_masks(self):
        left = SplitMask(np.array([[True, False], [False, False]]))
        right = SplitMask(np.array([[False, True], [False, False]]))

        assert (left | right).pairs.tolist() == [[0, 0], [0, 1]]

    def test_and_intersects_the_masks(self):
        left = SplitMask(np.array([[True, True], [False, False]]))
        right = SplitMask(np.array([[False, True], [True, False]]))

        assert (left & right).pairs.tolist() == [[0, 1]]

    def test_invert_flips_every_entry(self):
        scope = SplitMask(np.array([[True, False], [False, False]]))

        assert (~scope).pairs.tolist() == [[0, 1], [1, 0], [1, 1]]

    def test_train_and_test_partition_of_a_full_mask_is_disjoint(self):
        train = SplitMask(np.array([[True, True], [False, False]]))

        assert not (train & ~train).any()


class TestPredicates:
    @pytest.mark.parametrize(
        ("mask", "expected"),
        [
            pytest.param(np.zeros((2, 2), dtype=bool), False, id="all-false"),
            pytest.param(np.array([[False, True], [False, False]]), True, id="single-true"),
        ],
    )
    def test_any_reports_whether_a_pair_is_selected(self, mask, expected):
        assert SplitMask(mask).any() is expected

    def test_sum_counts_selected_pairs(self):
        scope = SplitMask(np.array([[True, True], [False, True]]))

        assert scope.sum() == 3

    def test_sum_returns_a_python_int(self):
        assert type(SplitMask(np.ones((2, 2), dtype=bool)).sum()) is int

    def test_len_matches_sum(self):
        scope = SplitMask(np.array([[True, True], [False, True]]))

        assert len(scope) == scope.sum()


class TestValueSemantics:
    def test_masks_with_equal_contents_are_equal(self):
        mask = np.array([[True, False], [False, True]])

        assert SplitMask(mask) == SplitMask(mask.copy())

    def test_masks_with_different_contents_are_unequal(self):
        assert SplitMask(np.ones((2, 2), dtype=bool)) != SplitMask(np.zeros((2, 2), dtype=bool))

    def test_a_shuffle_seed_does_not_affect_equality(self):
        scope = SplitMask(np.ones((2, 2), dtype=bool))

        assert scope == scope.shuffled(seed=3)

    def test_comparison_against_a_foreign_type_is_not_equal(self):
        assert SplitMask(np.ones((1, 1), dtype=bool)) != "not a mask"

    def test_equal_masks_hash_alike(self):
        mask = np.array([[True, False], [False, True]])

        assert hash(SplitMask(mask)) == hash(SplitMask(mask.copy()))

    def test_masks_are_usable_as_set_members(self):
        first = SplitMask(np.ones((2, 2), dtype=bool))
        second = SplitMask(np.zeros((2, 2), dtype=bool))

        assert len({first, SplitMask(np.ones((2, 2), dtype=bool)), second}) == 2
