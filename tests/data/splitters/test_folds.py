"""Tests for :mod:`drevalpy.data.splitters._folds`.

Mirrors the private module with the underscore stripped. All four built-in
splitters are assembled from these helpers, so the properties every mode inherits
- quality filtering, a disjoint train/validation/test partition, reproducibility
from ``random_state`` - are asserted here once, directly on the helpers, rather
than four times over through the registered splitters.
"""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.data.splitters._folds import (
    entity_masks,
    group_folds,
    observed_mask,
    pair_masks,
    rows_with_labels,
)
from tests.data.splitters._helpers import MockMuDataset, first_measured_pairs

FOLD_KWARGS = {"n_splits": 5, "validation_ratio": 0.1, "random_state": 42}


@pytest.fixture(scope="module")
def dataset() -> MockMuDataset:
    return MockMuDataset()


@pytest.fixture(scope="module")
def observed(dataset) -> np.ndarray:
    return observed_mask(dataset)


class TestObservedMask:
    def test_is_a_boolean_matrix_of_the_response_shape(self, dataset, observed):
        assert observed.dtype == bool
        assert observed.shape == dataset.response_matrix.shape

    def test_marks_the_measured_pairs(self, dataset, observed):
        np.testing.assert_array_equal(observed, ~np.isnan(dataset.response_matrix))

    def test_drops_pairs_that_fail_the_quality_filter(self):
        clean = MockMuDataset()
        failing = first_measured_pairs(clean, 3)

        filtered = observed_mask(MockMuDataset(failing_pairs=failing))

        assert not any(filtered[row, column] for row, column in failing)
        assert filtered.sum() == observed_mask(clean).sum() - len(failing)

    def test_does_not_mutate_the_response_matrix(self, dataset):
        before = dataset.response_matrix.copy()

        observed_mask(dataset)

        np.testing.assert_array_equal(np.isnan(dataset.response_matrix), np.isnan(before))


class TestGroupFolds:
    def test_yields_one_triple_per_split(self):
        assert len(list(group_folds(10, **FOLD_KWARGS))) == 5

    def test_every_group_is_in_exactly_one_test_set(self):
        test_sets = [test for _, _, test in group_folds(10, **FOLD_KWARGS)]

        assert sorted(int(i) for test in test_sets for i in test) == list(range(10))

    def test_the_three_index_sets_of_a_fold_are_disjoint(self):
        for train, validation, test in group_folds(10, **FOLD_KWARGS):
            assert set(train.tolist()).isdisjoint(validation.tolist())
            assert set(train.tolist()).isdisjoint(test.tolist())
            assert set(validation.tolist()).isdisjoint(test.tolist())

    def test_a_fold_covers_every_group(self):
        for train, validation, test in group_folds(10, **FOLD_KWARGS):
            assert sorted([*train.tolist(), *validation.tolist(), *test.tolist()]) == list(range(10))

    def test_holds_out_at_least_one_validation_group_when_the_ratio_is_positive(self):
        for _, validation, _ in group_folds(10, **FOLD_KWARGS):
            assert len(validation) >= 1

    def test_a_zero_ratio_yields_no_validation_group(self):
        for _, validation, _ in group_folds(10, n_splits=5, validation_ratio=0.0, random_state=42):
            assert len(validation) == 0

    def test_a_larger_ratio_holds_out_more(self):
        small = [len(v) for _, v, _ in group_folds(20, n_splits=4, validation_ratio=0.1, random_state=0)]
        large = [len(v) for _, v, _ in group_folds(20, n_splits=4, validation_ratio=0.5, random_state=0)]

        assert all(a < b for a, b in zip(small, large, strict=True))

    def test_is_reproducible_for_one_seed(self):
        first = [tuple(map(list, fold)) for fold in group_folds(12, **FOLD_KWARGS)]
        second = [tuple(map(list, fold)) for fold in group_folds(12, **FOLD_KWARGS)]

        assert first == second

    def test_another_seed_partitions_differently(self):
        a = [sorted(test.tolist()) for _, _, test in group_folds(12, n_splits=4, validation_ratio=0.1, random_state=0)]
        b = [sorted(test.tolist()) for _, _, test in group_folds(12, n_splits=4, validation_ratio=0.1, random_state=7)]

        assert a != b

    def test_rejects_more_splits_than_groups(self):
        with pytest.raises(ValueError, match="number of splits"):
            list(group_folds(3, n_splits=5, validation_ratio=0.1, random_state=42))


class TestEntityMasks:
    def test_rows_land_in_the_mask_of_their_own_side(self, observed):
        masks = entity_masks(observed, train=np.array([0, 1]), validation=np.array([2]), test=np.array([3]), axis=0)

        np.testing.assert_array_equal(masks.train.mask[[0, 1], :], observed[[0, 1], :])
        np.testing.assert_array_equal(masks.val.mask[2, :], observed[2, :])
        np.testing.assert_array_equal(masks.test.mask[3, :], observed[3, :])

    def test_a_row_split_leaves_foreign_rows_empty(self, observed):
        masks = entity_masks(observed, train=np.array([0]), validation=np.array([1]), test=np.array([2]), axis=0)

        assert not masks.train.mask[1:, :].any()

    def test_a_column_split_holds_out_drugs(self, observed):
        masks = entity_masks(observed, train=np.array([0, 1]), validation=np.array([2]), test=np.array([3]), axis=1)

        np.testing.assert_array_equal(masks.test.mask[:, 3], observed[:, 3])
        assert not masks.test.mask[:, [0, 1, 2]].any()

    def test_unobserved_pairs_are_never_selected(self, observed):
        rows = np.arange(observed.shape[0])

        masks = entity_masks(observed, train=rows, validation=np.array([], dtype=int), test=rows, axis=0)

        assert not (masks.train.mask & ~observed).any()

    def test_an_empty_index_set_yields_an_empty_mask(self, observed):
        masks = entity_masks(
            observed,
            train=np.array([0]),
            validation=np.array([], dtype=int),
            test=np.array([1]),
            axis=0,
        )

        assert not masks.val.mask.any()


class TestPairMasks:
    def test_selects_exactly_the_requested_positions(self, observed):
        rows, columns = np.where(observed)

        masks = pair_masks(
            observed.shape,
            rows,
            columns,
            train=np.array([0, 1]),
            validation=np.array([2]),
            test=np.array([3]),
        )

        assert masks.train.mask.sum() == 2
        assert masks.val.mask[rows[2], columns[2]]
        assert masks.test.mask[rows[3], columns[3]]

    def test_the_three_masks_do_not_overlap(self, observed):
        rows, columns = np.where(observed)

        masks = pair_masks(
            observed.shape,
            rows,
            columns,
            train=np.arange(5),
            validation=np.arange(5, 7),
            test=np.arange(7, 10),
        )

        assert not (masks.train.mask & masks.test.mask).any()
        assert not (masks.train.mask & masks.val.mask).any()

    def test_an_empty_position_set_yields_an_empty_mask(self, observed):
        rows, columns = np.where(observed)

        masks = pair_masks(
            observed.shape,
            rows,
            columns,
            train=np.arange(3),
            validation=np.array([], dtype=int),
            test=np.arange(3, 5),
        )

        assert not masks.val.mask.any()

    def test_masks_have_the_response_shape(self, observed):
        rows, columns = np.where(observed)

        masks = pair_masks(
            observed.shape,
            rows,
            columns,
            train=np.arange(2),
            validation=np.array([2]),
            test=np.array([3]),
        )

        assert masks.train.mask.shape == observed.shape


class TestRowsWithLabels:
    def test_returns_the_rows_carrying_a_selected_label(self):
        labels = np.array(["a", "b", "a", "c"])

        np.testing.assert_array_equal(rows_with_labels(labels, np.array(["a"])), [0, 2])

    def test_accepts_several_labels(self):
        labels = np.array(["a", "b", "a", "c"])

        np.testing.assert_array_equal(rows_with_labels(labels, np.array(["b", "c"])), [1, 3])

    def test_no_selected_label_yields_no_rows(self):
        labels = np.array(["a", "b"])

        assert rows_with_labels(labels, np.array([], dtype=labels.dtype)).size == 0

    def test_an_unknown_label_selects_nothing(self):
        labels = np.array(["a", "b"])

        assert rows_with_labels(labels, np.array(["z"])).size == 0
