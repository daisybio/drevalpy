"""Tests for :mod:`drevalpy.data.splitters._folds`, the helpers all four built-in splitters are assembled from."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.data.quality import CurveQuality
from drevalpy.data.splitters._folds import (
    entity_masks,
    group_folds,
    observed_mask,
    pair_masks,
    rows_with_labels,
)
from drevalpy.errors import UnknownNameError
from tests.data.splitters._helpers import MockDataset, first_measured_pairs

FOLD_KWARGS = {"n_splits": 5, "validation_ratio": 0.1, "random_state": 42}
NO_INDICES = np.array([], dtype=int)


@pytest.fixture(scope="module")
def dataset() -> MockDataset:
    return MockDataset()


@pytest.fixture(scope="module")
def observed(dataset) -> np.ndarray:
    return observed_mask(dataset, CurveQuality.default())


def test_observed_mask_marks_the_measured_pairs(dataset, observed):
    before = dataset.response_matrix.copy()

    assert observed.dtype == bool
    assert observed.shape == dataset.response_matrix.shape
    np.testing.assert_array_equal(observed, ~np.isnan(dataset.response_matrix))
    np.testing.assert_array_equal(np.isnan(dataset.response_matrix), np.isnan(before))


def test_observed_mask_drops_the_pairs_that_fail_the_rule_and_off_ignores_it_and_leaves_the_dataset_untouched():
    clean = MockDataset()
    failing = first_measured_pairs(clean, 3)
    dataset = MockDataset(failing_pairs=failing)
    layers = {name: dataset.get_response_layer(name).copy() for name in dataset.response_layer_names()}

    filtered = observed_mask(dataset, CurveQuality.default())
    unfiltered = observed_mask(dataset, CurveQuality.off())

    assert not any(filtered[row, column] for row, column in failing)
    assert filtered.sum() == observed_mask(clean, CurveQuality.default()).sum() - len(failing)
    np.testing.assert_array_equal(unfiltered, observed_mask(clean, CurveQuality.off()))
    for name, values in layers.items():
        np.testing.assert_array_equal(dataset.get_response_layer(name), values)


def test_off_works_without_quality_layers_and_infinite_responses_are_not_observed():
    dataset = MockDataset()
    row, column = first_measured_pairs(dataset, 1)[0]
    dataset.response_matrix[row, column] = np.inf
    dataset._layers = {}

    mask = observed_mask(dataset, CurveQuality.off())

    np.testing.assert_array_equal(mask, np.isfinite(dataset.response_matrix))
    assert not mask[row, column]


def test_a_rule_needing_a_missing_layer_raises_an_unknown_name_error_that_is_a_key_error():
    with pytest.raises(UnknownNameError, match="R2") as error:
        observed_mask(MockDataset(), CurveQuality(min_r2=0.5))

    assert isinstance(error.value, KeyError)


class TestGroupFolds:
    def test_every_group_is_in_exactly_one_test_set_and_each_fold_partitions_the_groups(self):
        folds = list(group_folds(10, **FOLD_KWARGS))

        assert len(folds) == 5
        assert sorted(int(i) for _, _, test in folds for i in test) == list(range(10))
        for train, validation, test in folds:
            assert sorted([*train.tolist(), *validation.tolist(), *test.tolist()]) == list(range(10))
            assert len(validation) >= 1

    def test_the_validation_ratio_controls_the_validation_size(self):
        def sizes(ratio):
            return [len(v) for _, v, _ in group_folds(20, n_splits=4, validation_ratio=ratio, random_state=0)]

        assert sizes(0.0) == [0] * 4
        assert all(a < b for a, b in zip(sizes(0.1), sizes(0.5), strict=True))

    def test_is_reproducible_for_one_seed_and_another_seed_partitions_differently(self):
        def folds(seed):
            return [tuple(map(list, f)) for f in group_folds(12, n_splits=4, validation_ratio=0.1, random_state=seed)]

        assert folds(0) == folds(0)
        assert folds(0) != folds(7)

    def test_rejects_more_splits_than_groups(self):
        with pytest.raises(ValueError, match="number of splits"):
            list(group_folds(3, n_splits=5, validation_ratio=0.1, random_state=42))


class TestEntityMasks:
    def test_rows_land_in_the_mask_of_their_own_side_and_foreign_rows_stay_empty(self, observed):
        masks = entity_masks(observed, train=np.array([0, 1]), validation=np.array([2]), test=np.array([3]), axis=0)

        np.testing.assert_array_equal(masks.train.mask[[0, 1], :], observed[[0, 1], :])
        np.testing.assert_array_equal(masks.val.mask[2, :], observed[2, :])
        np.testing.assert_array_equal(masks.test.mask[3, :], observed[3, :])
        assert not masks.train.mask[2:, :].any()

    def test_a_column_split_holds_out_drugs(self, observed):
        masks = entity_masks(observed, train=np.array([0, 1]), validation=np.array([2]), test=np.array([3]), axis=1)

        np.testing.assert_array_equal(masks.test.mask[:, 3], observed[:, 3])
        assert not masks.test.mask[:, [0, 1, 2]].any()

    def test_unobserved_pairs_are_never_selected_and_an_empty_index_set_yields_an_empty_mask(self, observed):
        rows = np.arange(observed.shape[0])

        masks = entity_masks(observed, train=rows, validation=NO_INDICES, test=rows, axis=0)

        assert not (masks.train.mask & ~observed).any()
        assert not masks.val.mask.any()


class TestPairMasks:
    def test_selects_exactly_the_requested_positions_without_overlap(self, observed):
        rows, columns = np.where(observed)

        masks = pair_masks(
            observed.shape, rows, columns, train=np.arange(5), validation=np.arange(5, 7), test=np.arange(7, 10)
        )

        assert masks.train.mask.shape == observed.shape
        assert masks.train.mask.sum() == 5
        assert masks.val.mask[rows[5], columns[5]]
        assert masks.test.mask[rows[7], columns[7]]
        assert not (masks.train.mask & masks.test.mask).any()
        assert not (masks.train.mask & masks.val.mask).any()

    def test_an_empty_position_set_yields_an_empty_mask(self, observed):
        rows, columns = np.where(observed)

        masks = pair_masks(
            observed.shape, rows, columns, train=np.arange(3), validation=NO_INDICES, test=np.arange(3, 5)
        )

        assert not masks.val.mask.any()


@pytest.mark.parametrize(
    ("selected", "expected"),
    [(["a"], [0, 2]), (["b", "c"], [1, 3]), ([], []), (["z"], [])],
    ids=["one", "several", "none", "unknown"],
)
def test_rows_with_labels(selected, expected):
    labels = np.array(["a", "b", "a", "c"])

    np.testing.assert_array_equal(rows_with_labels(labels, np.array(selected, dtype=labels.dtype)), expected)
