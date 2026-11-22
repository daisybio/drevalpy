"""Tests for :mod:`drevalpy.data.splitters._folds`, the helpers all four built-in splitters are assembled from."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.data.quality import CurveQuality
from drevalpy.data.splitters._folds import group_folds, group_out_folds, grouped_folds, observed_mask
from drevalpy.errors import UnknownNameError
from tests._fake_dataset import FakeResponseDataset
from tests.data.splitters._helpers import first_measured_pairs

FOLD_KWARGS = {"n_splits": 5, "validation_ratio": 0.1, "random_state": 42}


@pytest.fixture(scope="module")
def dataset() -> FakeResponseDataset:
    return FakeResponseDataset()


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
    clean = FakeResponseDataset()
    failing = first_measured_pairs(clean, 3)
    dataset = FakeResponseDataset(failing_pairs=failing)
    layers = {name: dataset.get_response_layer(name).copy() for name in dataset.response_layer_names()}

    filtered = observed_mask(dataset, CurveQuality.default())
    unfiltered = observed_mask(dataset, CurveQuality.off())

    assert not any(filtered[row, column] for row, column in failing)
    assert filtered.sum() == observed_mask(clean, CurveQuality.default()).sum() - len(failing)
    np.testing.assert_array_equal(unfiltered, observed_mask(clean, CurveQuality.off()))
    for name, values in layers.items():
        np.testing.assert_array_equal(dataset.get_response_layer(name), values)


def test_off_works_without_quality_layers_and_infinite_responses_are_not_observed():
    dataset = FakeResponseDataset(layers={})
    row, column = first_measured_pairs(dataset, 1)[0]
    dataset.response_matrix[row, column] = np.inf

    mask = observed_mask(dataset, CurveQuality.off())

    np.testing.assert_array_equal(mask, np.isfinite(dataset.response_matrix))
    assert not mask[row, column]


def test_a_rule_needing_a_missing_layer_raises_an_unknown_name_error_that_is_a_key_error():
    with pytest.raises(UnknownNameError, match="R2") as error:
        observed_mask(FakeResponseDataset(), CurveQuality(min_r2=0.5))

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


def _source_sample_dataset(studies: tuple[str, ...] = ("A", "B")) -> FakeResponseDataset:
    """A 12x8 fake whose rows 2k and 2k+1 are one source sample measured in two studies, as after ``concat``.

    The study of each row cycles through *studies*.
    """
    return FakeResponseDataset(
        shape=(12, 8),
        columns={
            "source_sample_id": [f"S{i // 2}" for i in range(12)],
            "study": [studies[i % len(studies)] for i in range(12)],
        },
    )


def _parts(fold) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return fold.train.mask, fold.val.mask, fold.test.mask


class TestGroupedFolds:
    @pytest.mark.parametrize("mode", ["LSO", "LDO", "LTO", "LPO"])
    def test_the_parts_partition_the_observed_pairs_and_each_pair_is_tested_once(self, dataset, observed, mode):
        folds = grouped_folds(
            dataset, CurveQuality.default(), mode=mode, n_splits=3, validation_ratio=0.3, random_state=0
        )

        tested = np.zeros_like(observed, dtype=int)
        for fold in folds:
            train, val, test = _parts(fold)
            np.testing.assert_array_equal(train | val | test, observed)
            assert not (train & val).any() and not (train & test).any() and not (val & test).any()
            tested += test
        np.testing.assert_array_equal(tested, observed.astype(int))

    def test_lso_holds_the_rows_of_one_source_sample_out_together(self):
        dataset = _source_sample_dataset()

        for fold in grouped_folds(dataset, CurveQuality.off(), mode="LSO", **FOLD_KWARGS):
            parts_of_sample = [
                {
                    name
                    for name, part in zip(("train", "val", "test"), _parts(fold), strict=True)
                    if part[2 * k : 2 * k + 2].any()
                }
                for k in range(6)
            ]
            assert all(len(parts) == 1 for parts in parts_of_sample)

    def test_lpo_holds_one_source_sample_and_drug_out_together(self):
        dataset = _source_sample_dataset()
        observed = observed_mask(dataset, CurveQuality.off())
        both = observed[0::2] & observed[1::2]

        for fold in grouped_folds(dataset, CurveQuality.off(), mode="LPO", **FOLD_KWARGS):
            for part in _parts(fold):
                np.testing.assert_array_equal(part[0::2] & both, part[1::2] & both)

    def test_lsto_tests_on_one_study_and_draws_validation_by_source_sample_and_drug(self):
        dataset = _source_sample_dataset(studies=("A", "B", "C"))
        observed = observed_mask(dataset, CurveQuality.off())
        studies = np.array(["A", "B", "C"] * 4)

        folds = group_out_folds(dataset, CurveQuality.off(), mode="LStO", validation_ratio=0.5, random_state=0)

        assert len(folds) == 3
        for study, fold in zip("ABC", folds, strict=True):
            train, val, test = _parts(fold)
            in_study = (studies == study)[:, None]
            np.testing.assert_array_equal(test, observed & in_study)
            np.testing.assert_array_equal(train | val, observed & ~in_study)
            rest = observed & ~in_study
            both = rest[0::2] & rest[1::2]
            assert both.any()
            np.testing.assert_array_equal(val[0::2] & both, val[1::2] & both)
