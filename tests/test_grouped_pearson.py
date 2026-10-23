"""Tests for :mod:`drevalpy._grouped_pearson`, the streaming per-group correlation.

``grouped_pearson`` replaces a ``groupby().apply(pearsonr)`` with ``bincount`` sums, so the
tests pin the numerical agreement with :func:`scipy.stats.pearsonr` as well as the
degenerate cases where the two disagree deliberately: a group with fewer than two
observations, and a group with no variance on one axis, both of which yield NaN here.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import pearsonr

from drevalpy._grouped_pearson import PearsonSums, grouped_pearson
from drevalpy.evaluation import CONSTANT_ATOL, is_constant, pearson


class TestGroupedPearson:
    def test_agrees_with_scipy_per_group(self):
        rng = np.random.default_rng(0)
        codes = np.repeat([0, 1], 25)
        x = rng.normal(size=50)
        y = rng.normal(size=50)

        result = grouped_pearson(codes, 2, x, y)

        for group in (0, 1):
            expected = pearsonr(x[codes == group], y[codes == group])[0]
            assert result[group] == pytest.approx(expected, abs=1e-9)

    def test_a_perfect_positive_relationship_is_one(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])

        assert grouped_pearson(np.zeros(4, dtype=int), 1, x, 2 * x + 1)[0] == pytest.approx(1.0)

    def test_a_perfect_negative_relationship_is_minus_one(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])

        assert grouped_pearson(np.zeros(4, dtype=int), 1, x, -x)[0] == pytest.approx(-1.0)

    def test_a_single_observation_group_is_nan(self):
        result = grouped_pearson(np.array([0, 1, 1]), 2, np.array([1.0, 1.0, 2.0]), np.array([1.0, 3.0, 4.0]))

        assert np.isnan(result[0])
        assert np.isfinite(result[1])

    def test_a_group_with_no_variance_is_nan(self):
        codes = np.zeros(4, dtype=int)

        result = grouped_pearson(codes, 1, np.ones(4), np.array([1.0, 2.0, 3.0, 4.0]))

        assert np.isnan(result[0])

    def test_an_empty_group_keeps_its_slot_as_nan(self):
        result = grouped_pearson(np.zeros(3, dtype=int), 3, np.array([1.0, 2.0, 3.0]), np.array([1.0, 3.0, 2.0]))

        assert result.shape == (3,)
        assert np.isnan(result[1:]).all()

    def test_negative_codes_are_dropped(self):
        codes = np.array([-1, 0, 0, 0])

        result = grouped_pearson(codes, 1, np.array([99.0, 1.0, 2.0, 3.0]), np.array([-99.0, 1.0, 2.0, 3.0]))

        assert result[0] == pytest.approx(1.0)

    def test_nan_observations_are_dropped(self):
        codes = np.zeros(5, dtype=int)
        x = np.array([np.nan, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 1.0, 2.0, 3.0, np.nan])

        result = grouped_pearson(codes, 1, x, y)

        assert result[0] == pytest.approx(1.0)

    def test_dropping_nans_can_take_a_group_below_the_minimum(self):
        codes = np.zeros(3, dtype=int)

        result = grouped_pearson(codes, 1, np.array([1.0, np.nan, np.nan]), np.array([1.0, 2.0, 3.0]))

        assert np.isnan(result[0])

    def test_a_higher_min_count_filters_small_groups(self):
        codes = np.zeros(3, dtype=int)
        x = np.array([1.0, 2.0, 3.0])

        assert np.isfinite(grouped_pearson(codes, 1, x, x, min_count=3)[0])
        assert np.isnan(grouped_pearson(codes, 1, x, x, min_count=4)[0])

    def test_results_stay_inside_the_unit_interval(self):
        rng = np.random.default_rng(1)
        codes = rng.integers(0, 5, size=200)
        x = rng.normal(size=200)

        result = grouped_pearson(codes, 5, x, x * 3.0)

        assert np.all(np.abs(result) <= 1.0)

    def test_all_observations_dropped_yields_all_nan(self):
        result = grouped_pearson(np.array([-1, -1]), 2, np.array([1.0, 2.0]), np.array([1.0, 2.0]))

        assert np.isnan(result).all()


class TestConstantDetection:
    def test_a_large_constant_series_is_not_mistaken_for_variation(self):
        x = np.full(50, 1234.5678901)

        assert np.isnan(grouped_pearson(np.zeros(50, dtype=int), 1, x, np.arange(50.0))[0])

    def test_a_range_within_the_tolerance_counts_as_constant(self):
        x = 5.0 + np.array([0.0, CONSTANT_ATOL / 10] * 2)

        assert np.isnan(grouped_pearson(np.zeros(4, dtype=int), 1, x, np.arange(4.0))[0])

    def test_a_half_unit_range_at_a_1e5_offset_counts_as_constant(self):
        # the relative tolerance of np.isclose makes the old ``range <= 1e-6`` rule stricter here
        x = 1e5 + np.array([0.0, 0.5] * 2)

        assert np.isnan(grouped_pearson(np.zeros(4, dtype=int), 1, x, np.arange(4.0))[0])

    def test_a_two_unit_range_at_a_1e5_offset_varies(self):
        x = 1e5 + np.array([0.0, 2.0] * 2)

        assert np.isfinite(grouped_pearson(np.zeros(4, dtype=int), 1, x, np.arange(4.0))[0])

    def test_an_empty_group_does_not_vary(self):
        sums = PearsonSums(2)
        sums.add(np.array([0, 0]), np.array([1.0, 2.0]), np.array([3.0, 5.0]))

        assert sums.varies().tolist() == [True, False]
        assert sums.y_varies().tolist() == [True, False]

    def test_each_group_is_judged_on_its_own_range(self):
        codes = np.array([0, 0, 1, 1, 2, 2])
        x = np.array([0.0, 5e-7, 0.0, 2e-6, 1e5, 1e5 + 0.5])

        result = grouped_pearson(codes, 3, x, np.array([0.0, 1.0] * 3))

        assert np.isnan(result).tolist() == [True, False, True]


SERIES = {
    "constant": np.full(6, 3.0),
    "zero_and_5e-7": np.array([0.0, 5e-7, 0.0, 5e-7, 0.0, 5e-7]),
    "zero_and_2e-6": np.array([0.0, 2e-6, 0.0, 2e-6, 0.0, 2e-6]),
    "1e5_offset_tiny_range": 1e5 + np.array([0.0, 1e-7, 0.0, 1e-7, 0.0, 1e-7]),
    "1e5_offset_half_range": 1e5 + np.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.5]),
    "1e5_offset_wide_range": 1e5 + np.array([0.0, 5.0, 0.0, 5.0, 0.0, 5.0]),
    "negative_1e5_offset": -1e5 + np.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.5]),
    "varying": np.arange(6.0),
}


class TestAgreementWithPooledRule:
    @pytest.mark.parametrize("name", SERIES)
    def test_grouped_and_pooled_agree_on_which_series_is_constant(self, name):
        series, varying = SERIES[name], np.arange(6.0)
        codes = np.zeros(6, dtype=int)

        assert np.isnan(grouped_pearson(codes, 1, series, varying)[0]) == is_constant(series)
        assert np.isnan(grouped_pearson(codes, 1, varying, series)[0]) == is_constant(series)
        assert (pearson(series, varying) == 0.0) == is_constant(series)
        assert np.isnan(pearson(varying, series)) == is_constant(series)

    @pytest.mark.parametrize("name", SERIES)
    def test_the_grouped_verdict_is_unchanged_by_any_order(self, name):
        rng = np.random.default_rng(0)
        series, varying = SERIES[name], np.arange(6.0)
        codes = np.zeros(6, dtype=int)
        expected = np.isnan(grouped_pearson(codes, 1, series, varying)[0])

        for _ in range(20):
            order = rng.permutation(6)
            assert np.isnan(grouped_pearson(codes, 1, series[order], varying[order])[0]) == expected


class TestStreaming:
    def test_folds_added_one_at_a_time_equal_one_pooled_batch(self):
        rng = np.random.default_rng(2)
        codes = rng.integers(0, 3, size=60)
        x, y = rng.normal(size=60), rng.normal(size=60)
        streamed = PearsonSums(3)
        streamed.add(codes[:30], x[:30], y[:30])
        streamed.add(codes[30:], x[30:], y[30:])

        np.testing.assert_allclose(streamed.correlations(), grouped_pearson(codes, 3, x, y))

    def test_counts_are_exposed_per_group(self):
        sums = PearsonSums(2)
        sums.add(np.array([0, 0, 1]), np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))

        np.testing.assert_array_equal(sums.counts, [2, 1])
