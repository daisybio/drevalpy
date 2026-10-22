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

from drevalpy._grouped_pearson import CONSTANT_TOLERANCE, PearsonSums, grouped_pearson


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
        x = 5.0 + np.array([0.0, CONSTANT_TOLERANCE / 10] * 2)

        assert np.isnan(grouped_pearson(np.zeros(4, dtype=int), 1, x, np.arange(4.0))[0])


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
