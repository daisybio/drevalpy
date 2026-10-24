"""Tests for :mod:`drevalpy._grouped_pearson`: agreement with scipy and NaN for degenerate groups."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import pearsonr

from drevalpy._grouped_pearson import PearsonSums, grouped_pearson
from drevalpy.evaluation import CONSTANT_ATOL, is_constant, pearson

X4 = np.array([1.0, 2.0, 3.0, 4.0])
ZEROS4 = np.zeros(4, dtype=int)


class TestGroupedPearson:
    def test_agrees_with_scipy_per_group_and_stays_in_the_unit_interval(self):
        rng = np.random.default_rng(0)
        codes = np.repeat([0, 1], 25)
        x = rng.normal(size=50)
        y = rng.normal(size=50)

        result = grouped_pearson(codes, 2, x, y)

        for group in (0, 1):
            assert result[group] == pytest.approx(pearsonr(x[codes == group], y[codes == group])[0], abs=1e-9)
        assert np.all(np.abs(grouped_pearson(rng.integers(0, 5, size=200), 5, np.tile(x, 4), np.tile(x, 4) * 3)) <= 1.0)

    def test_perfect_relationships(self):
        assert grouped_pearson(ZEROS4, 1, X4, 2 * X4 + 1)[0] == pytest.approx(1.0)
        assert grouped_pearson(ZEROS4, 1, X4, -X4)[0] == pytest.approx(-1.0)

    def test_degenerate_groups_are_nan(self):
        single = grouped_pearson(np.array([0, 1, 1]), 2, np.array([1.0, 1.0, 2.0]), np.array([1.0, 3.0, 4.0]))
        no_variance = grouped_pearson(ZEROS4, 1, np.ones(4), X4)
        empty = grouped_pearson(np.zeros(3, dtype=int), 3, np.array([1.0, 2.0, 3.0]), np.array([1.0, 3.0, 2.0]))
        all_dropped = grouped_pearson(np.array([-1, -1]), 2, np.array([1.0, 2.0]), np.array([1.0, 2.0]))

        assert np.isnan(single[0])
        assert np.isfinite(single[1])
        assert np.isnan(no_variance[0])
        assert empty.shape == (3,)
        assert np.isnan(empty[1:]).all()
        assert np.isnan(all_dropped).all()

    def test_negative_codes_and_nan_observations_are_dropped(self):
        negative = grouped_pearson(np.array([-1, 0, 0, 0]), 1, np.array([99.0, 1, 2, 3]), np.array([-99.0, 1, 2, 3]))
        nan = grouped_pearson(
            np.zeros(5, dtype=int), 1, np.array([np.nan, 1.0, 2.0, 3.0, 4.0]), np.array([0.0, 1.0, 2.0, 3.0, np.nan])
        )
        below_minimum = grouped_pearson(
            np.zeros(3, dtype=int), 1, np.array([1.0, np.nan, np.nan]), np.array([1.0, 2, 3])
        )

        assert negative[0] == pytest.approx(1.0)
        assert nan[0] == pytest.approx(1.0)
        assert np.isnan(below_minimum[0])

    def test_a_higher_min_count_filters_small_groups(self):
        codes = np.zeros(3, dtype=int)
        x = np.array([1.0, 2.0, 3.0])

        assert np.isfinite(grouped_pearson(codes, 1, x, x, min_count=3)[0])
        assert np.isnan(grouped_pearson(codes, 1, x, x, min_count=4)[0])


class TestConstantDetection:
    @pytest.mark.parametrize(
        ("x", "is_nan"),
        [
            pytest.param(np.full(50, 1234.5678901), True, id="large-constant"),
            pytest.param(5.0 + np.array([0.0, CONSTANT_ATOL / 10] * 25), True, id="within-tolerance"),
            pytest.param(1e5 + np.array([0.0, 0.5] * 25), True, id="half-unit-at-1e5"),
            pytest.param(1e5 + np.array([0.0, 2.0] * 25), False, id="two-units-at-1e5"),
        ],
    )
    def test_the_constant_rule(self, x, is_nan):
        assert np.isnan(grouped_pearson(np.zeros(50, dtype=int), 1, x, np.arange(50.0))[0]) is np.bool_(is_nan)

    def test_each_group_is_judged_on_its_own_range(self):
        codes = np.array([0, 0, 1, 1, 2, 2])
        x = np.array([0.0, 5e-7, 0.0, 2e-6, 1e5, 1e5 + 0.5])

        result = grouped_pearson(codes, 3, x, np.array([0.0, 1.0] * 3))

        assert np.isnan(result).tolist() == [True, False, True]

    def test_an_empty_group_does_not_vary(self):
        sums = PearsonSums(2)
        sums.add(np.array([0, 0]), np.array([1.0, 2.0]), np.array([3.0, 5.0]))

        assert sums.varies().tolist() == [True, False]
        assert sums.y_varies().tolist() == [True, False]


SERIES = {
    "constant": np.full(6, 3.0),
    "zero_and_5e-7": np.array([0.0, 5e-7] * 3),
    "zero_and_2e-6": np.array([0.0, 2e-6] * 3),
    "1e5_offset_tiny_range": 1e5 + np.array([0.0, 1e-7] * 3),
    "1e5_offset_half_range": 1e5 + np.array([0.0, 0.5] * 3),
    "1e5_offset_wide_range": 1e5 + np.array([0.0, 5.0] * 3),
    "negative_1e5_offset": -1e5 + np.array([0.0, 0.5] * 3),
    "varying": np.arange(6.0),
}


@pytest.mark.parametrize("name", SERIES)
def test_grouped_and_pooled_agree_on_which_series_is_constant_in_any_order(name):
    series, varying = SERIES[name], np.arange(6.0)
    codes = np.zeros(6, dtype=int)
    rng = np.random.default_rng(0)

    assert np.isnan(grouped_pearson(codes, 1, series, varying)[0]) == is_constant(series)
    assert np.isnan(grouped_pearson(codes, 1, varying, series)[0]) == is_constant(series)
    assert (pearson(series, varying) == 0.0) == is_constant(series)
    assert np.isnan(pearson(varying, series)) == is_constant(series)
    for _ in range(10):
        order = rng.permutation(6)
        assert np.isnan(grouped_pearson(codes, 1, series[order], varying[order])[0]) == is_constant(series)


def test_streamed_folds_equal_one_pooled_batch_and_counts_are_per_group():
    rng = np.random.default_rng(2)
    codes = rng.integers(0, 3, size=60)
    x, y = rng.normal(size=60), rng.normal(size=60)
    streamed = PearsonSums(3)
    streamed.add(codes[:30], x[:30], y[:30])
    streamed.add(codes[30:], x[30:], y[30:])

    np.testing.assert_allclose(streamed.correlations(), grouped_pearson(codes, 3, x, y))
    np.testing.assert_array_equal(streamed.counts, np.bincount(codes, minlength=3))
