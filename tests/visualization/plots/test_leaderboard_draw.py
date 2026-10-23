"""Tests for :mod:`drevalpy.visualization.plots._leaderboard_draw`.

The panel-level behaviour (labels, titles, ranking) is pinned through the leaderboard in
``test_leaderboard.py``; this file holds the pure helpers.
"""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.visualization.plots._leaderboard_draw import (
    COMPETITOR_COLOR,
    axis_bounds,
    get_bar_color,
    get_test_mode_name,
    gradient_char_colors,
)


class TestGetBarColor:
    @pytest.mark.parametrize(
        ("rank", "expected"),
        [
            pytest.param(0, "#F4D03F", id="gold"),
            pytest.param(1, "#BDC3C7", id="silver"),
            pytest.param(2, "#E67E22", id="bronze"),
        ],
    )
    def test_top_three_get_medal_colors_at_full_opacity(self, rank, expected):
        assert get_bar_color(rank, False) == {"color": expected, "alpha": 1.0}

    def test_fourth_place_onwards_gets_the_competitor_color(self):
        assert get_bar_color(3, False) == {"color": COMPETITOR_COLOR, "alpha": 0.85}

    def test_baselines_are_grey_regardless_of_rank(self):
        assert get_bar_color(0, True) == get_bar_color(9, True) == {"color": "#5a5a5a", "alpha": 1.0}


class TestGradientCharColors:
    def test_returns_one_color_per_character(self):
        assert len(gradient_char_colors("DrEval")) == len("DrEval")

    def test_every_color_is_a_six_digit_hex_string(self):
        assert all(len(c) == 7 and c.startswith("#") for c in gradient_char_colors("Leaderboard"))

    def test_gradient_runs_from_teal_to_purple(self):
        colors = gradient_char_colors("Leaderboard")

        assert (colors[0], colors[-1]) == ("#14b8a6", "#9d4edd")

    def test_single_character_avoids_a_zero_division(self):
        assert gradient_char_colors("X") == ["#14b8a6"]

    def test_empty_title_yields_no_colors(self):
        assert gradient_char_colors("") == []


class TestGetTestModeName:
    @pytest.mark.parametrize(
        ("mode", "expected"),
        [
            pytest.param("LSO", "5-Fold Leave-Sample-Out Cross Validation", id="lso"),
            pytest.param("LDO", "5-Fold Leave-Drug-Out Cross Validation", id="ldo"),
            pytest.param("LPO", "5-Fold Leave-Pair-Out Cross Validation", id="lpo"),
            pytest.param("LTO", "5-Fold Leave-Tissue-Out Cross Validation", id="lto"),
            pytest.param("LStO", "Leave-Study-Out Cross Validation", id="lsto"),
        ],
    )
    def test_known_modes_are_spelled_out(self, mode, expected):
        assert get_test_mode_name(mode, 5) == expected

    def test_the_fold_count_comes_from_the_argument(self):
        assert get_test_mode_name("LPO", 3) == "3-Fold Leave-Pair-Out Cross Validation"

    def test_unknown_modes_pass_through_unchanged(self):
        assert get_test_mode_name("CROSS_STUDY", 5) == "CROSS_STUDY"


class TestAxisBounds:
    """An all-NaN metric column used to take the whole report down."""

    def test_bounds_stay_finite_when_every_value_is_nan(self):
        left, right = axis_bounds(np.array([np.nan, np.nan]), np.array([np.nan, np.nan]))

        assert np.isfinite([left, right]).all()
        assert left < right

    def test_a_nan_standard_deviation_is_treated_as_zero(self):
        assert axis_bounds(np.array([1.0]), np.array([np.nan])) == axis_bounds(np.array([1.0]), np.array([0.0]))

    def test_the_upper_bound_clears_the_tallest_bar(self):
        _, right = axis_bounds(np.array([0.2, 0.8]), np.array([0.0, 0.1]))

        assert right > 0.9

    def test_negative_values_are_inside_the_axis(self):
        """A normalized correlation below the reference model is negative."""
        left, right = axis_bounds(np.array([-0.4, 0.3]), np.array([0.0, 0.0]))

        assert left < -0.4
        assert right > 0.3

    def test_a_single_zero_value_still_yields_an_ordered_axis(self):
        left, right = axis_bounds(np.array([0.0]), np.array([0.0]))

        assert left < right
