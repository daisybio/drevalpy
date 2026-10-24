"""Tests for :mod:`drevalpy.visualization.plots._leaderboard_draw` (pure helpers)."""

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
    @pytest.mark.parametrize(("rank", "expected"), [(0, "#F4D03F"), (1, "#BDC3C7"), (2, "#E67E22")])
    def test_top_three_get_medal_colors_at_full_opacity(self, rank, expected):
        assert get_bar_color(rank, False) == {"color": expected, "alpha": 1.0}

    def test_fourth_place_is_a_competitor_and_baselines_are_grey_regardless_of_rank(self):
        assert get_bar_color(3, False) == {"color": COMPETITOR_COLOR, "alpha": 0.85}
        assert get_bar_color(0, True) == get_bar_color(9, True) == {"color": "#5a5a5a", "alpha": 1.0}


class TestGradientCharColors:
    def test_one_hex_color_per_character_running_from_teal_to_purple(self):
        colors = gradient_char_colors("Leaderboard")

        assert len(colors) == len("Leaderboard")
        assert all(len(c) == 7 and c.startswith("#") for c in colors)
        assert (colors[0], colors[-1]) == ("#14b8a6", "#9d4edd")

    def test_single_character_avoids_a_zero_division_and_empty_title_yields_nothing(self):
        assert gradient_char_colors("X") == ["#14b8a6"]
        assert gradient_char_colors("") == []


class TestGetTestModeName:
    @pytest.mark.parametrize(
        ("mode", "expected"),
        [
            ("LSO", "3-Fold Leave-Sample-Out Cross Validation"),
            ("LDO", "3-Fold Leave-Drug-Out Cross Validation"),
            ("LPO", "3-Fold Leave-Pair-Out Cross Validation"),
            ("LTO", "3-Fold Leave-Tissue-Out Cross Validation"),
            ("LStO", "Leave-Study-Out Cross Validation"),
            ("CROSS_STUDY", "CROSS_STUDY"),
        ],
    )
    def test_known_modes_are_spelled_out_with_the_fold_count_and_unknown_ones_pass_through(self, mode, expected):
        assert get_test_mode_name(mode, 3) == expected


class TestAxisBounds:
    def test_bounds_stay_finite_when_every_value_is_nan(self):
        left, right = axis_bounds(np.array([np.nan, np.nan]), np.array([np.nan, np.nan]))

        assert np.isfinite([left, right]).all()
        assert left < right

    def test_a_nan_standard_deviation_is_treated_as_zero(self):
        assert axis_bounds(np.array([1.0]), np.array([np.nan])) == axis_bounds(np.array([1.0]), np.array([0.0]))

    def test_the_axis_clears_the_tallest_bar_and_negative_values(self):
        left, right = axis_bounds(np.array([-0.4, 0.8]), np.array([0.0, 0.1]))

        assert left < -0.4
        assert right > 0.9

    def test_a_single_zero_value_still_yields_an_ordered_axis(self):
        left, right = axis_bounds(np.array([0.0]), np.array([0.0]))

        assert left < right
