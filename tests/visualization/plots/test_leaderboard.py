"""Tests for :mod:`drevalpy.visualization.plots.leaderboard`.

``_create_figure`` updates the global ``plt.rcParams`` with the dark theme and
never restores it. That is asserted as current behaviour, and every test in this
module runs inside an ``rc_context`` so the mutation cannot leak into the rest
of the suite.

The PCC panel is the plot that broke when it hard-coded ``"Pearson: normalized"``
while ``normalize()`` emits ``"Pearson"``: the column came out all-NaN and
``set_xlim`` raised ``ValueError: Axis limits cannot be NaN or Inf``. Both halves
of that failure are pinned - the metric-name resolution and the NaN-safe axis.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pytest

from drevalpy.types.results.experiment import ExperimentResult
from drevalpy.visualization.plots.leaderboard import (
    COMPETITOR_COLOR,
    DARK_THEME,
    LeaderboardVisualization,
    _axis_bounds,
    _build_leaderboard_df,
    _figure_geometry,
    _get_bar_color,
    _get_test_mode_name,
    _gradient_char_colors,
)
from tests.synthetic import NORMALIZED_METRIC, REFERENCE_MODEL, make_experiment_result, make_run_result

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(autouse=True)
def _isolate_matplotlib_state():
    """Contain the rcParams mutation in ``_create_figure`` and close figures."""
    with plt.rc_context():
        yield
    plt.close("all")


@pytest.fixture
def experiment() -> ExperimentResult:
    return make_experiment_result()


@pytest.fixture
def computed(experiment) -> LeaderboardVisualization:
    plot = LeaderboardVisualization()
    plot.compute(experiment)
    return plot


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
        assert _get_bar_color(rank, False) == {"color": expected, "alpha": 1.0}

    def test_fourth_place_onwards_gets_the_competitor_color(self):
        assert _get_bar_color(3, False) == {"color": COMPETITOR_COLOR, "alpha": 0.85}

    def test_baselines_are_grey_regardless_of_rank(self):
        assert _get_bar_color(0, True) == _get_bar_color(9, True) == {"color": "#5a5a5a", "alpha": 1.0}


class TestGradientCharColors:
    def test_returns_one_color_per_character(self):
        assert len(_gradient_char_colors("DrEval")) == len("DrEval")

    def test_every_color_is_a_six_digit_hex_string(self):
        assert all(len(c) == 7 and c.startswith("#") for c in _gradient_char_colors("Leaderboard"))

    def test_gradient_runs_from_teal_to_purple(self):
        colors = _gradient_char_colors("Leaderboard")

        assert (colors[0], colors[-1]) == ("#14b8a6", "#9d4edd")

    def test_single_character_avoids_a_zero_division(self):
        assert _gradient_char_colors("X") == ["#14b8a6"]

    def test_empty_title_yields_no_colors(self):
        assert _gradient_char_colors("") == []


class TestGetTestModeName:
    @pytest.mark.parametrize(
        ("mode", "expected"),
        [
            pytest.param("LCO", "10-Fold Leave-Cell-Out Cross Validation", id="lco"),
            pytest.param("LDO", "10-Fold Leave-Drug-Out Cross Validation", id="ldo"),
            pytest.param("LPO", "10-Fold Leave-Pair-Out Cross Validation", id="lpo"),
            pytest.param("LTO", "10-Fold Leave-Tissue-Out Cross Validation", id="lto"),
        ],
    )
    def test_known_modes_are_spelled_out(self, mode, expected):
        assert _get_test_mode_name(mode) == expected

    def test_unknown_modes_pass_through_unchanged(self):
        assert _get_test_mode_name("CROSS_STUDY") == "CROSS_STUDY"


class TestBuildLeaderboardDf:
    def test_has_one_row_per_model(self, experiment):
        df = _build_leaderboard_df(experiment)

        assert len(df) == len(experiment.models)

    def test_columns_are_the_aggregated_schema(self, experiment):
        df = _build_leaderboard_df(experiment)

        assert list(df.columns) == ["algorithm", "PCC", "PCC_std", "RMSE", "RMSE_std", "is_baseline"]

    def test_rows_are_ordered_by_descending_normalized_pearson(self, experiment):
        df = _build_leaderboard_df(experiment)

        assert df["PCC"].is_monotonic_decreasing

    def test_naive_models_are_flagged_as_baselines(self, experiment):
        df = _build_leaderboard_df(experiment).set_index("algorithm")

        assert bool(df.loc["NaiveMeanEffectsPredictor", "is_baseline"]) is True
        assert bool(df.loc["ElasticNet", "is_baseline"]) is False

    def test_single_fold_models_get_a_zero_standard_deviation(self):
        result = ExperimentResult([make_run_result(model_name="Solo")])

        df = _build_leaderboard_df(result)

        assert (df["PCC_std"].iloc[0], df["RMSE_std"].iloc[0]) == (0.0, 0.0)

    def test_randomized_runs_are_excluded(self):
        result = ExperimentResult(
            [
                make_run_result(model_name="ElasticNet", fold_index=0),
                make_run_result(model_name="ElasticNet", fold_index=1),
                make_run_result(model_name="RandomForest", randomization=("gene_expression", "permutation")),
            ]
        )

        df = _build_leaderboard_df(result)

        assert df["algorithm"].tolist() == ["ElasticNet"]

    def test_returns_the_empty_schema_when_every_run_is_randomized(self):
        result = ExperimentResult([make_run_result(randomization=("gene_expression", "permutation"))])

        df = _build_leaderboard_df(result)

        assert df.empty
        assert list(df.columns) == ["algorithm", "PCC", "PCC_std", "RMSE", "RMSE_std", "is_baseline"]


class TestMetricNameContract:
    """``normalize()`` emits plain metric names; the panel must read those."""

    def test_a_normalized_experiment_yields_finite_pcc_values(self):
        normalized = make_experiment_result(n_models=4, n_folds=3).normalize(REFERENCE_MODEL)

        df = _build_leaderboard_df(normalized)

        assert df["PCC"].notna().all()

    def test_the_reference_model_is_gone_from_the_ranking(self):
        normalized = make_experiment_result(n_models=4, n_folds=3).normalize(REFERENCE_MODEL)

        df = _build_leaderboard_df(normalized)

        assert REFERENCE_MODEL not in df["algorithm"].tolist()

    def test_the_legacy_suffixed_spelling_is_still_read(self):
        """Results serialized by older releases only carry the suffixed key."""
        result = ExperimentResult(
            [make_run_result(fold_index=i, metrics={NORMALIZED_METRIC: 0.4 + 0.1 * i, "RMSE": 1.0}) for i in range(2)]
        )

        df = _build_leaderboard_df(result)

        assert df["PCC"].tolist() == [pytest.approx(0.45)]

    def test_the_plain_name_wins_when_both_are_present(self):
        result = ExperimentResult([make_run_result(metrics={"Pearson": 0.9, NORMALIZED_METRIC: 0.1, "RMSE": 1.0})])

        df = _build_leaderboard_df(result)

        assert df["PCC"].tolist() == [pytest.approx(0.9)]

    def test_a_metric_no_run_reports_becomes_nan_rather_than_raising(self):
        result = ExperimentResult([make_run_result(metrics={"MSE": 0.5})])

        df = _build_leaderboard_df(result)

        assert df["PCC"].isna().all()


class TestFigureGeometry:
    """The 96-model production report overlapped every tick label at a fixed height."""

    def test_a_small_experiment_keeps_the_original_canvas(self):
        height, font_adder, _ = _figure_geometry(12)

        assert (height, font_adder) == (12.0, 6)

    def test_height_grows_with_the_model_count(self):
        assert _figure_geometry(96)[0] > _figure_geometry(20)[0] > 12.0 - 1e-9

    def test_every_model_gets_at_least_a_quarter_inch_of_height(self):
        height, _, _ = _figure_geometry(96)

        assert height / 96 > 0.25

    def test_the_font_shrinks_as_the_list_gets_long(self):
        assert _figure_geometry(96)[1] < _figure_geometry(40)[1] < _figure_geometry(10)[1]

    def test_height_is_capped_so_the_png_stays_writable(self):
        assert _figure_geometry(10_000)[0] == 60.0


class TestAxisBounds:
    """An all-NaN metric column used to take the whole report down."""

    def test_bounds_stay_finite_when_every_value_is_nan(self):
        left, right = _axis_bounds(np.array([np.nan, np.nan]), np.array([np.nan, np.nan]))

        assert np.isfinite([left, right]).all()
        assert left < right

    def test_a_nan_standard_deviation_is_treated_as_zero(self):
        assert _axis_bounds(np.array([1.0]), np.array([np.nan])) == _axis_bounds(np.array([1.0]), np.array([0.0]))

    def test_the_upper_bound_clears_the_tallest_bar(self):
        _, right = _axis_bounds(np.array([0.2, 0.8]), np.array([0.0, 0.1]))

        assert right > 0.9

    def test_negative_values_are_inside_the_axis(self):
        """A normalized correlation below the reference model is negative."""
        left, right = _axis_bounds(np.array([-0.4, 0.3]), np.array([0.0, 0.0]))

        assert left < -0.4
        assert right > 0.3

    def test_a_single_zero_value_still_yields_an_ordered_axis(self):
        left, right = _axis_bounds(np.array([0.0]), np.array([0.0]))

        assert left < right


class TestCompute:
    def test_builds_two_ranked_panels(self, computed):
        assert len(computed._fig.axes) == 2

    def test_left_panel_ranks_by_pearson(self, computed):
        """The experiment is not normalized, so the label must not claim it is."""
        assert computed._fig.axes[0].get_xlabel() == "PCC"

    def test_left_panel_says_normalized_once_it_is(self, experiment):
        plot = LeaderboardVisualization()

        plot.compute(experiment.normalize(REFERENCE_MODEL))

        assert plot._fig.axes[0].get_xlabel() == "Normalized PCC"
        assert "Normalized Pearson" in plot._fig.axes[0].get_title()

    def test_right_panel_ranks_by_rmse(self, computed):
        assert computed._fig.axes[1].get_xlabel() == "Root Mean Square Error"

    def test_every_model_gets_a_tick_label(self, computed, experiment):
        labels = {label.get_text() for label in computed._fig.axes[0].get_yticklabels()}

        assert labels == set(experiment.model_names)

    def test_panels_are_titled_with_the_optimization_direction(self, computed):
        titles = [ax.get_title() for ax in computed._fig.axes]

        assert "higher is better" in titles[0]
        assert "lower is better" in titles[1]

    def test_stores_the_result(self, computed, experiment):
        assert computed._result is experiment

    def test_models_outside_the_podium_keep_the_default_label_color(self):
        experiment = make_experiment_result(n_models=5, n_folds=2)
        plot = LeaderboardVisualization()

        plot.compute(experiment)

        colors = [label.get_color() for label in plot._fig.axes[0].get_yticklabels()]
        assert DARK_THEME["text"] in colors

    def test_falls_back_to_a_placeholder_when_there_is_no_data(self):
        result = ExperimentResult([make_run_result(randomization=("gene_expression", "permutation"))])
        plot = LeaderboardVisualization()

        plot.compute(result)

        assert len(plot._fig.axes) == 1
        assert plot._fig.axes[0].texts[0].get_text() == "No data available for leaderboard"

    def test_a_normalized_experiment_renders_both_panels(self, experiment):
        """The regression: this used to raise ``Axis limits cannot be NaN or Inf``."""
        plot = LeaderboardVisualization()

        plot.compute(experiment.normalize(REFERENCE_MODEL))

        assert len(plot._fig.axes) == 2

    def test_models_without_a_pcc_value_are_dropped_from_that_panel_only(self):
        result = ExperimentResult(
            [
                make_run_result(model_name="WithPcc", fold_index=i, metrics={"Pearson": 0.5, "RMSE": 1.0})
                for i in range(2)
            ]
            + [make_run_result(model_name="NoPcc", fold_index=i, metrics={"RMSE": 2.0}) for i in range(2)]
        )
        plot = LeaderboardVisualization()

        plot.compute(result)

        pcc_labels = {label.get_text() for label in plot._fig.axes[0].get_yticklabels()}
        rmse_labels = {label.get_text() for label in plot._fig.axes[1].get_yticklabels()}
        assert pcc_labels == {"WithPcc"}
        assert rmse_labels == {"WithPcc", "NoPcc"}

    def test_skips_with_a_warning_when_no_metric_has_a_finite_value(self, caplog):
        result = ExperimentResult(
            [make_run_result(model_name="A", fold_index=i, metrics={"MSE": 0.5}) for i in range(2)]
        )
        plot = LeaderboardVisualization()

        with caplog.at_level(logging.WARNING, logger="drevalpy.visualization.plots.leaderboard"):
            plot.compute(result)

        assert plot._fig.axes[0].texts[0].get_text() == "No data available for leaderboard"
        assert any("no finite Pearson/RMSE values" in r.getMessage() for r in caplog.records)

    def test_a_large_experiment_gets_a_taller_canvas(self):
        """The 96-model report is the case a fixed 12-inch figure could not render."""
        plot = LeaderboardVisualization()

        plot.compute(make_experiment_result(n_models=40, n_folds=2))

        assert plot._fig.get_size_inches()[1] > 12.0

    def test_dataset_argument_is_accepted_and_ignored(self, experiment):
        plot = LeaderboardVisualization()

        plot.compute(experiment, dataset=object())

        assert plot._fig is not None

    def test_mutates_the_global_rcparams_with_the_dark_theme(self, experiment):
        plt.rcParams["figure.facecolor"] = "white"

        LeaderboardVisualization().compute(experiment)

        assert plt.rcParams["figure.facecolor"] == DARK_THEME["background"]


class TestRendering:
    def test_to_png_writes_a_png_file(self, computed, tmp_path):
        out = tmp_path / "leaderboard.png"

        computed.to_png(out)

        assert out.read_bytes().startswith(PNG_MAGIC)

    def test_to_multiqc_embeds_the_figure_under_the_registry_name(self, computed):
        sections = computed.to_multiqc()

        assert len(sections) == 1
        assert (sections[0].name, sections[0].anchor) == ("leaderboard", "leaderboard")
        assert sections[0].content is not None
        assert "data:image/png;base64," in sections[0].content


class TestGuardsBeforeCompute:
    def test_to_png_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match=r"Call compute\(\) before to_png\(\)"):
            LeaderboardVisualization().to_png(tmp_path / "l.png")

    def test_to_multiqc_raises(self):
        with pytest.raises(RuntimeError, match=r"Call compute\(\) before to_multiqc\(\)"):
            LeaderboardVisualization().to_multiqc()

    def test_show_raises(self):
        with pytest.raises(RuntimeError, match=r"Call compute\(\) before show\(\)"):
            LeaderboardVisualization().show()
