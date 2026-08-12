"""Tests for :mod:`drevalpy.visualization.plots.leaderboard`.

``_create_figure`` updates the global ``plt.rcParams`` with the dark theme and
never restores it. That is asserted as current behaviour, and every test in this
module runs inside an ``rc_context`` so the mutation cannot leak into the rest
of the suite.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from drevalpy.types.results.experiment import ExperimentResult
from drevalpy.visualization.plots.leaderboard import (
    COMPETITOR_COLOR,
    DARK_THEME,
    LeaderboardVisualization,
    _build_leaderboard_df,
    _get_bar_color,
    _get_test_mode_name,
    _gradient_char_colors,
)
from tests.synthetic import make_experiment_result, make_run_result

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


class TestCompute:
    def test_builds_two_ranked_panels(self, computed):
        assert len(computed._fig.axes) == 2

    def test_left_panel_ranks_by_normalized_pearson(self, computed):
        assert computed._fig.axes[0].get_xlabel() == "Normalized PCC"

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
