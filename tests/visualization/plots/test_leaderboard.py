"""Tests for the leaderboard plot: its table, its drawing helpers and the ``LeaderboardVisualization`` itself."""

from __future__ import annotations

import dataclasses
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from drevalpy.evaluation._scoring import run_metrics
from drevalpy.types.results.experiment import ExperimentResult
from drevalpy.types.results.run import RunResult
from drevalpy.visualization.plots._leaderboard_draw import (
    COMPETITOR_COLOR,
    DARK_THEME,
    axis_bounds,
    get_bar_color,
    get_test_mode_name,
    gradient_char_colors,
)
from drevalpy.visualization.plots._leaderboard_table import build_leaderboard_df
from drevalpy.visualization.plots.leaderboard import LeaderboardVisualization, _figure_geometry, _panels
from tests.synthetic import REFERENCE_MODEL, make_experiment_result, make_run_result

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(autouse=True)
def _isolate_matplotlib_state():
    with plt.rc_context():
        yield
    plt.close("all")


@pytest.fixture
def computed(experiment) -> LeaderboardVisualization:
    plot = LeaderboardVisualization()
    plot.compute(experiment)
    return plot


def _visible_axes(fig) -> list:
    return [ax for ax in fig.axes if ax.axison]


class TestFigureGeometry:
    def test_a_small_experiment_keeps_the_original_canvas(self):
        height, font_adder, _ = _figure_geometry(12)

        assert (height, font_adder) == (12.0, 6)

    def test_large_experiments_get_a_taller_canvas_a_smaller_font_and_a_capped_height(self):
        assert _figure_geometry(96)[0] > _figure_geometry(20)[0] > 12.0 - 1e-9
        assert _figure_geometry(96)[0] / 96 > 0.25
        assert _figure_geometry(96)[1] < _figure_geometry(40)[1] < _figure_geometry(10)[1]
        assert _figure_geometry(10_000)[0] == 60.0


class TestPanels:
    @staticmethod
    def _df(**columns):
        base = {"PCC": [0.1], "RMSE": [1.0], "Pearson": [np.nan], "Pearson_per_drug": [np.nan]}
        return pd.DataFrame({**base, **{k: [v] for k, v in columns.items()}})

    def test_panel_columns_follow_what_is_available(self):
        both = self._df(Pearson=0.8, Pearson_per_drug=0.3)

        assert [p.column for p in _panels(self._df(), normalized=False)] == ["PCC", "RMSE"]
        assert [p.column for p in _panels(self._df(Pearson=0.8), normalized=False)] == ["PCC", "RMSE"]
        assert [p.column for p in _panels(self._df(Pearson=0.8), normalized=True)] == ["PCC", "RMSE", "Pearson"]
        assert [p.column for p in _panels(both, normalized=True)] == ["PCC", "RMSE", "Pearson", "Pearson_per_drug"]

    def test_the_pcc_panel_names_its_normalization(self):
        assert _panels(self._df(), normalized=True)[0].xlabel == "Normalized PCC"
        assert _panels(self._df(), normalized=False)[0].xlabel == "PCC"


class TestFourPanels:
    @pytest.fixture
    def plot(self, experiment):
        plot = LeaderboardVisualization()
        plot.compute(experiment.normalize(REFERENCE_MODEL))
        return plot

    def test_a_normalized_experiment_draws_four_titled_panels_on_two_rows(self, plot):
        axes = _visible_axes(plot._fig)

        assert [ax.get_xlabel() for ax in axes] == [
            "Normalized PCC",
            "Root Mean Square Error",
            "Raw PCC",
            "Per-drug Pearson",
        ]
        assert axes[2].get_title() == "Raw PCC  \u2191  higher is better"
        assert axes[3].get_title() == "Per-drug Pearson  \u2191  higher is better"
        assert "Normalized Pearson" in axes[0].get_title()
        assert plot._fig.get_size_inches()[1] == pytest.approx(24.0)

    def test_the_reference_model_is_ranked_and_raw_pearson_ranks_by_raw_pearson(self, plot, experiment):
        raw = {m.model_name: np.mean([r.metrics["Pearson"] for r in m.runs]) for m in experiment.models}

        assert REFERENCE_MODEL in {label.get_text() for label in plot._fig.axes[0].get_yticklabels()}
        labels = [label.get_text() for label in plot._fig.axes[2].get_yticklabels()]
        assert labels == sorted(raw, key=raw.__getitem__, reverse=True)

    def test_two_panels_use_a_single_row(self):
        result = ExperimentResult(
            [
                make_run_result(
                    model_name=name, fold_index=i, n_pairs=1, n_drugs=1, metrics={"Pearson": 0.5, "RMSE": 1.0}
                )
                for name in ("A", "B")
                for i in range(2)
            ]
        )
        plot = LeaderboardVisualization()

        plot.compute(result)

        assert len(plot._fig.axes) == 2
        assert plot._fig.get_size_inches()[1] == pytest.approx(12.0)


class TestCompute:
    def test_a_plain_experiment_draws_three_directional_panels_with_every_model_labelled(self, computed, experiment):
        axes = _visible_axes(computed._fig)

        assert [ax.get_xlabel() for ax in axes] == ["PCC", "Root Mean Square Error", "Per-drug Pearson"]
        assert "higher is better" in axes[0].get_title()
        assert "lower is better" in axes[1].get_title()
        assert {label.get_text() for label in axes[0].get_yticklabels()} == set(experiment.model_names)
        assert computed._result is experiment

    def test_models_outside_the_podium_keep_the_default_label_color(self):
        plot = LeaderboardVisualization()

        plot.compute(make_experiment_result(n_models=5, n_folds=2))

        colors = [label.get_color() for label in plot._fig.axes[0].get_yticklabels()]
        assert DARK_THEME["text"] in colors

    def test_falls_back_to_a_placeholder_when_there_is_no_data(self):
        plot = LeaderboardVisualization()

        plot.compute(ExperimentResult([make_run_result(randomization=("gene_expression", "permutation"))]))

        assert len(plot._fig.axes) == 1
        assert plot._fig.axes[0].texts[0].get_text() == "No data available for leaderboard"

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

        assert {label.get_text() for label in plot._fig.axes[0].get_yticklabels()} == {"WithPcc"}
        assert {label.get_text() for label in plot._fig.axes[1].get_yticklabels()} == {"WithPcc", "NoPcc"}

    def test_skips_with_a_warning_when_no_metric_has_a_finite_value(self, caplog):
        result = ExperimentResult(
            [make_run_result(model_name="A", fold_index=i, metrics={"MSE": 0.5}) for i in range(2)]
        )
        plot = LeaderboardVisualization()

        with caplog.at_level(logging.WARNING, logger="drevalpy.visualization.plots.leaderboard"):
            plot.compute(result)

        assert plot._fig.axes[0].texts[0].get_text() == "No data available for leaderboard"
        assert any("no finite Pearson/RMSE values" in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize(
        "split_mode, n_folds, subtitle",
        [
            ("LSO", 3, "3-Fold Leave-Sample-Out Cross Validation"),
            ("LSO", 5, "5-Fold Leave-Sample-Out Cross Validation"),
            ("LStO", 3, "Leave-Study-Out Cross Validation"),
        ],
    )
    def test_the_subtitle_names_the_split_and_the_fold_count_where_it_applies(self, split_mode, n_folds, subtitle):
        plot = LeaderboardVisualization()

        plot.compute(make_experiment_result(n_models=3, n_folds=n_folds, split_mode=split_mode))

        texts = [t.get_text() for t in plot._fig.texts]
        assert any(t.endswith(subtitle) for t in texts)
        if split_mode == "LStO":
            assert not any("-Fold" in t for t in texts)

    def test_a_large_experiment_gets_a_taller_canvas(self):
        plot = LeaderboardVisualization()

        plot.compute(make_experiment_result(n_models=40, n_folds=2))

        assert plot._fig.get_size_inches()[1] > 12.0

    def test_mutates_the_global_rcparams_with_the_dark_theme(self, experiment):
        plt.rcParams["figure.facecolor"] = "white"

        LeaderboardVisualization().compute(experiment)

        assert plt.rcParams["figure.facecolor"] == DARK_THEME["background"]


class TestRenderingAndGuards:
    def test_to_png_writes_a_png_file(self, computed, tmp_path):
        computed.to_png(tmp_path / "leaderboard.png")

        assert (tmp_path / "leaderboard.png").read_bytes().startswith(PNG_MAGIC)

    def test_to_multiqc_embeds_the_figure_under_the_registry_name(self, computed):
        (section,) = computed.to_multiqc()

        assert (section.name, section.anchor) == ("leaderboard", "leaderboard")
        assert "data:image/png;base64," in section.content


COLUMNS = [
    "algorithm",
    "PCC",
    "PCC_std",
    "RMSE",
    "RMSE_std",
    "Pearson",
    "Pearson_std",
    "Pearson_per_drug",
    "Pearson_per_drug_std",
    "is_baseline",
]


def _run(model: str, fold: int, seed: int, *, noise: float = 0.5) -> RunResult:
    rng = np.random.default_rng(seed)
    drugs = np.repeat([f"D{i}" for i in range(4)], 6).astype(object)
    samples = np.tile([f"S{i}" for i in range(6)], 4).astype(object)
    truth = rng.normal(size=24) + np.repeat([0.0, 2.0, 4.0, 6.0], 6)
    predictions = truth + rng.normal(scale=noise, size=24)
    return RunResult(
        model_name=model,
        dataset_name="D",
        fold_index=fold,
        predictions=predictions,
        ground_truth=truth,
        sample_ids=samples,
        drug_ids=drugs,
        split_mode="LSO",
        fold_id=f"fold_{fold}",
        metrics=run_metrics(predictions, truth),
    )


def _per_drug_pearson(run: RunResult) -> float:
    return float(
        np.mean(
            [
                np.corrcoef(run.predictions[run.drug_ids == drug], run.ground_truth[run.drug_ids == drug])[0, 1]
                for drug in np.unique(run.drug_ids)
            ]
        )
    )


@pytest.fixture
def noisy_experiment() -> ExperimentResult:
    runs = [
        _run(name, fold, seed=10 * index + fold, noise=noise)
        for index, (name, noise) in enumerate([(REFERENCE_MODEL, 2.0), ("ElasticNet", 1.0), ("RandomForest", 0.4)])
        for fold in range(3)
    ]
    return ExperimentResult(runs)


def _runs_of(noisy_experiment: ExperimentResult, name: str) -> list[RunResult]:
    return next(m for m in noisy_experiment.models if m.model_name == name).runs


class TestUnnormalized:
    def test_schema_ordering_and_baseline_flags(self, noisy_experiment):
        df = build_leaderboard_df(noisy_experiment)

        assert list(df.columns) == COLUMNS
        assert len(df) == len(noisy_experiment.models)
        assert df["PCC"].is_monotonic_decreasing
        assert df["Pearson"].tolist() == df["PCC"].tolist()
        flags = df.set_index("algorithm")["is_baseline"]
        assert bool(flags[REFERENCE_MODEL]) is True
        assert bool(flags["ElasticNet"]) is False

    def test_means_and_sample_std_are_taken_over_folds(self, noisy_experiment):
        row = build_leaderboard_df(noisy_experiment).set_index("algorithm").loc["ElasticNet"]
        runs = _runs_of(noisy_experiment, "ElasticNet")
        pearson = [r.metrics["Pearson"] for r in runs]

        assert row["PCC"] == pytest.approx(np.mean(pearson))
        assert row["PCC_std"] == pytest.approx(np.std(pearson, ddof=1))
        assert row["RMSE"] == pytest.approx(np.mean([r.metrics["RMSE"] for r in runs]))

    def test_per_drug_pearson_averages_the_drugs_then_the_folds(self, noisy_experiment):
        row = build_leaderboard_df(noisy_experiment).set_index("algorithm").loc["RandomForest"]
        per_fold = [_per_drug_pearson(r) for r in _runs_of(noisy_experiment, "RandomForest")]

        assert row["Pearson_per_drug"] == pytest.approx(np.mean(per_fold))
        assert row["Pearson_per_drug_std"] == pytest.approx(np.std(per_fold, ddof=1))

    def test_single_fold_models_get_a_zero_standard_deviation(self):
        df = build_leaderboard_df(ExperimentResult([_run("Solo", 0, seed=1)]))

        assert df.filter(like="_std").iloc[0].tolist() == [0.0] * 4

    def test_a_constant_prediction_within_every_drug_scores_zero_per_drug(self):
        run = _run("Constant", 0, seed=2)
        run.predictions = np.repeat(run.ground_truth[::6], 6)

        df = build_leaderboard_df(ExperimentResult([run]))

        assert df["Pearson_per_drug"].iloc[0] == 0.0
        assert df["Pearson_per_drug_std"].iloc[0] == 0.0

    def test_randomized_runs_are_excluded_and_all_randomized_gives_the_empty_schema(self):
        randomized = make_run_result(model_name="RandomForest", randomization=("gene_expression", "permutation"))
        result = ExperimentResult(
            [
                make_run_result(model_name="ElasticNet", fold_index=0),
                make_run_result(model_name="ElasticNet", fold_index=1),
                randomized,
            ]
        )

        assert build_leaderboard_df(result)["algorithm"].tolist() == ["ElasticNet"]
        empty = build_leaderboard_df(ExperimentResult([randomized]))
        assert empty.empty
        assert list(empty.columns) == COLUMNS

    def test_reads_the_plain_pearson_key_and_a_missing_metric_becomes_nan(self):
        result = ExperimentResult([make_run_result(metrics={"Pearson": 0.9, "RMSE": 1.0})])
        missing = ExperimentResult([make_run_result(metrics={"MSE": 0.5})])

        assert build_leaderboard_df(result)["PCC"].tolist() == [pytest.approx(0.9)]
        assert build_leaderboard_df(missing)["PCC"].isna().all()


class TestNormalized:
    """After ``normalize()`` the runs hold residuals; RMSE and raw Pearson come from ``raw_metrics``."""

    @pytest.fixture
    def df(self, noisy_experiment):
        return build_leaderboard_df(noisy_experiment.normalize(REFERENCE_MODEL)).set_index("algorithm")

    def test_pcc_is_normalized_while_rmse_and_raw_pearson_are_unnormalized(self, df, noisy_experiment):
        normalized_runs = _runs_of(noisy_experiment.normalize(REFERENCE_MODEL), "ElasticNet")
        runs = _runs_of(noisy_experiment, "ElasticNet")

        assert df.loc["ElasticNet", "PCC"] == pytest.approx(np.mean([r.metrics["Pearson"] for r in normalized_runs]))
        assert df.loc["ElasticNet", "RMSE"] == pytest.approx(np.mean([r.metrics["RMSE"] for r in runs]))
        assert df.loc["ElasticNet", "Pearson"] == pytest.approx(np.mean([r.metrics["Pearson"] for r in runs]))
        assert df["PCC"].notna().all()

    def test_per_drug_pearson_is_of_the_unnormalized_predictions(self, df, noisy_experiment):
        for name in ("RandomForest", REFERENCE_MODEL):
            expected = np.mean([_per_drug_pearson(r) for r in _runs_of(noisy_experiment, name)])
            assert df.loc[name, "Pearson_per_drug"] == pytest.approx(expected)

    def test_the_reference_model_is_a_baseline_with_zero_normalized_pcc_and_its_own_rmse(self, df, noisy_experiment):
        reference = df.loc[REFERENCE_MODEL]
        runs = _runs_of(noisy_experiment, REFERENCE_MODEL)

        assert bool(reference["is_baseline"]) is True
        assert reference["PCC"] == reference["PCC_std"] == 0.0
        assert reference["RMSE"] == pytest.approx(np.mean([r.metrics["RMSE"] for r in runs]))

    def test_runs_without_raw_metrics_leave_raw_pearson_and_per_drug_empty(self, noisy_experiment):
        normalized = noisy_experiment.normalize(REFERENCE_MODEL)
        for model in normalized.models:
            model.runs = [dataclasses.replace(r, raw_metrics={}, raw_per_drug_pearson=None) for r in model.runs]
        normalized.reference_runs = ()

        df = build_leaderboard_df(normalized)

        assert df["Pearson"].isna().all()
        assert df["Pearson_per_drug"].isna().all()
        assert df["RMSE"].notna().all()


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
