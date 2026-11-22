"""Tests for the metric-distribution plots (heatmap, violin) and the shared ``_utils`` helpers."""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import pytest
from plotly.basedatatypes import BaseFigure

from drevalpy.types.results.experiment import ExperimentResult
from drevalpy.visualization.base import show
from drevalpy.visualization.plots._utils import placeholder_figure, runs_frame
from drevalpy.visualization.plots.heatmap import (
    HeatmapPlot,
    _columns_for_setting,
    _ssmd_matrix,
    compute_ssmd,
)
from drevalpy.visualization.plots.violin import ViolinPlot, _violins
from tests.synthetic import REFERENCE_MODEL, make_experiment_result, make_run_result


def _ssmd_frame() -> pd.DataFrame:
    return pd.DataFrame({"algorithm": ["A", "A", "B", "B"], "MSE": [1.0, 2.0, 5.0, 6.0]})


@pytest.fixture(scope="module")
def heatmap(experiment) -> BaseFigure:
    return HeatmapPlot().figure(experiment)


class TestSsmdMatrix:
    def test_returns_an_empty_frame_when_the_metric_is_absent(self):
        assert _ssmd_matrix(_ssmd_frame(), "Pearson").empty

    def test_is_a_square_antisymmetric_matrix_with_a_zero_diagonal(self):
        matrix = _ssmd_matrix(_ssmd_frame(), "MSE")

        assert list(matrix.index) == list(matrix.columns) == ["A", "B"]
        assert matrix.loc["A", "A"] == matrix.loc["B", "B"] == 0.0
        assert matrix.loc["A", "B"] == pytest.approx(-matrix.loc["B", "A"])
        assert matrix.loc["A", "B"] < 0

    def test_ignores_nan_folds(self):
        df = pd.DataFrame({"algorithm": list("AAABBB"), "MSE": [np.nan, 2.0, 4.0, 5.0, 6.0, 7.0]})

        assert _ssmd_matrix(df, "MSE").loc["A", "B"] == pytest.approx(compute_ssmd([2.0, 4.0], [5.0, 6.0, 7.0]))

    def test_is_nan_when_both_models_have_zero_variance(self):
        df = _ssmd_frame()
        df["MSE"] = [1.0, 1.0, 2.0, 2.0]

        assert math.isnan(_ssmd_matrix(df, "MSE").loc["A", "B"])


@pytest.mark.parametrize(
    ("setting", "expected"),
    [
        ("r2", ["R^2"]),
        ("correlations", ["Pearson", "Spearman", "Kendall"]),
        ("errors", ["MSE", "RMSE", "MAE"]),
        ("unknown", []),
    ],
)
def test_columns_for_setting_selects_the_metrics_belonging_to_the_panel(setting, expected):
    metric_cols = ["R^2", "Pearson", "Spearman", "Kendall", "MSE", "RMSE", "MAE"]

    assert _columns_for_setting(setting, metric_cols) == expected


class TestHeatmapFigure:
    def test_builds_five_stacked_panels_with_models_as_rows(self, heatmap, experiment):
        fig = heatmap

        assert [trace.type for trace in fig.data] == ["heatmap"] * 5
        assert set(fig.data[0].y) == set(experiment.model_names)
        assert list(fig.data[0].x) == ["R^2"]
        assert set(fig.data[3].x) == set(fig.data[3].y) == set(experiment.model_names)
        assert "±" in fig.data[0].text[0][0]
        assert fig.layout.height == 500 + len(experiment.models) * 35
        assert fig.layout.width == 1300
        assert fig.layout.title.text == "Heatmap of the evaluation metrics"

    def test_panels_without_available_metrics_are_skipped(self):
        result = ExperimentResult(
            [
                make_run_result(model_name=name, fold_index=i, metrics={"MSE": 0.4 + i + offset})
                for offset, name in enumerate("AB")
                for i in range(2)
            ]
        )
        fig = HeatmapPlot().figure(result)

        panels = [list(trace.x) for trace in fig.data]
        assert panels[0] == ["MSE"]
        assert sorted(panels[1]) == ["A", "B"]
        assert len(panels) == 2

    def test_model_names_with_underscores_stay_whole(self):
        result = ExperimentResult(
            [
                make_run_result(model_name=name, fold_index=i, metrics={"MSE": 0.4 + i + offset})
                for offset, name in enumerate(["Elastic_Net", "Random_Forest"])
                for i in range(2)
            ]
        )
        fig = HeatmapPlot().figure(result)

        assert sorted(fig.data[1].y) == ["Elastic_Net", "Random_Forest"]
        assert sorted(fig.data[1].x) == ["Elastic_Net", "Random_Forest"]

    def test_a_normalized_experiment_populates_every_panel(self):
        fig = HeatmapPlot().figure(make_experiment_result(n_models=3, n_folds=2).normalize(REFERENCE_MODEL))

        assert len(fig.data) == 5

    def test_warns_when_no_expected_metric_is_present(self, caplog):
        with caplog.at_level(logging.WARNING, logger="drevalpy.visualization.plots.heatmap"):
            HeatmapPlot().figure(ExperimentResult([make_run_result(metrics={"NotAMetric": 1.0})]))

        assert any("none of the expected metrics" in r.getMessage() for r in caplog.records)


class TestHeatmapOutputs:
    def test_sections_are_a_single_native_heatmap_section(self, experiment):
        (section,) = HeatmapPlot().sections(experiment)

        assert (section.name, section.anchor) == ("Performance Heatmap", "dreval_heatmap")
        assert section.plot is not None
        assert "Mean metric values per model" in section.description

    def test_show_delegates_to_the_plotly_figure(self, experiment, monkeypatch):
        calls: list = []
        monkeypatch.setattr(BaseFigure, "show", lambda self, *a, **kw: calls.append(self))

        show(HeatmapPlot(), experiment)

        assert [len(fig.data) for fig in calls] == [5]


#: Metrics the plot draws: the seven base metrics.
N_PLOTTED_METRICS = 7


@pytest.fixture(scope="module")
def violin(experiment) -> BaseFigure:
    return ViolinPlot().figure(experiment)


class TestViolinFigure:
    def test_draws_one_violin_per_model_and_metric_holding_one_point_per_fold(self, violin, experiment):
        fig = violin
        trace = fig.data[0]

        assert len(fig.data) == len(experiment.models) * N_PLOTTED_METRICS
        assert {t.type for t in fig.data} == {"violin"}
        assert trace.name == f"{experiment.model_names[0]}: MSE"
        assert len(trace.y) == experiment.models[0].n_folds
        assert trace.box.visible is True
        assert trace.meanline.visible is True
        assert fig.layout.title.text == "All Metrics"
        assert (fig.layout.height, fig.layout.width) == (600, 1100)

    def test_violins_are_keyed_by_model_and_metric_and_hold_one_value_per_run(self, experiment):
        run = experiment.models[0].runs[0]

        violins = _violins(experiment)

        assert len(violins) == len(experiment.models) * N_PLOTTED_METRICS
        assert violins[(run.model_name, "MSE")][run.run_id] == run.metrics["MSE"]
        assert all(len(values) == experiment.models[0].n_folds for values in violins.values())

    def test_model_names_with_underscores_stay_whole(self):
        result = ExperimentResult(
            [
                make_run_result(model_name=name, fold_index=i, metrics={"MSE": 0.4 + i})
                for name in ("Elastic_Net",)
                for i in range(2)
            ]
        )

        assert [trace.name for trace in ViolinPlot().figure(result).data] == ["Elastic_Net: MSE"]

    def test_metrics_absent_from_the_result_are_dropped(self):
        result = ExperimentResult(
            [make_run_result(fold_index=i, metrics={"MSE": 0.5 + i, "Pearson": np.nan}) for i in range(2)]
        )
        fig = ViolinPlot().figure(result)

        assert [trace.name for trace in fig.data] == ["ElasticNet: MSE"]

    def test_skips_cleanly_with_a_warning_when_no_metric_survives(self, caplog):
        result = ExperimentResult([make_run_result(fold_index=i, metrics={"Pearson": np.nan}) for i in range(2)])
        plot = ViolinPlot()

        with caplog.at_level(logging.WARNING, logger="drevalpy.visualization.plots.violin"):
            assert plot.figure(result).data == ()
            assert plot.sections(result) == []

        assert any("no metric has a finite value" in r.getMessage() for r in caplog.records)

    def test_a_normalized_experiment_still_draws_every_base_metric(self):
        normalized = make_experiment_result(n_models=3, n_folds=2).normalize(REFERENCE_MODEL)
        fig = ViolinPlot().figure(normalized)

        assert len(fig.data) == len(normalized.models) * N_PLOTTED_METRICS


class TestViolinOutputs:
    def test_sections_are_a_single_native_violin_section(self, experiment):
        (section,) = ViolinPlot().sections(experiment)

        assert (section.name, section.anchor) == ("Metric Distributions", "dreval_violin")
        assert section.plot is not None
        assert "Distribution of evaluation metrics" in section.description

    def test_section_draws_one_violin_per_model_and_metric_like_the_figure(self):
        result = ExperimentResult(
            [
                make_run_result(
                    model_name=name, fold_index=i, metrics={"MSE": scale * (i + 1), "Pearson": 0.5 + i / 10}
                )
                for name, scale in (("ModelA", 1.0), ("ModelB", 10.0))
                for i in range(3)
            ]
        )

        (section,) = ViolinPlot().sections(result)

        dataset = section.plot.datasets[0]
        distributions = {
            (header.namespace, header.title): sorted(dataset.violin_value_by_sample_by_metric[anchor].values())
            for anchor, header in dataset.header_by_metric.items()
        }
        assert distributions == {
            ("ModelA", "MSE"): [1.0, 2.0, 3.0],
            ("ModelB", "MSE"): [10.0, 20.0, 30.0],
            ("ModelA", "Pearson"): [0.5, 0.6, 0.7],
            ("ModelB", "Pearson"): [0.5, 0.6, 0.7],
        }
        assert sorted(f"{model}: {metric}" for model, metric in distributions) == sorted(
            trace.name for trace in ViolinPlot().figure(result).data
        )

    def test_a_model_without_finite_values_for_a_metric_gets_no_violin_for_it(self):
        result = ExperimentResult(
            [
                make_run_result(model_name="ModelA", fold_index=i, metrics={"MSE": 1.0 + i, "Pearson": 0.5})
                for i in range(2)
            ]
            + [
                make_run_result(model_name="ModelB", fold_index=i, metrics={"MSE": 5.0 + i, "Pearson": np.nan})
                for i in range(2)
            ]
        )

        assert set(_violins(result)) == {("ModelA", "MSE"), ("ModelB", "MSE"), ("ModelA", "Pearson")}
        assert [trace.name for trace in ViolinPlot().figure(result).data] == [
            "ModelA: MSE",
            "ModelB: MSE",
            "ModelA: Pearson",
        ]

    def test_show_delegates_to_the_plotly_figure(self, experiment, monkeypatch):
        calls: list = []
        monkeypatch.setattr(BaseFigure, "show", lambda self, *a, **kw: calls.append(self))

        show(ViolinPlot(), experiment)

        assert [len(fig.data) for fig in calls] == [len(experiment.models) * N_PLOTTED_METRICS]


class TestComputeSsmd:
    def test_sign_and_antisymmetry(self):
        a, b = [2.0, 3.0, 5.0], [1.0, 2.0, 2.5]

        assert compute_ssmd(a, a) == 0.0
        assert compute_ssmd(a, b) > 0
        assert compute_ssmd(a, b) == pytest.approx(-compute_ssmd(b, a))

    def test_matches_the_closed_form_and_returns_a_builtin_float(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([0.0, 0.5, 1.5])
        expected = (a.mean() - b.mean()) / math.sqrt(a.var(ddof=1) + b.var(ddof=1))

        assert compute_ssmd(a, b) == pytest.approx(expected)
        assert type(compute_ssmd([1, 2, 3], [0, 1, 2])) is float

    def test_is_nan_when_both_inputs_are_constant(self):
        assert math.isnan(compute_ssmd([2.0, 2.0], [1.0, 1.0]))


class TestPlaceholderFigure:
    def test_holds_only_the_message_on_an_axis_free_canvas(self):
        fig = placeholder_figure("Nothing here", (4, 3))

        assert tuple(fig.get_size_inches()) == (4, 3)
        assert [text.get_text() for text in fig.axes[0].texts] == ["Nothing here"]
        assert not fig.axes[0].axison


class TestRunsFrame:
    def test_one_row_per_run_with_identity_columns_and_every_metric(self, experiment):
        df = runs_frame(experiment)

        assert len(df) == sum(m.n_folds for m in experiment.models)
        assert list(df.columns[:4]) == ["algorithm", "rand_setting", "test_mode", "CV_split"]
        assert {"MSE", "RMSE", "MAE", "R^2", "Pearson", "Spearman", "Kendall"} <= set(df.columns)
        assert set(df["test_mode"]) == {experiment.split_mode}
        assert set(df["rand_setting"]) == {"predictions"}
        assert list(df.index) == list(range(len(df)))

    def test_randomized_runs_get_a_composite_setting_label(self):
        result = ExperimentResult([make_run_result(randomization=("gene_expression", "permutation"))])

        df = runs_frame(result, randomization="all")

        assert df["rand_setting"].tolist() == ["gene_expression_permutation"]

    def test_default_selection_excludes_randomized_and_robustness_variants(self):
        result = ExperimentResult(
            [
                make_run_result(),
                make_run_result(randomization=("gene_expression", "permutation")),
                make_run_result(fold_metadata={"robustness_repeat": 0}),
            ]
        )

        assert len(runs_frame(result)) == 1
        assert len(runs_frame(result, randomization="all", robustness="all")) == 3
