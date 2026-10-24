"""Tests for the metric-distribution plots (heatmap, violin) and the shared ``_utils`` helpers."""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import pytest
from plotly.basedatatypes import BaseFigure

from drevalpy.types.results.experiment import ExperimentResult
from drevalpy.visualization.plots._utils import compute_ssmd, runs_frame
from drevalpy.visualization.plots.heatmap import (
    HeatmapVisualization,
    _calc_summary_metric,
    _columns_for_setting,
    _setting_groups,
    _ssmd_matrix,
)
from drevalpy.visualization.plots.violin import ViolinVisualization
from tests.synthetic import REFERENCE_MODEL, make_experiment_result, make_run_result


def _ssmd_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"MSE": [1.0, 2.0, 5.0, 6.0]},
        index=[
            "A_predictions_LPO_split_0",
            "A_predictions_LPO_split_1",
            "B_predictions_LPO_split_0",
            "B_predictions_LPO_split_1",
        ],
    )


@pytest.fixture(scope="module")
def heatmap(experiment) -> HeatmapVisualization:
    plot = HeatmapVisualization()
    plot.compute(experiment)
    return plot


class TestSettingGroups:
    def test_keeps_the_first_three_index_tokens_aligned_with_the_frame(self):
        df = _ssmd_frame()

        groups = _setting_groups(df)

        assert sorted(groups.unique()) == ["A_predictions_LPO", "B_predictions_LPO"]
        assert list(groups.index) == list(df.index)


class TestCalcSummaryMetric:
    def test_averages_each_column_ignoring_nans(self):
        x = pd.DataFrame({"a": [1.0, 3.0], "b": [2.0, 6.0]})

        assert _calc_summary_metric(x).to_dict() == {"a": 2.0, "b": 4.0}
        assert _calc_summary_metric(pd.DataFrame({"a": [1.0, np.nan, 3.0]}))["a"] == 2.0
        assert math.isnan(_calc_summary_metric(pd.DataFrame({"a": [np.nan, np.nan]}))["a"])

    def test_std_error_divides_the_std_by_sqrt_n(self):
        x = pd.DataFrame({"a": [1.0, 3.0]})

        assert _calc_summary_metric(x, std_error=True)["a"] == pytest.approx(1.0 / math.sqrt(2))


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
        df = pd.DataFrame(
            {"MSE": [np.nan, 2.0, 4.0, 5.0, 6.0, 7.0]},
            index=[f"{model}_predictions_LPO_split_{i}" for model in "AB" for i in range(3)],
        )

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


class TestHeatmapCompute:
    def test_builds_five_stacked_panels_with_models_as_rows(self, heatmap, experiment):
        fig = heatmap._fig

        assert [trace.type for trace in fig.data] == ["heatmap"] * 5
        assert set(fig.data[0].y) == set(experiment.model_names)
        assert list(fig.data[0].x) == ["R^2"]
        assert set(fig.data[3].x) == set(fig.data[3].y) == set(experiment.model_names)
        assert "±" in fig.data[0].text[0][0]
        assert fig.layout.height == 500 + len(experiment.models) * 35
        assert fig.layout.width == 1300
        assert fig.layout.title.text == "Heatmap of the evaluation metrics"
        assert heatmap._result is experiment

    def test_panels_without_available_metrics_are_skipped(self):
        result = ExperimentResult(
            [
                make_run_result(model_name=name, fold_index=i, metrics={"MSE": 0.4 + i + offset})
                for offset, name in enumerate("AB")
                for i in range(2)
            ]
        )
        plot = HeatmapVisualization()

        plot.compute(result)

        panels = [list(trace.x) for trace in plot._fig.data]
        assert panels[0] == ["MSE"]
        assert sorted(panels[1]) == ["A", "B"]
        assert len(panels) == 2

    def test_a_normalized_experiment_populates_every_panel(self):
        plot = HeatmapVisualization()

        plot.compute(make_experiment_result(n_models=3, n_folds=2).normalize(REFERENCE_MODEL))

        assert len(plot._fig.data) == 5

    def test_warns_when_no_expected_metric_is_present(self, caplog):
        plot = HeatmapVisualization()

        with caplog.at_level(logging.WARNING, logger="drevalpy.visualization.plots.heatmap"):
            plot.compute(ExperimentResult([make_run_result(metrics={"NotAMetric": 1.0})]))

        assert any("none of the expected metrics" in r.getMessage() for r in caplog.records)


class TestHeatmapOutputsAndGuards:
    def test_to_multiqc_returns_a_single_native_heatmap_section(self, heatmap):
        (section,) = heatmap.to_multiqc()

        assert (section.name, section.anchor) == ("Performance Heatmap", "dreval_heatmap")
        assert section.plot is not None
        assert "Mean metric values per model" in section.description

    def test_show_delegates_to_the_plotly_figure(self, heatmap, monkeypatch):
        calls: list = []
        monkeypatch.setattr(BaseFigure, "show", lambda self, *a, **kw: calls.append(self))

        heatmap.show()

        assert calls == [heatmap._fig]


#: Metrics the plot draws: the seven base metrics.
N_PLOTTED_METRICS = 7


@pytest.fixture(scope="module")
def violin(experiment) -> ViolinVisualization:
    plot = ViolinVisualization()
    plot.compute(experiment)
    return plot


class TestViolinCompute:
    def test_draws_one_violin_per_model_and_metric_holding_one_point_per_fold(self, violin, experiment):
        fig = violin._fig
        trace = fig.data[0]

        assert len(fig.data) == len(experiment.models) * N_PLOTTED_METRICS
        assert {t.type for t in fig.data} == {"violin"}
        assert trace.name == f"{experiment.model_names[0]}: R^2"
        assert len(trace.y) == experiment.models[0].n_folds
        assert trace.box.visible is True
        assert trace.meanline.visible is True
        assert fig.layout.title.text == "All Metrics"
        assert (fig.layout.height, fig.layout.width) == (600, 1100)

    def test_multiqc_payload_is_keyed_by_execution_identity_and_holds_raw_metrics(self, violin, experiment):
        expected = {run.execution_id for model in experiment.models for run in model.select_runs()}
        run = experiment.models[0].runs[0]

        assert set(violin._data) == expected
        assert violin._data[run.execution_id] == run.metrics

    def test_metrics_absent_from_the_result_are_dropped(self):
        result = ExperimentResult(
            [make_run_result(fold_index=i, metrics={"MSE": 0.5 + i, "Pearson": np.nan}) for i in range(2)]
        )
        plot = ViolinVisualization()

        plot.compute(result)

        assert [trace.name for trace in plot._fig.data] == ["ElasticNet: MSE"]

    def test_skips_cleanly_with_a_warning_when_no_metric_survives(self, caplog):
        result = ExperimentResult([make_run_result(fold_index=i, metrics={"Pearson": np.nan}) for i in range(2)])
        plot = ViolinVisualization()

        with caplog.at_level(logging.WARNING, logger="drevalpy.visualization.plots.violin"):
            plot.compute(result)

        assert plot._fig.data == ()
        assert plot.to_multiqc() == []
        assert any("no metric has a finite value" in r.getMessage() for r in caplog.records)

    def test_a_normalized_experiment_still_draws_every_base_metric(self):
        normalized = make_experiment_result(n_models=3, n_folds=2).normalize(REFERENCE_MODEL)
        plot = ViolinVisualization()

        plot.compute(normalized)

        assert len(plot._fig.data) == len(normalized.models) * N_PLOTTED_METRICS


class TestViolinOutputsAndGuards:
    def test_to_multiqc_returns_a_single_native_violin_section(self, violin):
        (section,) = violin.to_multiqc()

        assert (section.name, section.anchor) == ("Metric Distributions", "dreval_violin")
        assert section.plot is not None
        assert "Distribution of evaluation metrics" in section.description

    def test_show_delegates_to_the_plotly_figure(self, violin, monkeypatch):
        calls: list = []
        monkeypatch.setattr(BaseFigure, "show", lambda self, *a, **kw: calls.append(self))

        violin.show()

        assert calls == [violin._fig]


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


class TestRunsFrame:
    def test_one_row_per_run_with_identity_columns_and_every_metric(self, experiment):
        df = runs_frame(experiment)

        assert len(df) == sum(m.n_folds for m in experiment.models)
        assert list(df.columns[:4]) == ["algorithm", "rand_setting", "test_mode", "CV_split"]
        assert {"MSE", "RMSE", "MAE", "R^2", "Pearson", "Spearman", "Kendall"} <= set(df.columns)
        assert set(df["test_mode"]) == {experiment.split_mode}
        assert set(df["rand_setting"]) == {"predictions"}
        assert list(df.index) == list(range(len(df)))

    def test_randomized_runs_get_a_composite_setting_label_also_in_the_index(self):
        result = ExperimentResult([make_run_result(randomization=("gene_expression", "permutation"))])

        df = runs_frame(result, indexed=True, randomization="all")

        assert df["rand_setting"].tolist() == ["gene_expression_permutation"]
        assert "gene_expression_permutation" in df.index[0]

    def test_indexed_encodes_model_setting_mode_and_split_without_changing_the_rows(self, experiment):
        indexed = runs_frame(experiment, indexed=True)

        assert indexed.index[0] == f"{experiment.model_names[0]}_predictions_LPO_split_0"
        assert runs_frame(experiment).to_numpy().tolist() == indexed.to_numpy().tolist()

    def test_default_selection_excludes_randomized_and_robustness_variants(self):
        result = ExperimentResult(
            [
                make_run_result(),
                make_run_result(randomization=("gene_expression", "permutation")),
                make_run_result(fold_metadata={"robustness_trial": 0}),
            ]
        )

        assert len(runs_frame(result)) == 1
        assert len(runs_frame(result, randomization="all", robustness="all")) == 3
