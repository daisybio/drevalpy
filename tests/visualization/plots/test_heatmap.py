"""Tests for :mod:`drevalpy.visualization.plots.heatmap` (``to_png`` goes through kaleido and is skipped)."""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import pytest
from plotly.basedatatypes import BaseFigure

from drevalpy.types.results.experiment import ExperimentResult
from drevalpy.visualization.plots._utils import compute_ssmd
from drevalpy.visualization.plots.heatmap import (
    HeatmapVisualization,
    _calc_summary_metric,
    _columns_for_setting,
    _setting_groups,
    _ssmd_matrix,
)
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
def experiment() -> ExperimentResult:
    return make_experiment_result()


@pytest.fixture(scope="module")
def computed(experiment) -> HeatmapVisualization:
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


class TestCompute:
    def test_builds_five_stacked_panels_with_models_as_rows(self, computed, experiment):
        fig = computed._fig

        assert [trace.type for trace in fig.data] == ["heatmap"] * 5
        assert set(fig.data[0].y) == set(experiment.model_names)
        assert list(fig.data[0].x) == ["R^2"]
        assert set(fig.data[3].x) == set(fig.data[3].y) == set(experiment.model_names)
        assert "±" in fig.data[0].text[0][0]
        assert fig.layout.height == 500 + len(experiment.models) * 35
        assert fig.layout.width == 1300
        assert fig.layout.title.text == "Heatmap of the evaluation metrics"
        assert computed._result is experiment

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


class TestOutputsAndGuards:
    def test_to_multiqc_returns_a_single_native_heatmap_section(self, computed):
        (section,) = computed.to_multiqc()

        assert (section.name, section.anchor) == ("Performance Heatmap", "dreval_heatmap")
        assert section.plot is not None
        assert "Mean metric values per model" in section.description

    def test_show_delegates_to_the_plotly_figure(self, computed, monkeypatch):
        calls: list = []
        monkeypatch.setattr(BaseFigure, "show", lambda self, *a, **kw: calls.append(self))

        computed.show()

        assert calls == [computed._fig]

    @pytest.mark.parametrize("method", ["to_png", "to_multiqc", "show"])
    def test_methods_raise_before_compute(self, tmp_path, method):
        args = (tmp_path / "h.png",) if method == "to_png" else ()

        with pytest.raises(RuntimeError, match=rf"Call compute\(\) before {method}\(\)"):
            getattr(HeatmapVisualization(), method)(*args)
