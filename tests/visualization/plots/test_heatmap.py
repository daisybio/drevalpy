"""Tests for :mod:`drevalpy.visualization.plots.heatmap`.

``compute()`` is asserted in-process on the resulting Plotly figure. ``to_png()``
is deliberately not exercised: it goes through kaleido and costs roughly 13
seconds, which is not worth it for a hook that runs on every commit.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import pytest
from plotly.basedatatypes import BaseFigure

from drevalpy.types.results.experiment import ExperimentResult
from drevalpy.visualization.plots.heatmap import (
    HeatmapVisualization,
    _build_df_from_experiment,
    _calc_summary_metric,
    _columns_for_setting,
    _compute_ssmd,
    _resolve_metric_columns,
    _setting_groups,
)
from tests.synthetic import NORMALIZED_METRIC, REFERENCE_MODEL, make_experiment_result, make_run_result


def _ssmd_frame() -> pd.DataFrame:
    """Two models x two folds, indexed the way ``_build_df_from_experiment`` does."""
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


class TestBuildDfFromExperiment:
    def test_has_one_row_per_run(self, experiment):
        df = _build_df_from_experiment(experiment)

        assert len(df) == sum(m.n_folds for m in experiment.models)

    def test_carries_the_identity_columns_plus_every_metric(self, experiment):
        df = _build_df_from_experiment(experiment)

        assert list(df.columns[:4]) == ["algorithm", "rand_setting", "test_mode", "CV_split"]
        assert {"MSE", "RMSE", "MAE", "R^2", "Pearson", "Spearman", "Kendall"} <= set(df.columns)

    def test_index_encodes_model_setting_mode_and_split(self, experiment):
        df = _build_df_from_experiment(experiment)

        assert df.index[0] == f"{experiment.model_names[0]}_predictions_LPO_split_0"

    def test_unrandomized_runs_are_labelled_predictions(self, experiment):
        df = _build_df_from_experiment(experiment)

        assert set(df["rand_setting"]) == {"predictions"}

    def test_randomized_runs_are_labelled_by_view_and_mode(self):
        result = ExperimentResult([make_run_result(randomization=("gene_expression", "permutation"))])

        df = _build_df_from_experiment(result)

        assert df["rand_setting"].tolist() == ["gene_expression_permutation"]


class TestSettingGroups:
    def test_keeps_the_first_three_index_tokens(self):
        df = _ssmd_frame()

        assert sorted(_setting_groups(df).unique()) == ["A_predictions_LPO", "B_predictions_LPO"]

    def test_is_aligned_with_the_frame_index(self):
        df = _ssmd_frame()

        assert list(_setting_groups(df).index) == list(df.index)


class TestCalcSummaryMetric:
    def test_averages_each_column(self):
        x = pd.DataFrame({"a": [1.0, 3.0], "b": [2.0, 6.0]})

        assert _calc_summary_metric(x).to_dict() == {"a": 2.0, "b": 4.0}

    def test_std_error_divides_the_std_by_sqrt_n(self):
        x = pd.DataFrame({"a": [1.0, 3.0]})

        assert _calc_summary_metric(x, std_error=True)["a"] == pytest.approx(1.0 / math.sqrt(2))

    def test_ignores_nans_when_averaging(self):
        x = pd.DataFrame({"a": [1.0, np.nan, 3.0]})

        assert _calc_summary_metric(x)["a"] == 2.0

    def test_all_nan_column_stays_nan(self):
        x = pd.DataFrame({"a": [np.nan, np.nan]})

        assert math.isnan(_calc_summary_metric(x)["a"])

    def test_result_is_indexed_by_column(self):
        x = pd.DataFrame({"a": [1.0], "b": [2.0]})

        assert list(_calc_summary_metric(x).index) == ["a", "b"]


class TestComputeSsmd:
    def test_returns_an_empty_frame_when_the_metric_is_absent(self):
        assert _compute_ssmd(_ssmd_frame(), "Pearson").empty

    def test_is_a_square_model_by_model_matrix(self):
        matrix = _compute_ssmd(_ssmd_frame(), "MSE")

        assert list(matrix.index) == list(matrix.columns) == ["A", "B"]

    def test_diagonal_is_zero(self):
        matrix = _compute_ssmd(_ssmd_frame(), "MSE")

        assert matrix.loc["A", "A"] == 0.0
        assert matrix.loc["B", "B"] == 0.0

    def test_off_diagonal_is_antisymmetric(self):
        matrix = _compute_ssmd(_ssmd_frame(), "MSE")

        assert matrix.loc["A", "B"] == pytest.approx(-matrix.loc["B", "A"])

    def test_is_negative_when_the_row_model_scores_lower(self):
        matrix = _compute_ssmd(_ssmd_frame(), "MSE")

        assert matrix.loc["A", "B"] < 0

    def test_is_nan_when_both_models_have_zero_variance(self):
        df = pd.DataFrame(
            {"MSE": [1.0, 1.0, 2.0, 2.0]},
            index=[
                "A_predictions_LPO_split_0",
                "A_predictions_LPO_split_1",
                "B_predictions_LPO_split_0",
                "B_predictions_LPO_split_1",
            ],
        )

        assert math.isnan(_compute_ssmd(df, "MSE").loc["A", "B"])


class TestColumnsForSetting:
    @pytest.mark.parametrize(
        ("setting", "expected"),
        [
            pytest.param("r2", ["R^2"], id="r2"),
            pytest.param("correlations", ["Pearson", "Spearman", "Kendall"], id="correlations"),
            pytest.param("errors", ["MSE", "RMSE", "MAE"], id="errors"),
            pytest.param("unknown", [], id="unknown_setting"),
        ],
    )
    def test_selects_the_metrics_belonging_to_the_panel(self, setting, expected):
        metric_cols = ["R^2", "Pearson", "Spearman", "Kendall", "MSE", "RMSE", "MAE"]

        assert _columns_for_setting(setting, metric_cols) == expected

    def test_normalized_variants_are_grouped_with_their_base_metric(self):
        assert _columns_for_setting("r2", ["R^2", "R^2: normalized"]) == ["R^2", "R^2: normalized"]

    def test_normalized_errors_are_excluded_by_exact_match(self):
        assert _columns_for_setting("errors", ["MSE: normalized"]) == []


class TestCompute:
    def test_builds_five_stacked_heatmap_panels(self, computed):
        assert [trace.type for trace in computed._fig.data] == ["heatmap"] * 5

    def test_rows_are_the_models(self, computed, experiment):
        assert set(computed._fig.data[0].y) == set(experiment.model_names)

    def test_first_panel_shows_r2_only(self, computed):
        assert list(computed._fig.data[0].x) == ["R^2"]

    def test_ssmd_panels_are_model_by_model(self, computed, experiment):
        ssmd_trace = computed._fig.data[3]

        assert set(ssmd_trace.x) == set(ssmd_trace.y) == set(experiment.model_names)

    def test_cells_are_annotated_with_mean_and_standard_error(self, computed):
        assert "±" in computed._fig.data[0].text[0][0]

    def test_layout_scales_the_height_with_the_model_count(self, computed, experiment):
        assert computed._fig.layout.height == 500 + len(experiment.models) * 35
        assert computed._fig.layout.width == 1300

    def test_layout_is_titled(self, computed):
        assert computed._fig.layout.title.text == "Heatmap of the evaluation metrics"

    def test_stores_the_result_for_to_multiqc(self, computed, experiment):
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

    def test_dataset_argument_is_accepted_and_ignored(self, experiment):
        plot = HeatmapVisualization()

        plot.compute(experiment, dataset=object())

        assert plot._fig is not None


class TestMetricNameResolution:
    """``normalize()`` emits plain names; older results carry suffixed ones."""

    def test_a_normalized_experiment_populates_every_panel(self):
        normalized = make_experiment_result(n_models=3, n_folds=2).normalize(REFERENCE_MODEL)
        plot = HeatmapVisualization()

        plot.compute(normalized)

        assert len(plot._fig.data) == 5

    def test_the_legacy_suffixed_spelling_is_folded_onto_the_base_name(self):
        result = ExperimentResult(
            [
                make_run_result(model_name=name, fold_index=i, metrics={NORMALIZED_METRIC: 0.4 + i + offset})
                for offset, name in enumerate("AB")
                for i in range(2)
            ]
        )
        plot = HeatmapVisualization()

        plot.compute(result)

        assert [list(trace.x) for trace in plot._fig.data][0] == ["Pearson"]

    def test_resolve_metric_columns_prefers_the_plain_name(self):
        result = ExperimentResult([make_run_result(metrics={"Pearson": 0.9, NORMALIZED_METRIC: 0.1})])
        df = _build_df_from_experiment(result)

        renamed, columns = _resolve_metric_columns(result, df)

        assert columns == ["Pearson"]
        assert renamed["Pearson"].tolist() == [0.9]

    def test_warns_when_no_expected_metric_is_present(self, caplog):
        result = ExperimentResult([make_run_result(metrics={"NotAMetric": 1.0})])
        plot = HeatmapVisualization()

        with caplog.at_level(logging.WARNING, logger="drevalpy.visualization.plots.heatmap"):
            plot.compute(result)

        assert any("none of the expected metrics" in r.getMessage() for r in caplog.records)


class TestToMultiqc:
    def test_returns_a_single_native_heatmap_section(self, computed):
        sections = computed.to_multiqc()

        assert len(sections) == 1
        assert (sections[0].name, sections[0].anchor) == ("Performance Heatmap", "dreval_heatmap")

    def test_section_carries_a_native_multiqc_plot(self, computed):
        assert computed.to_multiqc()[0].plot is not None

    def test_section_is_described(self, computed):
        assert "Mean metric values per model" in computed.to_multiqc()[0].description


class TestShow:
    def test_delegates_to_the_plotly_figure(self, computed, monkeypatch):
        calls: list = []
        monkeypatch.setattr(BaseFigure, "show", lambda self, *a, **kw: calls.append(self))

        computed.show()

        assert calls == [computed._fig]


class TestGuardsBeforeCompute:
    def test_to_png_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match=r"Call compute\(\) before to_png\(\)"):
            HeatmapVisualization().to_png(tmp_path / "h.png")

    def test_to_multiqc_raises(self):
        with pytest.raises(RuntimeError, match=r"Call compute\(\) before to_multiqc\(\)"):
            HeatmapVisualization().to_multiqc()

    def test_show_raises(self):
        with pytest.raises(RuntimeError, match=r"Call compute\(\) before show\(\)"):
            HeatmapVisualization().show()
