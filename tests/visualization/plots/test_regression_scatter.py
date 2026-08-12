"""Tests for :mod:`drevalpy.visualization.plots.regression_scatter`.

This is the only plot registered for ``ModelResult`` rather than
``ExperimentResult``. Plotly Express adds an OLS trendline trace next to every
group's marker trace, so trace counts are twice the number of groups.
"""

from __future__ import annotations

import numpy as np
import pytest
from plotly.basedatatypes import BaseFigure

from drevalpy.types.results.model import ModelResult
from drevalpy.visualization.plots.regression_scatter import (
    RegressionScatterVisualization,
    _build_regression_df,
)
from tests.synthetic import DEFAULT_DATASET_NAME, make_model_result, make_run_result

N_FOLDS = 2
N_PAIRS = 8
N_DRUGS = 4
N_CELL_LINES = 5


@pytest.fixture(scope="module")
def model_result() -> ModelResult:
    return make_model_result(n_folds=N_FOLDS, n_pairs=N_PAIRS)


@pytest.fixture(scope="module")
def computed(model_result) -> RegressionScatterVisualization:
    plot = RegressionScatterVisualization()
    plot.compute(model_result)
    return plot


def _model_result(runs) -> ModelResult:
    return ModelResult(model_name="ElasticNet", dataset_name=DEFAULT_DATASET_NAME, runs=runs)


class TestBuildRegressionDf:
    def test_has_one_row_per_prediction(self, model_result):
        df = _build_regression_df(model_result)

        assert len(df) == N_FOLDS * N_PAIRS

    def test_columns_cover_truth_prediction_and_identity(self, model_result):
        df = _build_regression_df(model_result)

        assert list(df.columns) == ["y_true", "y_pred", "algorithm", "CV_split", "drug_name", "cell_line_name"]

    def test_identity_columns_are_strings(self, model_result):
        df = _build_regression_df(model_result)

        assert df["drug_name"].map(type).eq(str).all()
        assert df["cell_line_name"].map(type).eq(str).all()

    def test_every_fold_is_represented(self, model_result):
        df = _build_regression_df(model_result)

        assert sorted(df["CV_split"].unique()) == list(range(N_FOLDS))

    def test_randomized_runs_are_excluded(self):
        result = _model_result(
            [
                make_run_result(fold_index=0, n_pairs=4),
                make_run_result(fold_index=1, n_pairs=4, randomization=("gene_expression", "permutation")),
            ]
        )

        df = _build_regression_df(result)

        assert df["CV_split"].unique().tolist() == [0]

    def test_is_empty_when_every_run_is_randomized(self):
        result = _model_result([make_run_result(randomization=("gene_expression", "permutation"))])

        assert _build_regression_df(result).empty


class TestCompute:
    def test_draws_a_marker_and_a_trendline_trace_per_drug(self, computed):
        assert len(computed._fig.data) == 2 * N_DRUGS
        assert [trace.mode for trace in computed._fig.data[:2]] == ["markers", "lines"]

    def test_traces_are_named_after_the_grouping_column(self, computed):
        assert sorted({trace.name for trace in computed._fig.data}) == [f"D_{i}" for i in range(N_DRUGS)]

    def test_grouping_by_cell_line_is_supported(self, model_result):
        plot = RegressionScatterVisualization()

        plot.compute(model_result, group_by="cell_line_name")

        assert plot._group_by == "cell_line_name"
        assert len(plot._fig.data) == 2 * N_CELL_LINES

    def test_title_names_the_model(self, computed, model_result):
        assert computed._fig.layout.title.text == f"{model_result.model_name}: Regression plot"

    def test_both_axes_share_one_square_range(self, computed):
        assert computed._fig.layout.xaxis.range == computed._fig.layout.yaxis.range

    def test_groups_with_a_single_observation_are_filtered_out(self):
        result = _model_result([make_run_result(n_pairs=5, n_drugs=4)])
        plot = RegressionScatterVisualization()

        plot.compute(result)

        assert {trace.name for trace in plot._fig.data} == {"D_0"}

    def test_no_data_yields_an_empty_figure(self):
        result = _model_result([make_run_result(randomization=("gene_expression", "permutation"))])
        plot = RegressionScatterVisualization()

        plot.compute(result)

        assert plot._fig.data == ()

    def test_stores_the_result(self, computed, model_result):
        assert computed._result is model_result

    def test_group_by_defaults_to_drug_name(self):
        assert RegressionScatterVisualization()._group_by == "drug_name"

    def test_dataset_argument_is_accepted_and_ignored(self, model_result):
        plot = RegressionScatterVisualization()

        plot.compute(model_result, dataset=object())

        assert len(plot._fig.data) == 2 * N_DRUGS


class TestToMultiqc:
    def test_returns_a_single_scatter_section_anchored_on_the_model(self, computed, model_result):
        sections = computed.to_multiqc()

        assert len(sections) == 1
        assert sections[0].anchor == f"dreval_scatter_{model_result.model_name}"

    def test_section_is_named_and_described_with_the_fold_count(self, computed, model_result):
        section = computed.to_multiqc()[0]

        assert section.name == f"Regression Scatter: {model_result.model_name}"
        assert f"across {N_FOLDS} fold(s)" in section.description

    def test_section_carries_a_native_multiqc_plot(self, computed):
        assert computed.to_multiqc()[0].plot is not None

    def test_nan_pairs_are_masked_out_of_the_scatter_payload(self, monkeypatch):
        from multiqc.plots import scatter as mqc_scatter

        captured: list = []
        monkeypatch.setattr(mqc_scatter, "plot", lambda datasets, pconfig: captured.append(datasets))
        run = make_run_result(n_pairs=6)
        run.predictions[0] = np.nan
        run.ground_truth[1] = np.nan
        plot = RegressionScatterVisualization()
        plot._result = _model_result([run])

        plot.to_multiqc()

        assert len(captured[0][0]["fold_0"]) == 4

    def test_randomized_runs_contribute_no_dataset(self, monkeypatch):
        from multiqc.plots import scatter as mqc_scatter

        captured: list = []
        monkeypatch.setattr(mqc_scatter, "plot", lambda datasets, pconfig: captured.append(datasets))
        plot = RegressionScatterVisualization()
        plot._result = _model_result(
            [
                make_run_result(fold_index=0, n_pairs=4),
                make_run_result(fold_index=1, n_pairs=4, randomization=("gene_expression", "permutation")),
            ]
        )

        plot.to_multiqc()

        assert [set(dataset) for dataset in captured[0]] == [{"fold_0"}]


class TestShow:
    def test_delegates_to_the_plotly_figure(self, computed, monkeypatch):
        calls: list = []
        monkeypatch.setattr(BaseFigure, "show", lambda self, *a, **kw: calls.append(self))

        computed.show()

        assert calls == [computed._fig]


class TestGuardsBeforeCompute:
    def test_to_png_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match=r"Call compute\(\) before to_png\(\)"):
            RegressionScatterVisualization().to_png(tmp_path / "rs.png")

    def test_to_multiqc_raises(self):
        with pytest.raises(RuntimeError, match=r"Call compute\(\) before to_multiqc\(\)"):
            RegressionScatterVisualization().to_multiqc()

    def test_show_raises(self):
        with pytest.raises(RuntimeError, match=r"Call compute\(\) before show\(\)"):
            RegressionScatterVisualization().show()
