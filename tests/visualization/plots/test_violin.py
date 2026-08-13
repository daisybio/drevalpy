"""Tests for :mod:`drevalpy.visualization.plots.violin`.

``to_multiqc`` passes a ``pconfig`` without a ``title``, so MultiQC logs a
validation warning. That is existing behaviour and is asserted as such rather
than worked around.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from plotly.basedatatypes import BaseFigure

from drevalpy.types.results.experiment import ExperimentResult
from drevalpy.visualization.plots.violin import ViolinVisualization, _build_df_from_experiment
from tests.synthetic import REFERENCE_MODEL, make_experiment_result, make_run_result

#: Metrics the plot draws: the seven base metrics, normalized variants excluded.
N_PLOTTED_METRICS = 7


@pytest.fixture(scope="module")
def experiment() -> ExperimentResult:
    return make_experiment_result()


@pytest.fixture(scope="module")
def computed(experiment) -> ViolinVisualization:
    plot = ViolinVisualization()
    plot.compute(experiment)
    return plot


class TestBuildDfFromExperiment:
    def test_has_one_row_per_run(self, experiment):
        df = _build_df_from_experiment(experiment)

        assert len(df) == sum(m.n_folds for m in experiment.models)

    def test_carries_the_identity_columns(self, experiment):
        df = _build_df_from_experiment(experiment)

        assert list(df.columns[:4]) == ["algorithm", "rand_setting", "test_mode", "CV_split"]

    def test_test_mode_comes_from_the_experiment_split_mode(self, experiment):
        df = _build_df_from_experiment(experiment)

        assert set(df["test_mode"]) == {experiment.split_mode}

    def test_uses_a_default_range_index(self, experiment):
        df = _build_df_from_experiment(experiment)

        assert list(df.index) == list(range(len(df)))

    def test_randomized_runs_get_a_composite_setting_label(self):
        result = ExperimentResult([make_run_result(randomization=("methylation", "invariant"))])

        df = _build_df_from_experiment(result)

        assert df["rand_setting"].tolist() == ["methylation_invariant"]


class TestCompute:
    def test_draws_one_violin_per_model_and_metric(self, computed, experiment):
        assert len(computed._fig.data) == len(experiment.models) * N_PLOTTED_METRICS

    def test_every_trace_is_a_violin(self, computed):
        assert {trace.type for trace in computed._fig.data} == {"violin"}

    def test_trace_names_pair_a_model_with_a_metric(self, computed, experiment):
        assert computed._fig.data[0].name == f"{experiment.model_names[0]}: R^2"

    def test_normalized_metrics_are_not_drawn(self, computed):
        assert not any("normalized" in trace.name for trace in computed._fig.data)

    def test_each_violin_holds_one_point_per_fold(self, computed, experiment):
        assert len(computed._fig.data[0].y) == experiment.models[0].n_folds

    def test_violins_show_a_box_and_a_mean_line(self, computed):
        trace = computed._fig.data[0]

        assert trace.box.visible is True
        assert trace.meanline.visible is True

    def test_layout_is_titled_and_sized(self, computed):
        assert computed._fig.layout.title.text == "All Metrics"
        assert (computed._fig.layout.height, computed._fig.layout.width) == (600, 1100)

    def test_multiqc_payload_is_keyed_by_model_and_fold(self, computed, experiment):
        expected = {f"{model.model_name}_fold{run.fold_index}" for model in experiment.models for run in model.runs}

        assert set(computed._data) == expected

    def test_multiqc_payload_holds_the_raw_per_fold_metrics(self, computed, experiment):
        run = experiment.models[0].runs[0]

        assert computed._data[f"{experiment.models[0].model_name}_fold{run.fold_index}"] == run.metrics

    def test_metrics_absent_from_the_result_are_dropped(self):
        result = ExperimentResult(
            [make_run_result(fold_index=i, metrics={"MSE": 0.5 + i, "Pearson": np.nan}) for i in range(2)]
        )
        plot = ViolinVisualization()

        plot.compute(result)

        assert [trace.name for trace in plot._fig.data] == ["ElasticNet: MSE"]

    def test_dataset_argument_is_accepted_and_ignored(self, experiment):
        plot = ViolinVisualization()

        plot.compute(experiment, dataset=object())

        assert plot._fig is not None

    def test_skips_cleanly_with_a_warning_when_no_metric_survives(self, caplog):
        """A plot with nothing to show must warn and skip, not emit an empty figure."""
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


class TestToMultiqc:
    def test_returns_a_single_native_violin_section(self, computed):
        sections = computed.to_multiqc()

        assert len(sections) == 1
        assert (sections[0].name, sections[0].anchor) == ("Metric Distributions", "dreval_violin")

    def test_section_carries_a_native_multiqc_plot(self, computed):
        assert computed.to_multiqc()[0].plot is not None

    def test_section_is_described(self, computed):
        assert "Distribution of evaluation metrics" in computed.to_multiqc()[0].description


class TestShow:
    def test_delegates_to_the_plotly_figure(self, computed, monkeypatch):
        calls: list = []
        monkeypatch.setattr(BaseFigure, "show", lambda self, *a, **kw: calls.append(self))

        computed.show()

        assert calls == [computed._fig]


class TestGuardsBeforeCompute:
    def test_to_png_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match=r"Call compute\(\) before to_png\(\)"):
            ViolinVisualization().to_png(tmp_path / "v.png")

    def test_to_multiqc_raises(self):
        with pytest.raises(RuntimeError, match=r"Call compute\(\) before to_multiqc\(\)"):
            ViolinVisualization().to_multiqc()

    def test_show_raises(self):
        with pytest.raises(RuntimeError, match=r"Call compute\(\) before show\(\)"):
            ViolinVisualization().show()
