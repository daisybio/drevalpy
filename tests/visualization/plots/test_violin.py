"""Tests for :mod:`drevalpy.visualization.plots.violin`."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from plotly.basedatatypes import BaseFigure

from drevalpy.types.results.experiment import ExperimentResult
from drevalpy.visualization.plots.violin import ViolinVisualization
from tests.synthetic import REFERENCE_MODEL, make_experiment_result, make_run_result

#: Metrics the plot draws: the seven base metrics.
N_PLOTTED_METRICS = 7


@pytest.fixture(scope="module")
def experiment() -> ExperimentResult:
    return make_experiment_result()


@pytest.fixture(scope="module")
def computed(experiment) -> ViolinVisualization:
    plot = ViolinVisualization()
    plot.compute(experiment)
    return plot


class TestCompute:
    def test_draws_one_violin_per_model_and_metric_holding_one_point_per_fold(self, computed, experiment):
        fig = computed._fig
        trace = fig.data[0]

        assert len(fig.data) == len(experiment.models) * N_PLOTTED_METRICS
        assert {t.type for t in fig.data} == {"violin"}
        assert trace.name == f"{experiment.model_names[0]}: R^2"
        assert len(trace.y) == experiment.models[0].n_folds
        assert trace.box.visible is True
        assert trace.meanline.visible is True
        assert fig.layout.title.text == "All Metrics"
        assert (fig.layout.height, fig.layout.width) == (600, 1100)

    def test_multiqc_payload_is_keyed_by_execution_identity_and_holds_raw_metrics(self, computed, experiment):
        expected = {run.execution_id for model in experiment.models for run in model.select_runs()}
        run = experiment.models[0].runs[0]

        assert set(computed._data) == expected
        assert computed._data[run.execution_id] == run.metrics

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


class TestOutputsAndGuards:
    def test_to_multiqc_returns_a_single_native_violin_section(self, computed):
        (section,) = computed.to_multiqc()

        assert (section.name, section.anchor) == ("Metric Distributions", "dreval_violin")
        assert section.plot is not None
        assert "Distribution of evaluation metrics" in section.description

    def test_show_delegates_to_the_plotly_figure(self, computed, monkeypatch):
        calls: list = []
        monkeypatch.setattr(BaseFigure, "show", lambda self, *a, **kw: calls.append(self))

        computed.show()

        assert calls == [computed._fig]

    @pytest.mark.parametrize("method", ["to_png", "to_multiqc", "show"])
    def test_methods_raise_before_compute(self, tmp_path, method):
        args = (tmp_path / "v.png",) if method == "to_png" else ()

        with pytest.raises(RuntimeError, match=rf"Call compute\(\) before {method}\(\)"):
            getattr(ViolinVisualization(), method)(*args)
