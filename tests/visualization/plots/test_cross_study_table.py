"""Tests for :mod:`drevalpy.visualization.plots.cross_study_table`.

``compute`` looks for ``rand_setting`` values containing ``"cross-study-"``.
Nothing in the package emits such a setting today, so in practice the plot always
falls through to :func:`_build_simple_table`; the fallback is therefore treated
as the primary behaviour, with the cross-study branch covered by a hand-built
result.
"""

from __future__ import annotations

import pytest
from plotly.basedatatypes import BaseFigure

from drevalpy.types.results.experiment import ExperimentResult
from drevalpy.visualization.plots._utils import runs_frame
from drevalpy.visualization.plots.cross_study_table import (
    CrossStudyTableVisualization,
    _build_simple_table,
)
from tests.synthetic import make_experiment_result, make_run_result

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _cross_study_result() -> ExperimentResult:
    """An experiment carrying the ``cross-study-*`` setting the plot looks for."""
    return ExperimentResult(
        [
            make_run_result(model_name="ElasticNet", fold_index=i, randomization=("cross-study-CCLE", "eval"))
            for i in (0, 1)
        ]
        + [make_run_result(model_name="ElasticNet", fold_index=i) for i in (0, 1)]
    )


@pytest.fixture(scope="module")
def experiment() -> ExperimentResult:
    return make_experiment_result()


@pytest.fixture(scope="module")
def computed(experiment) -> CrossStudyTableVisualization:
    plot = CrossStudyTableVisualization()
    plot.compute(experiment)
    return plot


class TestBuildSimpleTable:
    def test_produces_a_single_plotly_table(self, experiment):
        fig = _build_simple_table(experiment)

        assert [trace.type for trace in fig.data] == ["table"]

    def test_header_starts_with_the_model_column(self, experiment):
        fig = _build_simple_table(experiment)

        assert fig.data[0].header.values[0] == "Model"

    def test_header_lists_every_metric_alphabetically(self, experiment):
        metrics = sorted({m for model in experiment.models for m in model.aggregate_metrics})

        fig = _build_simple_table(experiment)

        assert list(fig.data[0].header.values[1:]) == metrics

    def test_first_column_holds_the_model_names(self, experiment):
        fig = _build_simple_table(experiment)

        assert list(fig.data[0].cells.values[0]) == experiment.model_names

    def test_cells_are_formatted_as_mean_plus_minus_std(self, experiment):
        fig = _build_simple_table(experiment)

        assert " ± " in fig.data[0].cells.values[1][0]

    def test_metrics_missing_for_a_model_render_as_not_available(self):
        result = ExperimentResult(
            [
                make_run_result(model_name="OnlyMse", metrics={"MSE": 0.5}),
                make_run_result(model_name="OnlyMae", metrics={"MAE": 0.25}),
            ]
        )

        fig = _build_simple_table(result)

        assert list(fig.data[0].header.values) == ["Model", "MAE", "MSE"]
        assert list(fig.data[0].cells.values[1]) == ["N/A", "0.250 ± 0.000"]

    def test_is_titled(self, experiment):
        assert _build_simple_table(experiment).layout.title.text == "Model Performance Summary"


class TestComputeWithoutCrossStudyData:
    def test_finds_no_cross_study_datasets(self, computed):
        assert computed._cross_study_datasets == []

    def test_builds_no_per_dataset_figures(self, computed):
        assert computed._figures == {}

    def test_falls_back_to_the_simple_summary_table(self, computed):
        assert computed._fig.layout.title.text == "Model Performance Summary"

    def test_stores_the_result(self, computed, experiment):
        assert computed._result is experiment

    def test_dataset_argument_is_accepted_and_ignored(self, experiment):
        plot = CrossStudyTableVisualization()

        plot.compute(experiment, dataset=object())

        assert plot._fig is not None


class TestComputeWithCrossStudyData:
    def test_extracts_the_target_dataset_name(self):
        plot = CrossStudyTableVisualization()

        plot.compute(_cross_study_result())

        assert plot._cross_study_datasets == ["CCLE_eval"]

    def test_builds_one_figure_per_cross_study_dataset(self):
        plot = CrossStudyTableVisualization()

        plot.compute(_cross_study_result())

        assert list(plot._figures) == ["CCLE_eval"]
        assert plot._fig is plot._figures["CCLE_eval"]

    def test_figure_is_titled_with_the_target_dataset(self):
        plot = CrossStudyTableVisualization()

        plot.compute(_cross_study_result())

        assert plot._fig.layout.title.text == "Evaluation Metrics for Cross-Study Predictions to CCLE_eval"

    def test_only_cross_study_runs_are_summarized(self):
        plot = CrossStudyTableVisualization()

        plot.compute(_cross_study_result())

        assert list(plot._mean_metrics[0].index) == ["ElasticNet"]

    def test_models_are_ordered_by_mean_mse(self):
        runs = [
            make_run_result(
                model_name=name,
                fold_index=fold,
                randomization=("cross-study-CCLE", "eval"),
                metrics={"MSE": mse + fold},
            )
            for name, mse in (("Worse", 5.0), ("Better", 1.0))
            for fold in (0, 1)
        ]
        plot = CrossStudyTableVisualization()

        plot.compute(ExperimentResult(runs))

        assert list(plot._mean_metrics[0].index) == ["Better", "Worse"]

    def test_std_rows_follow_the_sorted_mean_rows(self):
        plot = CrossStudyTableVisualization()

        plot.compute(_cross_study_result())

        assert list(plot._std_metrics[0].index) == list(plot._mean_metrics[0].index)


class TestCrossStudyDataframeQuirk:
    def test_nothing_in_the_package_emits_a_cross_study_setting(self, experiment):
        """Guards the documented fallback: ``rand_setting`` never carries the prefix."""
        df = runs_frame(experiment, indexed=True)

        assert not df["rand_setting"].str.contains("cross-study-").any()

    def test_a_cross_study_setting_reaches_the_index_when_present(self):
        """The compute() branch keys off the index, so the label has to survive into it."""
        df = runs_frame(_cross_study_result(), indexed=True)

        assert any("cross-study-CCLE_eval" in idx for idx in df.index)


class TestToMultiqc:
    def test_returns_a_single_native_table_section(self, computed):
        sections = computed.to_multiqc()

        assert len(sections) == 1
        assert (sections[0].name, sections[0].anchor) == ("Model Summary Table", "dreval_summary_table")

    def test_section_carries_a_native_multiqc_plot(self, computed):
        assert computed.to_multiqc()[0].plot is not None

    def test_payload_holds_a_mean_and_a_std_column_per_metric(self, computed, monkeypatch):
        from multiqc.plots import table as mqc_table

        captured: list = []
        monkeypatch.setattr(mqc_table, "plot", lambda data, headers, pconfig: captured.append((data, headers)))

        computed.to_multiqc()

        data, headers = captured[0]
        assert set(data["ElasticNet"]) == set(headers)
        assert {"MSE_mean", "MSE_std"} <= set(headers)


class TestRendering:
    def test_to_png_writes_a_png_file(self, computed, tmp_path):
        out = tmp_path / "table.png"

        computed.to_png(out)

        assert out.read_bytes().startswith(PNG_MAGIC)


class TestShow:
    def test_delegates_to_the_plotly_figure(self, computed, monkeypatch):
        calls: list = []
        monkeypatch.setattr(BaseFigure, "show", lambda self, *a, **kw: calls.append(self))

        computed.show()

        assert calls == [computed._fig]


class TestGuardsBeforeCompute:
    def test_to_png_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match=r"Call compute\(\) before to_png\(\)"):
            CrossStudyTableVisualization().to_png(tmp_path / "t.png")

    def test_to_multiqc_raises(self):
        with pytest.raises(RuntimeError, match=r"Call compute\(\) before to_multiqc\(\)"):
            CrossStudyTableVisualization().to_multiqc()

    def test_show_raises(self):
        with pytest.raises(RuntimeError, match=r"Call compute\(\) before show\(\)"):
            CrossStudyTableVisualization().show()
