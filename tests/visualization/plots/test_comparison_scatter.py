"""Tests for :mod:`drevalpy.visualization.plots.comparison_scatter`.

The plot enumerates every model pair for ``to_multiqc`` but only draws the first
pair into the Plotly figure. Below two models it silently produces an empty
figure rather than raising, which is asserted here as the contract.
"""

from __future__ import annotations

import math
from itertools import combinations

import pytest
from plotly.basedatatypes import BaseFigure

from drevalpy.types.results.experiment import ExperimentResult
from drevalpy.visualization.plots.comparison_scatter import (
    ComparisonScatterVisualization,
    _collect_model_predictions,
    _compute_pair_points,
)
from tests.synthetic import make_experiment_result, make_run_result

N_MODELS = 3
N_FOLDS = 2
N_PAIRS = 6


@pytest.fixture(scope="module")
def experiment() -> ExperimentResult:
    return make_experiment_result(n_models=N_MODELS, n_folds=N_FOLDS, n_pairs=N_PAIRS)


@pytest.fixture(scope="module")
def computed(experiment) -> ComparisonScatterVisualization:
    plot = ComparisonScatterVisualization()
    plot.compute(experiment)
    return plot


class TestCollectModelPredictions:
    def test_is_keyed_by_model_then_fold_then_sample(self, experiment):
        collected = _collect_model_predictions(experiment)

        assert set(collected) == set(experiment.model_names)
        first = collected[experiment.model_names[0]]
        assert sorted(first) == list(range(N_FOLDS))
        assert sorted(first[0]) == list(range(N_PAIRS))

    def test_values_are_the_run_predictions_as_floats(self, experiment):
        run = experiment.models[0].runs[0]

        collected = _collect_model_predictions(experiment)

        assert collected[run.model_name][run.fold_index][0] == pytest.approx(float(run.predictions[0]))

    def test_randomized_runs_are_excluded(self):
        result = ExperimentResult(
            [
                make_run_result(model_name="ElasticNet", fold_index=0),
                make_run_result(
                    model_name="ElasticNet", fold_index=1, randomization=("gene_expression", "permutation")
                ),
            ]
        )

        assert sorted(_collect_model_predictions(result)["ElasticNet"]) == [0]

    def test_a_fully_randomized_model_maps_to_an_empty_dict(self):
        result = ExperimentResult([make_run_result(randomization=("gene_expression", "permutation"))])

        assert _collect_model_predictions(result) == {"ElasticNet": {}}


class TestComputePairPoints:
    def test_produces_one_point_per_shared_sample_and_fold(self, experiment):
        collected = _collect_model_predictions(experiment)
        name_a, name_b = experiment.model_names[:2]

        points = _compute_pair_points(collected, name_a, name_b)

        assert len(points) == N_FOLDS * N_PAIRS

    def test_points_pair_the_two_models_predictions(self):
        collected = {"A": {0: {0: 1.0}}, "B": {0: {0: 2.0}}}

        assert _compute_pair_points(collected, "A", "B") == [{"x": 1.0, "y": 2.0}]

    def test_unknown_model_names_yield_no_points(self, experiment):
        collected = _collect_model_predictions(experiment)

        assert _compute_pair_points(collected, "Missing", "AlsoMissing") == []

    def test_only_folds_present_in_both_models_are_paired(self):
        collected = {"A": {0: {0: 1.0}, 1: {0: 9.0}}, "B": {1: {0: 2.0}}}

        assert _compute_pair_points(collected, "A", "B") == [{"x": 9.0, "y": 2.0}]

    def test_only_sample_indices_present_in_both_models_are_paired(self):
        collected = {"A": {0: {0: 1.0, 1: 5.0}}, "B": {0: {1: 2.0}}}

        assert _compute_pair_points(collected, "A", "B") == [{"x": 5.0, "y": 2.0}]

    def test_pairs_with_a_nan_on_either_side_are_dropped(self):
        collected = {"A": {0: {0: math.nan, 1: 1.0}}, "B": {0: {0: 2.0, 1: math.nan}}}

        assert _compute_pair_points(collected, "A", "B") == []


class TestCompute:
    def test_draws_only_the_first_pair(self, computed):
        assert len(computed._fig.data) == 1

    def test_the_single_trace_is_a_marker_scatter(self, computed):
        trace = computed._fig.data[0]

        assert (trace.type, trace.mode) == ("scatter", "markers")

    def test_stores_every_model_pair(self, computed, experiment):
        pairs = [(a, b) for a, b, _ in computed._pair_data]

        assert pairs == list(combinations(experiment.model_names, 2))

    def test_every_stored_pair_holds_all_shared_points(self, computed):
        assert {len(points) for _, _, points in computed._pair_data} == {N_FOLDS * N_PAIRS}

    def test_axes_are_labelled_with_the_first_pair(self, computed, experiment):
        name_a, name_b = experiment.model_names[:2]

        assert computed._fig.layout.xaxis.title.text == name_a
        assert computed._fig.layout.yaxis.title.text == name_b

    def test_layout_is_titled_with_the_legend_suppressed(self, computed):
        assert computed._fig.layout.title.text == "Pairwise Model Prediction Comparison"
        assert computed._fig.layout.showlegend is False

    def test_a_single_model_yields_an_empty_figure_rather_than_an_error(self):
        result = ExperimentResult([make_run_result(model_name="Solo", fold_index=i) for i in range(2)])
        plot = ComparisonScatterVisualization()

        plot.compute(result)

        assert plot._fig.data == ()
        assert plot._pair_data == []

    def test_a_single_model_leaves_the_axis_titles_unset(self):
        result = ExperimentResult([make_run_result(model_name="Solo")])
        plot = ComparisonScatterVisualization()

        plot.compute(result)

        assert plot._fig.layout.xaxis.title.text is None

    def test_recomputing_resets_the_previous_pair_data(self, experiment):
        plot = ComparisonScatterVisualization()
        plot.compute(experiment)

        plot.compute(ExperimentResult([make_run_result(model_name="Solo")]))

        assert plot._pair_data == []

    def test_pairs_with_no_shared_folds_are_not_stored(self):
        result = ExperimentResult(
            [
                make_run_result(model_name="OnlyFold0", fold_index=0),
                make_run_result(model_name="OnlyFold1", fold_index=1),
            ]
        )
        plot = ComparisonScatterVisualization()

        plot.compute(result)

        assert plot._pair_data == []
        assert plot._fig.data == ()

    def test_dataset_argument_is_accepted_and_ignored(self, experiment):
        plot = ComparisonScatterVisualization()

        plot.compute(experiment, dataset=object())

        assert len(plot._fig.data) == 1


class TestToMultiqc:
    def test_returns_one_section_per_model_pair(self, computed, experiment):
        sections = computed.to_multiqc()

        assert len(sections) == math.comb(len(experiment.model_names), 2)

    def test_section_anchors_name_both_models(self, computed, experiment):
        name_a, name_b = experiment.model_names[:2]

        assert computed.to_multiqc()[0].anchor == f"dreval_comp_{name_a}_vs_{name_b}"

    def test_sections_carry_native_multiqc_plots(self, computed):
        assert all(section.plot is not None for section in computed.to_multiqc())

    def test_returns_nothing_when_there_are_no_pairs(self):
        result = ExperimentResult([make_run_result(model_name="Solo")])
        plot = ComparisonScatterVisualization()
        plot.compute(result)

        assert plot.to_multiqc() == []


class TestShow:
    def test_delegates_to_the_plotly_figure(self, computed, monkeypatch):
        calls: list = []
        monkeypatch.setattr(BaseFigure, "show", lambda self, *a, **kw: calls.append(self))

        computed.show()

        assert calls == [computed._fig]


class TestGuardsBeforeCompute:
    def test_to_png_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match=r"Call compute\(\) before to_png\(\)"):
            ComparisonScatterVisualization().to_png(tmp_path / "cs.png")

    def test_to_multiqc_raises(self):
        with pytest.raises(RuntimeError, match=r"Call compute\(\) before to_multiqc\(\)"):
            ComparisonScatterVisualization().to_multiqc()

    def test_show_raises(self):
        with pytest.raises(RuntimeError, match=r"Call compute\(\) before show\(\)"):
            ComparisonScatterVisualization().show()
