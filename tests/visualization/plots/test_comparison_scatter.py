"""Tests for :mod:`drevalpy.visualization.plots.comparison_scatter`.

The plot's contract is that its retained state and its report payload are both
bounded by ``models x groups``: one correlation per model per drug (or cell
line), with model selection deferred to two Plotly dropdowns. Nothing here may
scale with the number of predictions - that regression is what
``tests/test_visualization_payload_policy.py`` guards.

The figure is emitted as raw HTML calling ``Plotly.newPlot`` against MultiQC's
bundled Plotly global rather than as a native MultiQC plot, so the assertions
below are on ``Section.content``.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest
from plotly.basedatatypes import BaseFigure

from drevalpy.types.results.experiment import ExperimentResult
from drevalpy.visualization.plots._group_metrics import GroupCorrelationMatrix, model_group_correlations
from drevalpy.visualization.plots.comparison_scatter import (
    _AXIS_RANGE,
    _POINTS_TRACE,
    ComparisonScatterVisualization,
    _axis_layout,
    _build_figure,
    _dropdown_buttons,
    _inline_plotly_html,
)
from tests.synthetic import make_experiment_result, make_run_result

N_MODELS = 3
N_FOLDS = 2
N_PAIRS = 20
N_DRUGS = 4
N_CELL_LINES = 5


@pytest.fixture(scope="module")
def experiment() -> ExperimentResult:
    return make_experiment_result(n_models=N_MODELS, n_folds=N_FOLDS, n_pairs=N_PAIRS)


@pytest.fixture(scope="module")
def computed(experiment) -> ComparisonScatterVisualization:
    plot = ComparisonScatterVisualization()
    plot.compute(experiment)
    return plot


@pytest.fixture(scope="module")
def matrix(experiment) -> GroupCorrelationMatrix:
    return model_group_correlations(experiment, "drug")


def _empty_matrix() -> GroupCorrelationMatrix:
    return GroupCorrelationMatrix("drug", (), (), np.empty((0, 0), dtype=np.float32))


class TestAxisLayout:
    def test_title_is_nested_so_plotly_3_keeps_it(self):
        """A bare string under ``title`` is silently dropped by a relayout."""
        layout = _axis_layout("drug", "ElasticNet")

        assert layout["title"] == {"text": "ElasticNet (per-drug Pearson)"}

    def test_range_is_the_fixed_correlation_range(self):
        assert _axis_layout("drug", "ElasticNet")["range"] == list(_AXIS_RANGE)

    def test_the_grouping_label_is_spelled_out(self):
        assert "per-cell line" in _axis_layout("cell_line", "ElasticNet")["title"]["text"]


class TestDropdownButtons:
    def test_one_button_per_model(self, matrix):
        assert len(_dropdown_buttons(matrix, "x")) == matrix.n_models

    def test_buttons_are_labelled_with_the_model_name(self, matrix):
        labels = [button["label"] for button in _dropdown_buttons(matrix, "y")]

        assert labels == list(matrix.model_names)

    def test_each_button_carries_one_models_group_vector(self, matrix):
        button = _dropdown_buttons(matrix, "x")[0]

        assert len(button["args"][0]["x"][0]) == matrix.n_groups

    def test_the_x_axis_buttons_restyle_x_and_relayout_xaxis(self, matrix):
        button = _dropdown_buttons(matrix, "x")[0]

        assert set(button["args"][0]) == {"x"}
        assert set(button["args"][1]) == {"xaxis"}

    def test_the_y_axis_buttons_restyle_y_and_relayout_yaxis(self, matrix):
        button = _dropdown_buttons(matrix, "y")[0]

        assert set(button["args"][0]) == {"y"}
        assert set(button["args"][1]) == {"yaxis"}

    def test_restyles_target_only_the_points_trace(self, matrix):
        """Without an explicit index Plotly cycles the restyle onto the reference line."""
        assert all(button["args"][2] == [_POINTS_TRACE] for button in _dropdown_buttons(matrix, "x"))

    def test_undefined_correlations_become_zero_rather_than_nan(self):
        matrix = GroupCorrelationMatrix("drug", ("A",), ("D_0", "D_1"), np.array([[np.nan, 0.5]], dtype=np.float32))

        values = _dropdown_buttons(matrix, "x")[0]["args"][0]["x"][0]

        assert values[0] == 0.0
        assert all(not np.isnan(value) for value in values)

    def test_values_are_json_serialisable_python_floats(self, matrix):
        values = _dropdown_buttons(matrix, "x")[0]["args"][0]["x"][0]

        assert json.loads(json.dumps(values)) == values


class TestBuildFigure:
    def test_draws_a_points_trace_and_a_reference_line(self, matrix):
        fig = _build_figure(matrix)

        assert len(fig.data) == 2
        assert [trace.mode for trace in fig.data] == ["markers", "lines"]

    def test_the_points_trace_is_first(self, matrix):
        assert _build_figure(matrix).data[_POINTS_TRACE].mode == "markers"

    def test_the_points_trace_holds_one_point_per_group(self, matrix):
        assert len(_build_figure(matrix).data[0].x) == matrix.n_groups

    def test_the_reference_line_spans_the_axis_range(self, matrix):
        line = _build_figure(matrix).data[1]

        assert tuple(line.x) == _AXIS_RANGE
        assert tuple(line.y) == _AXIS_RANGE

    def test_both_axes_start_on_the_first_model(self, matrix):
        fig = _build_figure(matrix)
        expected = f"{matrix.model_names[0]} (per-drug Pearson)"

        assert fig.layout.xaxis.title.text == expected
        assert fig.layout.yaxis.title.text == expected

    def test_axes_share_the_fixed_correlation_range(self, matrix):
        fig = _build_figure(matrix)

        assert tuple(fig.layout.xaxis.range) == _AXIS_RANGE
        assert tuple(fig.layout.yaxis.range) == _AXIS_RANGE

    def test_there_are_two_dropdown_menus(self, matrix):
        assert len(_build_figure(matrix).layout.updatemenus) == 2

    def test_the_dropdowns_are_labelled_x_and_y(self, matrix):
        texts = [annotation.text for annotation in _build_figure(matrix).layout.annotations]

        assert texts == ["x-axis model:", "y-axis model:"]

    def test_hover_names_the_group(self, matrix):
        trace = _build_figure(matrix).data[0]

        assert "Drug:" in trace.hovertemplate
        assert tuple(trace.customdata) == matrix.group_names

    def test_the_legend_is_suppressed(self, matrix):
        assert _build_figure(matrix).layout.showlegend is False

    def test_an_empty_matrix_yields_an_empty_figure(self):
        assert _build_figure(_empty_matrix()).data == ()


class TestInlinePlotlyHtml:
    def test_emits_a_div_with_the_requested_id(self, matrix):
        html = _inline_plotly_html(_build_figure(matrix), "my_div")

        assert '<div id="my_div"' in html

    def test_calls_newplot_on_that_div(self, matrix):
        html = _inline_plotly_html(_build_figure(matrix), "my_div")

        assert "Plotly.newPlot(target" in html
        assert 'getElementById("my_div")' in html

    def test_does_not_bundle_a_second_copy_of_plotly(self, matrix):
        """MultiQC's template already loads Plotly and sets ``window.Plotly``."""
        html = _inline_plotly_html(_build_figure(matrix), "my_div")

        assert "plotly.min.js" not in html
        assert len(html) < 200_000

    def test_defers_when_plotly_is_not_yet_defined(self, matrix):
        html = _inline_plotly_html(_build_figure(matrix), "my_div")

        assert 'typeof Plotly === "undefined"' in html
        assert "DOMContentLoaded" in html

    def test_the_embedded_spec_is_valid_json(self, matrix):
        html = _inline_plotly_html(_build_figure(matrix), "my_div")

        spec = json.loads(re.search(r"var spec = (\{.*\});", html, re.DOTALL).group(1))

        assert set(spec) == {"data", "layout"}
        assert len(spec["data"]) == 2

    def test_numpy_values_survive_serialisation(self, matrix):
        html = _inline_plotly_html(_build_figure(matrix), "my_div")

        spec = json.loads(re.search(r"var spec = (\{.*\});", html, re.DOTALL).group(1))

        assert len(spec["layout"]["updatemenus"]) == 2


class TestCompute:
    def test_computes_a_matrix_per_grouping(self, computed):
        assert set(computed._matrices) == {"drug", "cell_line"}

    def test_each_matrix_is_models_by_groups(self, computed):
        assert computed._matrices["drug"].values.shape == (N_MODELS, N_DRUGS)
        assert computed._matrices["cell_line"].values.shape == (N_MODELS, N_CELL_LINES)

    def test_the_figure_shows_the_first_grouping(self, computed):
        assert len(computed._fig.data[0].x) == N_DRUGS

    def test_dataset_argument_is_accepted_and_ignored(self, experiment):
        plot = ComparisonScatterVisualization()

        plot.compute(experiment, dataset=object())

        assert set(plot._matrices) == {"drug", "cell_line"}

    def test_a_single_model_yields_an_empty_figure_rather_than_an_error(self):
        result = ExperimentResult([make_run_result(model_name="Solo", fold_index=i) for i in range(2)])
        plot = ComparisonScatterVisualization()

        plot.compute(result)

        assert plot._fig.data == ()
        assert plot._matrices == {}

    def test_recomputing_replaces_the_previous_matrices(self, experiment):
        plot = ComparisonScatterVisualization()
        plot.compute(experiment)

        plot.compute(ExperimentResult([make_run_result(model_name="Solo")]))

        assert plot._matrices == {}

    def test_models_with_no_defined_correlation_are_dropped(self):
        """A constant predictor has no correlation in any group."""
        constant = make_run_result(model_name="Constant", n_pairs=20)
        constant.predictions[:] = 1.0
        result = ExperimentResult(
            [constant, make_run_result(model_name="A", n_pairs=20), make_run_result(model_name="B", n_pairs=20)]
        )
        plot = ComparisonScatterVisualization()

        plot.compute(result)

        assert "Constant" not in plot._matrices["drug"].model_names

    def test_a_grouping_left_with_one_model_is_skipped(self):
        constant = make_run_result(model_name="Constant", n_pairs=20)
        constant.predictions[:] = 1.0
        result = ExperimentResult([constant, make_run_result(model_name="A", n_pairs=20)])
        plot = ComparisonScatterVisualization()

        plot.compute(result)

        assert plot._matrices == {}

    def test_an_all_randomized_experiment_produces_nothing(self):
        result = ExperimentResult(
            [
                make_run_result(model_name="A", randomization=("gene_expression", "permutation")),
                make_run_result(model_name="B", randomization=("gene_expression", "permutation")),
            ]
        )
        plot = ComparisonScatterVisualization()

        plot.compute(result)

        assert plot._matrices == {}
        assert plot._fig.data == ()


class TestToMultiqc:
    def test_returns_one_section_per_grouping(self, computed):
        assert len(computed.to_multiqc()) == 2

    def test_anchors_are_unique_and_name_the_grouping(self, computed):
        anchors = [section.anchor for section in computed.to_multiqc()]

        assert anchors == ["dreval_comp_scatter_drug", "dreval_comp_scatter_cell_line"]

    def test_sections_are_named_after_the_grouping(self, computed):
        names = [section.anchor for section in computed.to_multiqc()]

        assert len(set(names)) == 2

    def test_sections_carry_raw_html_rather_than_a_native_plot(self, computed):
        for section in computed.to_multiqc():
            assert section.plot is None
            assert "Plotly.newPlot" in section.content

    def test_each_section_targets_its_own_div(self, computed):
        divs = [re.search(r'<div id="([^"]+)"', section.content).group(1) for section in computed.to_multiqc()]

        assert divs == ["dreval_comp_scatter_drug_div", "dreval_comp_scatter_cell_line_div"]

    def test_descriptions_report_the_model_and_group_counts(self, computed):
        description = computed.to_multiqc()[0].description

        assert f"{N_MODELS} models" in description
        assert f"{N_DRUGS} drugs" in description

    def test_returns_nothing_when_there_is_no_comparison_to_make(self):
        result = ExperimentResult([make_run_result(model_name="Solo")])
        plot = ComparisonScatterVisualization()
        plot.compute(result)

        assert plot.to_multiqc() == []

    def test_the_payload_is_bounded_by_models_times_groups(self, computed):
        """~90 bytes per (model, group) value, nowhere near per-prediction."""
        total = sum(len(section.content) for section in computed.to_multiqc())

        assert total < 200 * N_MODELS * (N_DRUGS + N_CELL_LINES) + 10_000


class TestRendering:
    def test_to_png_writes_a_png_file(self, computed, tmp_path):
        out = tmp_path / "cs.png"

        computed.to_png(out)

        assert out.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


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


class TestRegistration:
    def test_keeps_its_registry_name(self):
        assert ComparisonScatterVisualization.registry_name == "comparison_scatter"
