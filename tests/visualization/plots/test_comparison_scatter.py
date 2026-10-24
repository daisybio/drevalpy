"""Tests for :mod:`drevalpy.visualization.plots.comparison_scatter` (payload bounded by models x groups)."""

from __future__ import annotations

import json
import re
import sys

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
N_SAMPLES = 5


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


def _concatenated_experiment() -> ExperimentResult:
    """Three models over runs whose sample IDs carry a ``<study>::`` prefix."""
    runs = []
    for model in "ABC":
        for fold in range(2):
            run = make_run_result(model_name=model, fold_index=fold, n_pairs=40, n_samples=8, n_drugs=5)
            run.sample_ids = np.array([f"{'GDSC2' if i % 2 else 'CTRPv2'}::CL_{i % 8}" for i in range(40)])
            runs.append(run)
    return ExperimentResult(runs)


def _empty_matrix() -> GroupCorrelationMatrix:
    return GroupCorrelationMatrix("drug", (), (), np.empty((0, 0), dtype=np.float32))


class TestAxisLayout:
    def test_title_is_nested_so_plotly_3_keeps_it_and_range_is_fixed(self):
        layout = _axis_layout("drug", "ElasticNet")

        assert layout["title"] == {"text": "ElasticNet (per-drug Pearson)"}
        assert layout["range"] == list(_AXIS_RANGE)
        assert "per-sample" in _axis_layout("sample", "ElasticNet")["title"]["text"]


class TestDropdownButtons:
    def test_one_labelled_button_per_model_carrying_its_group_vector(self, matrix):
        buttons = _dropdown_buttons(matrix, "x")

        assert [button["label"] for button in buttons] == list(matrix.model_names)
        assert len(buttons[0]["args"][0]["x"][0]) == matrix.n_groups
        values = buttons[0]["args"][0]["x"][0]
        assert json.loads(json.dumps(values)) == values

    @pytest.mark.parametrize("axis", ["x", "y"])
    def test_buttons_restyle_the_axis_and_target_only_the_points_trace(self, matrix, axis):
        for button in _dropdown_buttons(matrix, axis):
            assert set(button["args"][0]) == {axis}
            assert set(button["args"][1]) == {f"{axis}axis"}
            assert button["args"][2] == [_POINTS_TRACE]

    def test_undefined_correlations_become_zero_rather_than_nan(self):
        matrix = GroupCorrelationMatrix("drug", ("A",), ("D_0", "D_1"), np.array([[np.nan, 0.5]], dtype=np.float32))

        values = _dropdown_buttons(matrix, "x")[0]["args"][0]["x"][0]

        assert values == [0.0, 0.5]


class TestBuildFigure:
    def test_figure_structure(self, matrix):
        fig = _build_figure(matrix)
        points, line = fig.data
        expected = f"{matrix.model_names[0]} (per-drug Pearson)"

        assert (points.mode, line.mode) == ("markers", "lines")
        assert _POINTS_TRACE == 0
        assert len(points.x) == matrix.n_groups
        assert "Drug:" in points.hovertemplate
        assert tuple(points.customdata) == matrix.group_names
        assert tuple(line.x) == tuple(line.y) == _AXIS_RANGE
        assert fig.layout.xaxis.title.text == fig.layout.yaxis.title.text == expected
        assert tuple(fig.layout.xaxis.range) == tuple(fig.layout.yaxis.range) == _AXIS_RANGE
        assert len(fig.layout.updatemenus) == 2
        assert [a.text for a in fig.layout.annotations] == ["x-axis model:", "y-axis model:"]
        assert fig.layout.showlegend is False

    def test_an_empty_matrix_yields_an_empty_figure(self):
        assert _build_figure(GroupCorrelationMatrix("drug", (), (), np.empty((0, 0), dtype=np.float32))).data == ()


class TestInlinePlotlyHtml:
    def test_emits_a_div_calling_newplot_without_bundling_plotly(self, matrix):
        html = _inline_plotly_html(_build_figure(matrix), "my_div")

        assert '<div id="my_div"' in html
        assert "Plotly.newPlot(target" in html
        assert 'getElementById("my_div")' in html
        assert "plotly.min.js" not in html
        assert len(html) < 200_000
        assert 'typeof Plotly === "undefined"' in html
        assert "DOMContentLoaded" in html

    def test_the_embedded_spec_is_valid_json(self, matrix):
        html = _inline_plotly_html(_build_figure(matrix), "my_div")

        spec = json.loads(re.search(r"var spec = (\{.*\});", html, re.DOTALL).group(1))

        assert set(spec) == {"data", "layout"}
        assert len(spec["data"]) == 2
        assert len(spec["layout"]["updatemenus"]) == 2


class TestCompute:
    def test_computes_a_models_by_groups_matrix_per_grouping(self, computed):
        assert set(computed._matrices) == {"drug", "sample"}
        assert computed._matrices["drug"].values.shape == (N_MODELS, N_DRUGS)
        assert computed._matrices["sample"].values.shape == (N_MODELS, N_SAMPLES)
        assert len(computed._fig.data[0].x) == N_DRUGS

    def test_a_concatenated_run_gets_one_column_per_study(self):
        plot = ComparisonScatterVisualization()

        plot.compute(_concatenated_experiment())

        assert plot._matrices["study"].group_names == ("CTRPv2", "GDSC2")
        assert plot._matrices["study"].values.shape == (3, 2)

    def test_a_single_model_yields_an_empty_figure_and_recomputing_replaces_matrices(self, experiment):
        plot = ComparisonScatterVisualization()
        plot.compute(experiment)

        plot.compute(ExperimentResult([make_run_result(model_name="Solo", fold_index=i) for i in range(2)]))

        assert plot._fig.data == ()
        assert plot._matrices == {}

    def test_models_with_no_defined_correlation_are_dropped(self):
        constant = make_run_result(model_name="Constant", n_pairs=20)
        constant.predictions[:] = 1.0
        runs = [constant, make_run_result(model_name="A", n_pairs=20), make_run_result(model_name="B", n_pairs=20)]
        plot = ComparisonScatterVisualization()

        plot.compute(ExperimentResult(runs))
        assert "Constant" not in plot._matrices["drug"].model_names

        plot.compute(ExperimentResult(runs[:2]))
        assert plot._matrices == {}

    def test_an_all_randomized_experiment_produces_nothing(self):
        randomization = ("gene_expression", "permutation")
        result = ExperimentResult(
            [make_run_result(model_name=name, randomization=randomization) for name in ("A", "B")]
        )
        plot = ComparisonScatterVisualization()

        plot.compute(result)

        assert plot._matrices == {}
        assert plot._fig.data == ()
        assert plot.to_multiqc() == []


class TestToMultiqc:
    def test_one_raw_html_section_per_grouping_targeting_its_own_div(self, computed):
        sections = computed.to_multiqc()

        assert [s.anchor for s in sections] == ["dreval_comp_scatter_drug", "dreval_comp_scatter_sample"]
        assert all(s.plot is None and "Plotly.newPlot" in s.content for s in sections)
        divs = [re.search(r'<div id="([^"]+)"', s.content).group(1) for s in sections]
        assert divs == ["dreval_comp_scatter_drug_div", "dreval_comp_scatter_sample_div"]
        assert f"{N_MODELS} models" in sections[0].description
        assert f"{N_DRUGS} drugs" in sections[0].description

    def test_a_concatenated_run_gets_a_study_section_last(self):
        plot = ComparisonScatterVisualization()
        plot.compute(_concatenated_experiment())

        sections = plot.to_multiqc()

        assert sections[-1].anchor == "dreval_comp_scatter_study"
        assert sections[-1].name == "Study-wise comparison"
        assert "2 studies" in sections[-1].description

    def test_the_payload_is_bounded_by_models_times_groups(self, computed):
        total = sum(len(section.content) for section in computed.to_multiqc())

        assert total < 200 * N_MODELS * (N_DRUGS + N_SAMPLES) + 10_000


class TestRenderingAndGuards:
    @pytest.mark.skipif(sys.platform == "win32", reason="Kaleido 0.2.1 hangs on Windows")
    def test_to_png_writes_a_png_file(self, computed, tmp_path):
        computed.to_png(tmp_path / "cs.png")

        assert (tmp_path / "cs.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    def test_show_delegates_to_the_plotly_figure(self, computed, monkeypatch):
        calls: list = []
        monkeypatch.setattr(BaseFigure, "show", lambda self, *a, **kw: calls.append(self))

        computed.show()

        assert calls == [computed._fig]

    @pytest.mark.parametrize("method", ["to_png", "to_multiqc", "show"])
    def test_methods_raise_before_compute(self, tmp_path, method):
        args = (tmp_path / "cs.png",) if method == "to_png" else ()

        with pytest.raises(RuntimeError, match=rf"Call compute\(\) before {method}\(\)"):
            getattr(ComparisonScatterVisualization(), method)(*args)

    def test_keeps_its_registry_name(self):
        assert ComparisonScatterVisualization.registry_name == "comparison_scatter"
