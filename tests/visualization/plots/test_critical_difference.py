"""Tests for :mod:`drevalpy.visualization.plots.critical_difference` (Friedman needs >= 3 models)."""

from __future__ import annotations

import logging

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import pandas as pd
import plotly.colors as pc
import pytest
from upath import UPath

from drevalpy.types.results.experiment import ExperimentResult
from drevalpy.visualization.plots import critical_difference
from drevalpy.visualization.plots.critical_difference import (
    CriticalDifferenceVisualization,
    _build_cd_df,
    _crossbar_sets_from_adjacency,
    _draw_crossbars,
    _generate_discrete_palette,
    _nonsignificant_adjacency,
)
from tests._trusted_subprocess import run_trusted_python
from tests.synthetic import REFERENCE_MODEL, make_experiment_result, make_run_result

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

MODELS = ["A", "B", "C"]


def _sig_matrix(a_vs_b: float, a_vs_c: float, b_vs_c: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            [1.0, a_vs_b, a_vs_c],
            [a_vs_b, 1.0, b_vs_c],
            [a_vs_c, b_vs_c, 1.0],
        ],
        index=MODELS,
        columns=MODELS,
    )


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def computed(experiment) -> CriticalDifferenceVisualization:
    plot = CriticalDifferenceVisualization()
    plot.compute(experiment)
    return plot


class TestBuildCdDf:
    def test_one_row_per_run_in_the_expected_columns(self, experiment):
        df = _build_cd_df(experiment, "MSE")

        assert list(df.columns) == ["algorithm", "CV_split", "MSE"]
        assert len(df) == sum(m.n_folds for m in experiment.models)

    def test_randomized_runs_are_excluded(self):
        randomization = ("gene_expression", "permutation")
        result = ExperimentResult(
            [
                make_run_result(model_name="ElasticNet"),
                make_run_result(model_name="RandomForest", randomization=randomization),
            ]
        )

        assert _build_cd_df(result, "MSE")["algorithm"].tolist() == ["ElasticNet"]
        assert _build_cd_df(ExperimentResult(result.models[1].runs), "MSE").empty

    def test_missing_metrics_leave_no_rows_to_rank(self, experiment):
        df = _build_cd_df(experiment, "NotAMetric")

        assert df.empty
        assert list(df.columns) == ["algorithm", "CV_split", "NotAMetric"]

    def test_the_reference_model_of_a_normalized_experiment_takes_part(self, experiment):
        df = _build_cd_df(experiment.normalize(REFERENCE_MODEL), "Pearson")

        assert len(df) == sum(m.n_folds for m in experiment.models)
        assert (df[df["algorithm"] == REFERENCE_MODEL]["Pearson"] == 0.0).all()


class TestGenerateDiscretePalette:
    def test_short_requests_slice_the_base_palette(self):
        assert _generate_discrete_palette(3) == list(pc.qualitative.D3[:3])
        assert _generate_discrete_palette(0) == []

    def test_longer_requests_interpolate_between_the_palette_ends_as_hex(self):
        colors = _generate_discrete_palette(17)

        assert len(colors) == 17
        assert colors[0] == matplotlib.colors.to_hex(pc.qualitative.D3[0])
        assert colors[-1] == matplotlib.colors.to_hex(pc.qualitative.D3[-1])
        assert all(c.startswith("#") and len(c) == 7 for c in colors)


class TestAdjacencyAndCrossbars:
    def test_adjacency_marks_non_significant_pairs_and_the_diagonal_without_mutating(self):
        sig = _sig_matrix(a_vs_b=0.9, a_vs_c=0.001, b_vs_c=0.001)
        before = sig.copy()

        adjacency = _nonsignificant_adjacency(sig)

        assert bool(adjacency.loc["A", "B"]) is True
        assert bool(adjacency.loc["A", "C"]) is False
        assert adjacency.to_numpy().diagonal().all()
        assert list(adjacency.index) == list(adjacency.columns) == MODELS
        assert adjacency.dtypes.eq(bool).all()
        pd.testing.assert_frame_equal(sig, before)

    @pytest.mark.parametrize(
        "p_values, expected",
        [
            ((0.9, 0.001, 0.001), {"A": {"A", "B"}, "B": {"A", "B"}, "C": {"C"}}),
            ((0.001, 0.001, 0.001), {"A": {"A"}, "B": {"B"}, "C": {"C"}}),
            ((0.9, 0.9, 0.9), dict.fromkeys(MODELS, set(MODELS))),
        ],
    )
    def test_each_model_is_grouped_with_its_non_different_peers(self, p_values, expected):
        adjacency = _nonsignificant_adjacency(_sig_matrix(*p_values))

        assert _crossbar_sets_from_adjacency(adjacency) == expected

    def test_singleton_groups_draw_nothing_and_each_group_draws_one_line_in_the_model_color(self):
        ranks = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0})
        colors = {"A": "#ff0000", "B": "#00ff00", "C": "#0000ff"}
        _, ax = plt.subplots()

        assert _draw_crossbars(ax, ranks, {name: {name} for name in MODELS}, colors, {}) == -0.5
        assert len(ax.lines) == 0

        ypos = _draw_crossbars(ax, ranks, {"A": {"A", "B"}, "B": {"A", "B"}, "C": {"C"}}, colors, {})

        assert ypos == -1.5
        assert len(ax.lines) == 2
        assert matplotlib.colors.to_hex(ax.lines[0].get_color()) == "#ff0000"


class TestCompute:
    def test_title_markers_and_labels(self, computed, experiment):
        ax = computed._fig.axes[0]
        labels = {text.get_text() for text in ax.texts}

        assert "Critical Difference Diagram: Metric: MSE." in ax.get_title()
        assert "Friedman-Chi2 p-value" in ax.get_title()
        assert len(labels) == len(experiment.models)
        assert all(any(name in label for label in labels) for name in experiment.model_names)
        assert len(ax.collections) == len(experiment.models)
        assert ax.yaxis.get_visible() is False

    def test_metric_is_configurable_and_defaults_to_mse(self, experiment):
        plot = CriticalDifferenceVisualization()
        assert plot._metric == "MSE"

        plot.compute(experiment, metric="Pearson")

        assert plot._metric == "Pearson"
        assert "Metric: Pearson." in plot._fig.axes[0].get_title()

    def test_models_with_a_minority_fold_count_are_dropped(self):
        runs = [
            make_run_result(model_name=name, fold_index=fold)
            for name, n_folds in (("A", 3), ("B", 3), ("C", 3), ("D", 2))
            for fold in range(n_folds)
        ]
        plot = CriticalDifferenceVisualization()

        plot.compute(ExperimentResult(runs))

        labels = {text.get_text() for text in plot._fig.axes[0].texts}
        assert len(labels) == 3
        assert not any("D" in label for label in labels)

    def test_falls_back_to_a_placeholder_when_there_is_no_data(self):
        plot = CriticalDifferenceVisualization()

        plot.compute(ExperimentResult([make_run_result(randomization=("gene_expression", "permutation"))]))

        assert plot._fig.axes[0].texts[0].get_text() == "No data available"

    def test_fewer_than_three_models_is_skipped_instead_of_raising(self, caplog):
        plot = CriticalDifferenceVisualization()

        with caplog.at_level(logging.WARNING, logger="drevalpy.visualization.plots.critical_difference"):
            plot.compute(make_experiment_result(n_models=2))

        assert plot._fig.axes[0].texts[0].get_text() == "Not enough comparable models"
        assert any("only 2 models share" in r.getMessage() for r in caplog.records)

    def test_a_metric_absent_from_every_run_is_skipped_with_a_warning(self, experiment, caplog):
        plot = CriticalDifferenceVisualization()

        with caplog.at_level(logging.WARNING, logger="drevalpy.visualization.plots.critical_difference"):
            plot.compute(experiment, metric="NotAMetric")

        assert plot._fig.axes[0].texts[0].get_text() == "No data available"
        assert any("no finite NotAMetric values" in r.getMessage() for r in caplog.records)


class TestRenderingAndGuards:
    def test_to_png_writes_a_png_file(self, computed, tmp_path):
        computed.to_png(tmp_path / "cd.png")

        assert (tmp_path / "cd.png").read_bytes().startswith(PNG_MAGIC)

    def test_to_multiqc_embeds_the_figure_under_the_registry_name(self, computed):
        (section,) = computed.to_multiqc()

        assert (section.name, section.anchor) == ("critical_difference", "critical_difference")
        assert "data:image/png;base64," in section.content


class TestNoGlobalBackendSwitch:
    """Registering the builtins imports this module; a module-scope ``matplotlib.use`` broke notebooks."""

    pytestmark = pytest.mark.slow

    def test_the_module_does_not_call_matplotlib_use(self):
        assert "matplotlib.use(" not in UPath(critical_difference.__file__).read_text(encoding="utf-8")

    def test_importing_drevalpy_leaves_the_backend_alone(self):
        script = (
            "import matplotlib\n"
            "before = matplotlib.get_backend()\n"
            "import drevalpy  # noqa: F401\n"
            "print(before, matplotlib.get_backend())\n"
        )

        result = run_trusted_python(script)

        assert result.returncode == 0, result.stderr
        before, after = result.stdout.split()
        assert before == after

    def test_the_figure_renders_under_the_agg_backend(self, experiment, tmp_path):
        original = matplotlib.get_backend()
        matplotlib.use("agg")
        try:
            plot = CriticalDifferenceVisualization()
            plot.compute(experiment)
            plot.to_png(tmp_path / "headless.png")
        finally:
            matplotlib.use(original)

        assert (tmp_path / "headless.png").read_bytes().startswith(PNG_MAGIC)
