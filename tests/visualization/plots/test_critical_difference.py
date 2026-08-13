"""Tests for :mod:`drevalpy.visualization.plots.critical_difference`.

The Friedman test underpinning this plot needs at least three models with equal
fold counts. ``MULTIPLE_MODELS`` only demands two, so the plot skips with a
warning rather than letting SciPy raise; that and the modal-fold-count filtering
in ``_create_figure`` are asserted explicitly.
"""

from __future__ import annotations

import logging

import matplotlib.colors
import matplotlib.pyplot as plt
import pandas as pd
import plotly.colors as pc
import pytest

from drevalpy.types.results.experiment import ExperimentResult
from drevalpy.visualization.plots.critical_difference import (
    CriticalDifferenceVisualization,
    _build_cd_df,
    _crossbar_sets_from_adjacency,
    _draw_crossbars,
    _generate_discrete_palette,
    _nonsignificant_adjacency,
)
from tests.synthetic import NORMALIZED_METRIC, make_experiment_result, make_run_result

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

MODELS = ["A", "B", "C"]


def _sig_matrix(a_vs_b: float, a_vs_c: float, b_vs_c: float) -> pd.DataFrame:
    """Symmetric p-value matrix in the shape ``posthoc_conover_friedman`` returns."""
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
def experiment() -> ExperimentResult:
    return make_experiment_result()


@pytest.fixture
def computed(experiment) -> CriticalDifferenceVisualization:
    plot = CriticalDifferenceVisualization()
    plot.compute(experiment)
    return plot


class TestBuildCdDf:
    def test_columns_are_algorithm_split_and_metric(self, experiment):
        df = _build_cd_df(experiment, "MSE")

        assert list(df.columns) == ["algorithm", "CV_split", "MSE"]

    def test_has_one_row_per_run(self, experiment):
        df = _build_cd_df(experiment, "MSE")

        assert len(df) == sum(m.n_folds for m in experiment.models)

    def test_randomized_runs_are_excluded(self):
        result = ExperimentResult(
            [
                make_run_result(model_name="ElasticNet"),
                make_run_result(model_name="RandomForest", randomization=("gene_expression", "permutation")),
            ]
        )

        df = _build_cd_df(result, "MSE")

        assert df["algorithm"].tolist() == ["ElasticNet"]

    def test_missing_metrics_leave_no_rows_to_rank(self, experiment):
        """NaN rows are dropped, so an absent metric yields an empty frame."""
        df = _build_cd_df(experiment, "NotAMetric")

        assert df.empty
        assert list(df.columns) == ["algorithm", "CV_split", "NotAMetric"]

    def test_the_legacy_normalized_spelling_is_resolved(self):
        result = ExperimentResult([make_run_result(metrics={NORMALIZED_METRIC: 0.42})])

        df = _build_cd_df(result, "Pearson")

        assert df["Pearson"].tolist() == [0.42]

    def test_the_plain_name_wins_over_the_suffixed_one(self):
        result = ExperimentResult([make_run_result(metrics={"Pearson": 0.9, NORMALIZED_METRIC: 0.1})])

        df = _build_cd_df(result, "Pearson")

        assert df["Pearson"].tolist() == [0.9]

    def test_is_empty_when_every_run_is_randomized(self):
        result = ExperimentResult([make_run_result(randomization=("gene_expression", "permutation"))])

        assert _build_cd_df(result, "MSE").empty


class TestGenerateDiscretePalette:
    def test_short_requests_slice_the_base_palette(self):
        assert _generate_discrete_palette(3) == list(pc.qualitative.D3[:3])

    def test_returns_exactly_the_requested_number_of_colors(self):
        assert len(_generate_discrete_palette(17)) == 17

    def test_interpolates_beyond_the_base_palette(self):
        colors = _generate_discrete_palette(len(pc.qualitative.D3) + 2)

        assert colors[0] == matplotlib.colors.to_hex(pc.qualitative.D3[0])
        assert colors[-1] == matplotlib.colors.to_hex(pc.qualitative.D3[-1])

    def test_interpolated_colors_are_hex_strings(self):
        assert all(c.startswith("#") and len(c) == 7 for c in _generate_discrete_palette(15))

    def test_zero_colors_yields_an_empty_palette(self):
        assert _generate_discrete_palette(0) == []


class TestNonsignificantAdjacency:
    def test_marks_non_significant_pairs_as_adjacent(self):
        adjacency = _nonsignificant_adjacency(_sig_matrix(a_vs_b=0.9, a_vs_c=0.9, b_vs_c=0.9))

        assert bool(adjacency.loc["A", "B"]) is True

    def test_significant_pairs_are_not_adjacent(self):
        adjacency = _nonsignificant_adjacency(_sig_matrix(a_vs_b=0.001, a_vs_c=0.9, b_vs_c=0.9))

        assert bool(adjacency.loc["A", "B"]) is False

    def test_diagonal_is_marked_adjacent(self):
        """``sign_array`` writes -1 on the diagonal, so ``1 - (-1)`` is truthy."""
        adjacency = _nonsignificant_adjacency(_sig_matrix(a_vs_b=0.001, a_vs_c=0.001, b_vs_c=0.001))

        assert adjacency.to_numpy().diagonal().all()

    def test_labels_and_dtype_are_preserved(self):
        adjacency = _nonsignificant_adjacency(_sig_matrix(a_vs_b=0.9, a_vs_c=0.9, b_vs_c=0.9))

        assert list(adjacency.index) == list(adjacency.columns) == MODELS
        assert adjacency.dtypes.eq(bool).all()

    def test_does_not_mutate_the_input(self):
        sig = _sig_matrix(a_vs_b=0.001, a_vs_c=0.9, b_vs_c=0.9)
        before = sig.copy()

        _nonsignificant_adjacency(sig)

        pd.testing.assert_frame_equal(sig, before)


class TestCrossbarSetsFromAdjacency:
    def test_each_model_is_grouped_with_its_non_different_peers(self):
        adjacency = _nonsignificant_adjacency(_sig_matrix(a_vs_b=0.9, a_vs_c=0.001, b_vs_c=0.001))

        sets = _crossbar_sets_from_adjacency(adjacency)

        assert sets["A"] == {"A", "B"}
        assert sets["C"] == {"C"}

    def test_all_significant_yields_only_singletons(self):
        adjacency = _nonsignificant_adjacency(_sig_matrix(a_vs_b=0.001, a_vs_c=0.001, b_vs_c=0.001))

        assert _crossbar_sets_from_adjacency(adjacency) == {"A": {"A"}, "B": {"B"}, "C": {"C"}}

    def test_none_significant_groups_everything(self):
        adjacency = _nonsignificant_adjacency(_sig_matrix(a_vs_b=0.9, a_vs_c=0.9, b_vs_c=0.9))

        assert _crossbar_sets_from_adjacency(adjacency)["A"] == set(MODELS)


class TestDrawCrossbars:
    def test_singleton_groups_draw_nothing(self):
        _, ax = plt.subplots()
        ranks = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0})
        sets = {name: {name} for name in MODELS}

        ypos = _draw_crossbars(ax, ranks, sets, dict.fromkeys(MODELS, "#000000"), {})

        assert len(ax.lines) == 0
        assert ypos == -0.5

    def test_each_group_gets_one_line_and_shifts_the_offset(self):
        _, ax = plt.subplots()
        ranks = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0})
        sets = {"A": {"A", "B"}, "B": {"A", "B"}, "C": {"C"}}

        ypos = _draw_crossbars(ax, ranks, sets, dict.fromkeys(MODELS, "#000000"), {})

        assert len(ax.lines) == 2
        assert ypos == -1.5

    def test_lines_use_the_model_color(self):
        _, ax = plt.subplots()
        ranks = pd.Series({"A": 1.0, "B": 2.0})
        sets = {"A": {"A", "B"}, "B": {"A", "B"}}

        _draw_crossbars(ax, ranks, sets, {"A": "#ff0000", "B": "#00ff00"}, {})

        assert matplotlib.colors.to_hex(ax.lines[0].get_color()) == "#ff0000"


class TestCompute:
    def test_title_names_the_metric_and_the_friedman_p_value(self, computed):
        title = computed._fig.axes[0].get_title()

        assert "Critical Difference Diagram: Metric: MSE." in title
        assert "Friedman-Chi2 p-value" in title

    def test_every_model_is_placed_on_the_rank_axis(self, computed, experiment):
        labels = {text.get_text() for text in computed._fig.axes[0].texts}

        assert len(labels) == len(experiment.models)
        assert all(any(name in label for label in labels) for name in experiment.model_names)

    def test_one_marker_is_drawn_per_model(self, computed, experiment):
        assert len(computed._fig.axes[0].collections) == len(experiment.models)

    def test_rank_axis_hides_the_y_axis(self, computed):
        assert computed._fig.axes[0].yaxis.get_visible() is False

    def test_metric_is_configurable(self, experiment):
        plot = CriticalDifferenceVisualization()

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
        result = ExperimentResult([make_run_result(randomization=("gene_expression", "permutation"))])
        plot = CriticalDifferenceVisualization()

        plot.compute(result)

        assert plot._fig.axes[0].texts[0].get_text() == "No data available"

    def test_fewer_than_three_models_is_skipped_instead_of_raising(self, caplog):
        """``MULTIPLE_MODELS`` admits a two-model experiment, which Friedman cannot rank."""
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

    def test_the_legacy_normalized_spelling_is_still_ranked(self):
        """Results written before ``normalize()`` used plain names keep working."""
        runs = [
            make_run_result(
                model_name=name,
                fold_index=fold,
                metrics={NORMALIZED_METRIC: 0.5 + 0.1 * index + 0.01 * fold},
            )
            for index, name in enumerate(("A", "B", "C"))
            for fold in range(3)
        ]
        plot = CriticalDifferenceVisualization()

        plot.compute(ExperimentResult(runs), metric="Pearson")

        assert len(plot._fig.axes[0].collections) == 3

    def test_dataset_argument_is_accepted_and_ignored(self, experiment):
        plot = CriticalDifferenceVisualization()

        plot.compute(experiment, dataset=object())

        assert plot._fig is not None


class TestRendering:
    def test_to_png_writes_a_png_file(self, computed, tmp_path):
        out = tmp_path / "cd.png"

        computed.to_png(out)

        assert out.read_bytes().startswith(PNG_MAGIC)

    def test_to_multiqc_embeds_the_figure_under_the_registry_name(self, computed):
        sections = computed.to_multiqc()

        assert len(sections) == 1
        assert (sections[0].name, sections[0].anchor) == ("critical_difference", "critical_difference")
        assert sections[0].content is not None
        assert "data:image/png;base64," in sections[0].content


class TestGuardsBeforeCompute:
    def test_to_png_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match=r"Call compute\(\) before to_png\(\)"):
            CriticalDifferenceVisualization().to_png(tmp_path / "c.png")

    def test_to_multiqc_raises(self):
        with pytest.raises(RuntimeError, match=r"Call compute\(\) before to_multiqc\(\)"):
            CriticalDifferenceVisualization().to_multiqc()

    def test_show_raises(self):
        with pytest.raises(RuntimeError, match=r"Call compute\(\) before show\(\)"):
            CriticalDifferenceVisualization().show()

    def test_metric_defaults_to_mse(self):
        assert CriticalDifferenceVisualization()._metric == "MSE"
