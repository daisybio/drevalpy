"""Tests for :mod:`drevalpy.visualization.plots.leaderboard`.

``_create_figure`` updates the global ``plt.rcParams`` with the dark theme and
never restores it. That is asserted as current behaviour, and every test in this
module runs inside an ``rc_context`` so the mutation cannot leak into the rest
of the suite.

The PCC panel is the plot that broke when it read ``"Pearson: normalized"`` while
``normalize()`` emits ``"Pearson"``: the column came out all-NaN and ``set_xlim``
raised ``ValueError: Axis limits cannot be NaN or Inf``. The table is pinned in
``test_leaderboard_table.py`` and the NaN-safe axis in ``test_leaderboard_draw.py``;
this file pins the panels built from them.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pytest

from drevalpy.types.results.experiment import ExperimentResult
from drevalpy.visualization.plots._leaderboard_draw import DARK_THEME
from drevalpy.visualization.plots.leaderboard import LeaderboardVisualization, _figure_geometry, _panels
from tests.synthetic import REFERENCE_MODEL, make_experiment_result, make_run_result

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(autouse=True)
def _isolate_matplotlib_state():
    """Contain the rcParams mutation in ``_create_figure`` and close figures."""
    with plt.rc_context():
        yield
    plt.close("all")


@pytest.fixture
def experiment() -> ExperimentResult:
    return make_experiment_result()


@pytest.fixture
def computed(experiment) -> LeaderboardVisualization:
    plot = LeaderboardVisualization()
    plot.compute(experiment)
    return plot


class TestFigureGeometry:
    """The 96-model production report overlapped every tick label at a fixed height."""

    def test_a_small_experiment_keeps_the_original_canvas(self):
        height, font_adder, _ = _figure_geometry(12)

        assert (height, font_adder) == (12.0, 6)

    def test_height_grows_with_the_model_count(self):
        assert _figure_geometry(96)[0] > _figure_geometry(20)[0] > 12.0 - 1e-9

    def test_every_model_gets_at_least_a_quarter_inch_of_height(self):
        height, _, _ = _figure_geometry(96)

        assert height / 96 > 0.25

    def test_the_font_shrinks_as_the_list_gets_long(self):
        assert _figure_geometry(96)[1] < _figure_geometry(40)[1] < _figure_geometry(10)[1]

    def test_height_is_capped_so_the_png_stays_writable(self):
        assert _figure_geometry(10_000)[0] == 60.0


def _visible_axes(fig) -> list:
    return [ax for ax in fig.axes if ax.axison]


class TestPanels:
    @staticmethod
    def _df(**columns):
        import pandas as pd

        base = {"PCC": [0.1], "RMSE": [1.0], "Pearson": [np.nan], "Pearson_per_drug": [np.nan]}
        return pd.DataFrame({**base, **{k: [v] for k, v in columns.items()}})

    def test_pcc_and_rmse_are_always_drawn(self):
        assert [p.column for p in _panels(self._df(), normalized=False)] == ["PCC", "RMSE"]

    def test_raw_pearson_needs_a_normalized_experiment(self):
        df = self._df(Pearson=0.8)

        assert [p.column for p in _panels(df, normalized=False)] == ["PCC", "RMSE"]
        assert [p.column for p in _panels(df, normalized=True)] == ["PCC", "RMSE", "Pearson"]

    def test_per_drug_pearson_is_the_last_panel(self):
        df = self._df(Pearson=0.8, Pearson_per_drug=0.3)

        assert [p.column for p in _panels(df, normalized=True)] == ["PCC", "RMSE", "Pearson", "Pearson_per_drug"]

    def test_the_pcc_panel_names_its_normalization(self):
        assert _panels(self._df(), normalized=True)[0].xlabel == "Normalized PCC"
        assert _panels(self._df(), normalized=False)[0].xlabel == "PCC"


class TestFourPanels:
    """The 1.6.1 layout: normalized PCC, RMSE, raw Pearson and per-drug Pearson in a 2x2 grid."""

    @pytest.fixture
    def plot(self, experiment):
        plot = LeaderboardVisualization()
        plot.compute(experiment.normalize(REFERENCE_MODEL))
        return plot

    def test_a_normalized_experiment_draws_four_panels(self, plot):
        assert [ax.get_xlabel() for ax in _visible_axes(plot._fig)] == [
            "Normalized PCC",
            "Root Mean Square Error",
            "Raw PCC",
            "Per-drug Pearson",
        ]

    def test_the_panels_are_titled_with_the_optimization_direction(self, plot):
        titles = [ax.get_title() for ax in _visible_axes(plot._fig)]

        assert titles[2] == "Raw PCC  \u2191  higher is better"
        assert titles[3] == "Per-drug Pearson  \u2191  higher is better"

    def test_the_canvas_has_two_rows(self, plot):
        assert plot._fig.get_size_inches()[1] == pytest.approx(24.0)

    def test_the_reference_model_is_ranked_as_a_baseline(self, plot):
        labels = {label.get_text() for label in plot._fig.axes[0].get_yticklabels()}

        assert REFERENCE_MODEL in labels

    def test_the_raw_pearson_panel_ranks_by_raw_pearson(self, plot, experiment):
        raw = {m.model_name: np.mean([r.metrics["Pearson"] for r in m.runs]) for m in experiment.models}
        labels = [label.get_text() for label in plot._fig.axes[2].get_yticklabels()]

        assert labels == sorted(raw, key=raw.__getitem__, reverse=True)

    def test_a_plain_experiment_without_raw_duplicate_draws_three_panels(self, computed):
        assert [ax.get_xlabel() for ax in _visible_axes(computed._fig)] == [
            "PCC",
            "Root Mean Square Error",
            "Per-drug Pearson",
        ]

    def test_two_panels_use_a_single_row(self):
        result = ExperimentResult(
            [
                make_run_result(
                    model_name=name, fold_index=i, n_pairs=1, n_drugs=1, metrics={"Pearson": 0.5, "RMSE": 1.0}
                )
                for name in ("A", "B")
                for i in range(2)
            ]
        )
        plot = LeaderboardVisualization()

        plot.compute(result)

        assert len(plot._fig.axes) == 2
        assert plot._fig.get_size_inches()[1] == pytest.approx(12.0)


class TestCompute:
    def test_the_first_two_panels_are_pcc_and_rmse(self, computed):
        assert [ax.get_xlabel() for ax in computed._fig.axes[:2]] == ["PCC", "Root Mean Square Error"]

    def test_left_panel_ranks_by_pearson(self, computed):
        """The experiment is not normalized, so the label must not claim it is."""
        assert computed._fig.axes[0].get_xlabel() == "PCC"

    def test_left_panel_says_normalized_once_it_is(self, experiment):
        plot = LeaderboardVisualization()

        plot.compute(experiment.normalize(REFERENCE_MODEL))

        assert plot._fig.axes[0].get_xlabel() == "Normalized PCC"
        assert "Normalized Pearson" in plot._fig.axes[0].get_title()

    def test_right_panel_ranks_by_rmse(self, computed):
        assert computed._fig.axes[1].get_xlabel() == "Root Mean Square Error"

    def test_every_model_gets_a_tick_label(self, computed, experiment):
        labels = {label.get_text() for label in computed._fig.axes[0].get_yticklabels()}

        assert labels == set(experiment.model_names)

    def test_panels_are_titled_with_the_optimization_direction(self, computed):
        titles = [ax.get_title() for ax in computed._fig.axes]

        assert "higher is better" in titles[0]
        assert "lower is better" in titles[1]

    def test_stores_the_result(self, computed, experiment):
        assert computed._result is experiment

    def test_models_outside_the_podium_keep_the_default_label_color(self):
        experiment = make_experiment_result(n_models=5, n_folds=2)
        plot = LeaderboardVisualization()

        plot.compute(experiment)

        colors = [label.get_color() for label in plot._fig.axes[0].get_yticklabels()]
        assert DARK_THEME["text"] in colors

    def test_falls_back_to_a_placeholder_when_there_is_no_data(self):
        result = ExperimentResult([make_run_result(randomization=("gene_expression", "permutation"))])
        plot = LeaderboardVisualization()

        plot.compute(result)

        assert len(plot._fig.axes) == 1
        assert plot._fig.axes[0].texts[0].get_text() == "No data available for leaderboard"

    def test_a_normalized_experiment_renders(self, experiment):
        """The regression: this used to raise ``Axis limits cannot be NaN or Inf``."""
        plot = LeaderboardVisualization()

        plot.compute(experiment.normalize(REFERENCE_MODEL))

        assert len(_visible_axes(plot._fig)) == 4

    def test_models_without_a_pcc_value_are_dropped_from_that_panel_only(self):
        result = ExperimentResult(
            [
                make_run_result(model_name="WithPcc", fold_index=i, metrics={"Pearson": 0.5, "RMSE": 1.0})
                for i in range(2)
            ]
            + [make_run_result(model_name="NoPcc", fold_index=i, metrics={"RMSE": 2.0}) for i in range(2)]
        )
        plot = LeaderboardVisualization()

        plot.compute(result)

        pcc_labels = {label.get_text() for label in plot._fig.axes[0].get_yticklabels()}
        rmse_labels = {label.get_text() for label in plot._fig.axes[1].get_yticklabels()}
        assert pcc_labels == {"WithPcc"}
        assert rmse_labels == {"WithPcc", "NoPcc"}

    def test_skips_with_a_warning_when_no_metric_has_a_finite_value(self, caplog):
        result = ExperimentResult(
            [make_run_result(model_name="A", fold_index=i, metrics={"MSE": 0.5}) for i in range(2)]
        )
        plot = LeaderboardVisualization()

        with caplog.at_level(logging.WARNING, logger="drevalpy.visualization.plots.leaderboard"):
            plot.compute(result)

        assert plot._fig.axes[0].texts[0].get_text() == "No data available for leaderboard"
        assert any("no finite Pearson/RMSE values" in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize("n_folds", [3, 5])
    def test_the_subtitle_names_the_actual_fold_count(self, n_folds):
        plot = LeaderboardVisualization()

        plot.compute(make_experiment_result(n_models=3, n_folds=n_folds, split_mode="LSO"))

        assert any(f"{n_folds}-Fold Leave-Sample-Out Cross Validation" in t.get_text() for t in plot._fig.texts)

    def test_the_subtitle_of_a_leave_study_out_experiment_has_no_fold_count(self):
        plot = LeaderboardVisualization()

        plot.compute(make_experiment_result(n_models=3, n_folds=3, split_mode="LStO"))

        assert any(t.get_text().endswith("Leave-Study-Out Cross Validation") for t in plot._fig.texts)

    def test_a_large_experiment_gets_a_taller_canvas(self):
        """The 96-model report is the case a fixed 12-inch figure could not render."""
        plot = LeaderboardVisualization()

        plot.compute(make_experiment_result(n_models=40, n_folds=2))

        assert plot._fig.get_size_inches()[1] > 12.0

    def test_mutates_the_global_rcparams_with_the_dark_theme(self, experiment):
        plt.rcParams["figure.facecolor"] = "white"

        LeaderboardVisualization().compute(experiment)

        assert plt.rcParams["figure.facecolor"] == DARK_THEME["background"]


class TestRendering:
    def test_to_png_writes_a_png_file(self, computed, tmp_path):
        out = tmp_path / "leaderboard.png"

        computed.to_png(out)

        assert out.read_bytes().startswith(PNG_MAGIC)

    def test_to_multiqc_embeds_the_figure_under_the_registry_name(self, computed):
        sections = computed.to_multiqc()

        assert len(sections) == 1
        assert (sections[0].name, sections[0].anchor) == ("leaderboard", "leaderboard")
        assert sections[0].content is not None
        assert "data:image/png;base64," in sections[0].content


class TestGuardsBeforeCompute:
    def test_to_png_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match=r"Call compute\(\) before to_png\(\)"):
            LeaderboardVisualization().to_png(tmp_path / "l.png")

    def test_to_multiqc_raises(self):
        with pytest.raises(RuntimeError, match=r"Call compute\(\) before to_multiqc\(\)"):
            LeaderboardVisualization().to_multiqc()

    def test_show_raises(self):
        with pytest.raises(RuntimeError, match=r"Call compute\(\) before show\(\)"):
            LeaderboardVisualization().show()
