"""Tests for :mod:`drevalpy.visualization.plots.regression_scatter` (the only ``ModelResult`` plot)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure

from drevalpy.types.results.model import ModelResult
from drevalpy.visualization.base import save_png
from drevalpy.visualization.plots.regression_scatter import (
    RegressionScatterPlot,
    _pearson,
    _pooled_predictions,
)
from tests.synthetic import DEFAULT_DATASET_NAME, make_model_result, make_run_result

N_FOLDS = 2
N_PAIRS = 40
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _model_result(runs) -> ModelResult:
    return ModelResult(model_name="ElasticNet", dataset_name=DEFAULT_DATASET_NAME, runs=runs)


@pytest.fixture(scope="module")
def model_result() -> ModelResult:
    return make_model_result(n_folds=N_FOLDS, n_pairs=N_PAIRS)


@pytest.fixture(scope="module")
def figure(model_result) -> Figure:
    return RegressionScatterPlot().figure(model_result)


class TestPooledPredictions:
    def test_pools_every_fold_into_aligned_flat_float64_arrays(self, model_result):
        truth, prediction = _pooled_predictions(model_result)

        assert truth.shape == prediction.shape == (N_FOLDS * N_PAIRS,)
        assert truth.dtype == prediction.dtype == np.float64
        assert truth.nbytes + prediction.nbytes == 16 * N_FOLDS * N_PAIRS

    def test_values_come_from_the_runs_in_fold_order(self):
        runs = [make_run_result(fold_index=i, n_pairs=4) for i in range(2)]

        truth, prediction = _pooled_predictions(_model_result(runs))

        np.testing.assert_allclose(truth[:4], runs[0].ground_truth)
        np.testing.assert_allclose(truth[4:], runs[1].ground_truth)
        np.testing.assert_allclose(prediction[4:], runs[1].predictions)

    def test_a_nan_on_either_side_drops_the_pair(self):
        run = make_run_result(n_pairs=6)
        run.predictions[0] = np.nan
        run.ground_truth[1] = np.nan

        truth, prediction = _pooled_predictions(_model_result([run]))

        assert truth.size == prediction.size == 4

    def test_randomized_runs_are_excluded(self):
        randomization = ("gene_expression", "permutation")
        runs = [
            make_run_result(fold_index=0, n_pairs=4),
            make_run_result(fold_index=1, n_pairs=4, randomization=randomization),
        ]

        assert _pooled_predictions(_model_result(runs))[0].size == 4
        assert _pooled_predictions(_model_result(runs[1:]))[0].size == 0


class TestPearson:
    def test_perfect_and_inverted_relationships(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])

        assert _pearson(x, 2 * x) == pytest.approx(1.0)
        assert _pearson(x, -x) == pytest.approx(-1.0)

    @pytest.mark.parametrize(
        "x, y",
        [([1.0], [2.0]), ([], []), ([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]), ([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])],
    )
    def test_undefined_cases_are_nan(self, x, y):
        assert np.isnan(_pearson(np.array(x), np.array(y)))


class TestFigure:
    def test_draws_hexbin_identity_line_colorbar_and_labels(self, figure, model_result):
        ax = figure.axes[0]

        assert any(isinstance(collection, PolyCollection) for collection in ax.collections)
        assert len(ax.lines) == 1
        assert any("log scale" in other.get_ylabel() for other in figure.axes[1:])
        assert model_result.model_name in ax.get_title()
        assert (ax.get_xlabel(), ax.get_ylabel()) == ("Observed", "Predicted")
        assert ax.get_xlim() == ax.get_ylim()
        text = ax.texts[0].get_text()
        assert f"n = {N_FOLDS * N_PAIRS:,}" in text
        assert "Pearson =" in text
        assert "R²" in text

    def test_no_data_falls_back_to_a_placeholder(self):
        fig = RegressionScatterPlot().figure(
            _model_result([make_run_result(randomization=("gene_expression", "permutation"))])
        )

        assert fig.axes[0].texts[0].get_text() == "No data available"

    def test_a_degenerate_single_point_still_renders(self):
        fig = RegressionScatterPlot().figure(_model_result([make_run_result(n_pairs=1)]))

        low, high = fig.axes[0].get_xlim()
        assert low < high

    def test_a_constant_cloud_gets_a_widened_range(self):
        run = make_run_result(n_pairs=6)
        run.ground_truth[:] = 1.0
        run.predictions[:] = 1.0
        fig = RegressionScatterPlot().figure(_model_result([run]))

        assert fig.axes[0].get_xlim() == (0.5, 1.5)

    def test_one_instance_draws_each_result_on_its_own(self, model_result):
        plot = RegressionScatterPlot()
        plot.figure(model_result)

        fig = plot.figure(_model_result([make_run_result(n_pairs=3)]))

        assert "n = 3\n" in fig.axes[0].texts[0].get_text()

    def test_no_figure_enters_the_pyplot_registry(self, model_result):
        plt.close("all")

        fig = RegressionScatterPlot().figure(model_result)

        assert plt.get_fignums() == []
        assert isinstance(fig, Figure)
        assert fig.canvas.manager is None


class TestSections:
    def test_one_section_with_a_model_specific_anchor_and_an_embedded_png(self, model_result):
        plot = RegressionScatterPlot()
        (section,) = plot.sections(model_result)
        (other,) = plot.sections(make_model_result(model_name="RandomForest", n_folds=1, n_pairs=10))

        assert section.anchor == f"dreval_scatter_{model_result.model_name}"
        assert other.anchor != section.anchor
        assert model_result.model_name in section.name
        assert f"across {N_FOLDS} fold(s)" in section.description
        assert section.plot is None
        assert section.content is not None
        assert "data:image/png;base64," in section.content
        assert section.content.count("base64") == 1
        assert '"x"' not in section.content


class TestRendering:
    def test_save_png_writes_a_png_file(self, model_result, tmp_path):
        save_png(RegressionScatterPlot(), model_result, tmp_path / "rs.png")

        assert (tmp_path / "rs.png").read_bytes().startswith(PNG_MAGIC)

    def test_is_drawn_per_model(self):
        assert RegressionScatterPlot.name == "regression_scatter"
        assert RegressionScatterPlot.result_type is ModelResult
