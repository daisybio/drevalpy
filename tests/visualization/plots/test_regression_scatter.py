"""Tests for :mod:`drevalpy.visualization.plots.regression_scatter`.

This is the only plot registered for ``ModelResult`` rather than
``ExperimentResult``, so the report adds one module per model and the anchor must
carry the model name - the ``ImageVisualization`` default anchors on
``registry_name`` alone and would collide.

The plot is a matplotlib hexbin built on :class:`matplotlib.figure.Figure`
directly, so nothing enters pyplot's global figure registry; that is asserted
explicitly, since a leak there is invisible until a 96-model run exhausts memory.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure

from drevalpy.types.results.model import ModelResult
from drevalpy.visualization.plots.regression_scatter import (
    RegressionScatterVisualization,
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
def computed(model_result) -> RegressionScatterVisualization:
    plot = RegressionScatterVisualization()
    plot.compute(model_result)
    return plot


class TestPooledPredictions:
    def test_pools_every_fold_into_two_flat_arrays(self, model_result):
        truth, prediction = _pooled_predictions(model_result)

        assert truth.shape == prediction.shape == (N_FOLDS * N_PAIRS,)

    def test_arrays_are_float64(self, model_result):
        truth, prediction = _pooled_predictions(model_result)

        assert truth.dtype == prediction.dtype == np.float64

    def test_values_come_from_the_runs(self):
        run = make_run_result(n_pairs=5)

        truth, prediction = _pooled_predictions(_model_result([run]))

        np.testing.assert_allclose(truth, run.ground_truth)
        np.testing.assert_allclose(prediction, run.predictions)

    def test_pairs_stay_aligned_across_folds(self):
        runs = [make_run_result(fold_index=i, n_pairs=4) for i in range(2)]

        truth, prediction = _pooled_predictions(_model_result(runs))

        np.testing.assert_allclose(truth[:4], runs[0].ground_truth)
        np.testing.assert_allclose(truth[4:], runs[1].ground_truth)

    def test_a_nan_on_either_side_drops_the_pair(self):
        run = make_run_result(n_pairs=6)
        run.predictions[0] = np.nan
        run.ground_truth[1] = np.nan

        truth, prediction = _pooled_predictions(_model_result([run]))

        assert truth.size == prediction.size == 4

    def test_randomized_runs_are_excluded(self):
        result = _model_result(
            [
                make_run_result(fold_index=0, n_pairs=4),
                make_run_result(fold_index=1, n_pairs=4, randomization=("gene_expression", "permutation")),
            ]
        )

        truth, _ = _pooled_predictions(result)

        assert truth.size == 4

    def test_is_empty_when_every_run_is_randomized(self):
        result = _model_result([make_run_result(randomization=("gene_expression", "permutation"))])

        truth, prediction = _pooled_predictions(result)

        assert truth.size == prediction.size == 0

    def test_retained_size_is_two_floats_per_prediction(self, model_result):
        """8 bytes per value, against the 240 bytes a point dict used to cost."""
        truth, prediction = _pooled_predictions(model_result)

        assert truth.nbytes + prediction.nbytes == 16 * N_FOLDS * N_PAIRS


class TestPearson:
    def test_a_perfect_relationship_is_one(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])

        assert _pearson(x, 2 * x) == pytest.approx(1.0)

    def test_an_inverted_relationship_is_minus_one(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])

        assert _pearson(x, -x) == pytest.approx(-1.0)

    def test_a_single_point_is_nan(self):
        assert np.isnan(_pearson(np.array([1.0]), np.array([2.0])))

    def test_an_empty_input_is_nan(self):
        assert np.isnan(_pearson(np.empty(0), np.empty(0)))

    def test_zero_variance_on_either_side_is_nan(self):
        varying = np.array([1.0, 2.0, 3.0])
        constant = np.ones(3)

        assert np.isnan(_pearson(constant, varying))
        assert np.isnan(_pearson(varying, constant))


class TestCompute:
    def test_builds_a_hexbin_collection(self, computed):
        ax = computed._fig.axes[0]

        assert any(isinstance(collection, PolyCollection) for collection in ax.collections)

    def test_draws_the_identity_line(self, computed):
        assert len(computed._fig.axes[0].lines) == 1

    def test_adds_a_density_colorbar(self, computed):
        labels = [ax.get_ylabel() for ax in computed._fig.axes[1:]]

        assert any("log scale" in label for label in labels)

    def test_title_names_the_model(self, computed, model_result):
        assert model_result.model_name in computed._fig.axes[0].get_title()

    def test_axes_are_labelled_observed_and_predicted(self, computed):
        ax = computed._fig.axes[0]

        assert (ax.get_xlabel(), ax.get_ylabel()) == ("Observed", "Predicted")

    def test_both_axes_share_one_square_range(self, computed):
        ax = computed._fig.axes[0]

        assert ax.get_xlim() == ax.get_ylim()

    def test_annotates_the_sample_count_and_the_fit(self, computed):
        text = computed._fig.axes[0].texts[0].get_text()

        assert f"n = {N_FOLDS * N_PAIRS:,}" in text
        assert "Pearson =" in text
        assert "R²" in text

    def test_stores_the_result(self, computed, model_result):
        assert computed._result is model_result

    def test_stores_the_pooled_arrays_rather_than_a_dataframe(self, computed):
        assert isinstance(computed._ground_truth, np.ndarray)
        assert isinstance(computed._predictions, np.ndarray)

    def test_dataset_argument_is_accepted_and_ignored(self, model_result):
        plot = RegressionScatterVisualization()

        plot.compute(model_result, dataset=object())

        assert plot._fig is not None

    def test_no_data_falls_back_to_a_placeholder(self):
        result = _model_result([make_run_result(randomization=("gene_expression", "permutation"))])
        plot = RegressionScatterVisualization()

        plot.compute(result)

        assert plot._fig.axes[0].texts[0].get_text() == "No data available"

    def test_a_degenerate_single_point_still_renders(self):
        run = make_run_result(n_pairs=1)
        plot = RegressionScatterVisualization()

        plot.compute(_model_result([run]))

        assert plot._fig.axes[0].get_xlim()[0] < plot._fig.axes[0].get_xlim()[1]

    def test_a_constant_cloud_gets_a_widened_range(self):
        run = make_run_result(n_pairs=6)
        run.ground_truth[:] = 1.0
        run.predictions[:] = 1.0
        plot = RegressionScatterVisualization()

        plot.compute(_model_result([run]))

        assert plot._fig.axes[0].get_xlim() == (0.5, 1.5)

    def test_recomputing_replaces_the_pooled_arrays(self, model_result):
        plot = RegressionScatterVisualization()
        plot.compute(model_result)

        plot.compute(_model_result([make_run_result(n_pairs=3)]))

        assert plot._ground_truth.size == 3


class TestPyplotIsNotUsed:
    def test_no_figure_enters_the_pyplot_registry(self, model_result):
        plt.close("all")

        RegressionScatterVisualization().compute(model_result)

        assert plt.get_fignums() == []

    def test_the_figure_is_a_plain_matplotlib_figure(self, computed):
        assert isinstance(computed._fig, Figure)
        assert computed._fig.canvas.manager is None


class TestToMultiqc:
    def test_returns_a_single_section(self, computed):
        assert len(computed.to_multiqc()) == 1

    def test_the_anchor_carries_the_model_name(self, computed, model_result):
        """One module is added per model, so a shared anchor would collide."""
        assert computed.to_multiqc()[0].anchor == f"dreval_scatter_{model_result.model_name}"

    def test_anchors_differ_between_models(self, model_result):
        other = make_model_result(model_name="RandomForest", n_folds=1, n_pairs=10)
        first, second = RegressionScatterVisualization(), RegressionScatterVisualization()
        first.compute(model_result)
        second.compute(other)

        assert first.to_multiqc()[0].anchor != second.to_multiqc()[0].anchor

    def test_the_section_is_named_after_the_model(self, computed, model_result):
        assert model_result.model_name in computed.to_multiqc()[0].name

    def test_the_description_reports_the_fold_count(self, computed):
        assert f"across {N_FOLDS} fold(s)" in computed.to_multiqc()[0].description

    def test_embeds_a_base64_png_rather_than_a_native_plot(self, computed):
        section = computed.to_multiqc()[0]

        assert section.plot is None
        assert "data:image/png;base64," in section.content

    def test_the_payload_is_an_image_not_a_point_list(self, computed):
        """Bounded by the rendered image, not by the number of predictions."""
        content = computed.to_multiqc()[0].content

        assert content.count("base64") == 1
        assert '"x"' not in content


class TestRendering:
    def test_to_png_writes_a_png_file(self, computed, tmp_path):
        out = tmp_path / "rs.png"

        computed.to_png(out)

        assert out.read_bytes().startswith(PNG_MAGIC)


class TestGuardsBeforeCompute:
    def test_to_png_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match=r"Call compute\(\) before to_png\(\)"):
            RegressionScatterVisualization().to_png(tmp_path / "rs.png")

    def test_to_multiqc_raises(self):
        with pytest.raises(RuntimeError, match=r"Call compute\(\) before to_multiqc\(\)"):
            RegressionScatterVisualization().to_multiqc()

    def test_show_raises(self):
        with pytest.raises(RuntimeError, match=r"Call compute\(\) before show\(\)"):
            RegressionScatterVisualization().show()


class TestRegistration:
    def test_keeps_its_registry_name(self):
        assert RegressionScatterVisualization.registry_name == "regression_scatter"

    def test_is_still_registered_for_model_results(self):
        from drevalpy.registry.visualization import visualization_registry

        assert visualization_registry._result_types["regression_scatter"] == "ModelResult"
