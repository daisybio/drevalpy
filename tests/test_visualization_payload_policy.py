"""Regression guard: visualization payloads must not scale with the prediction count.

``ComparisonScatterPlot`` once retained a point dict per shared prediction per model pair
(~250 GB at 96 models), and ``RegressionScatterPlot`` a milder version of the same. Plots now
keep no state, so the tests grow the number of *predictions* while holding models and groups
fixed and require the report payload and the intermediate arrays not to follow.
"""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.types.results.experiment import ExperimentResult
from drevalpy.types.results.model import ModelResult
from drevalpy.visualization.plots.comparison_scatter import ComparisonScatterPlot, _matrices
from drevalpy.visualization.plots.regression_scatter import RegressionScatterPlot, _pooled_predictions
from tests.synthetic import DEFAULT_DATASET_NAME, make_run_result

N_MODELS, N_DRUGS, N_SAMPLES = 4, 5, 4
SMALL_ROWS, LARGE_ROWS = 100, 800
GROWTH_TOLERANCE = 2.0
FIXED_ALLOWANCE_BYTES = 64_000


def _runs(model: str, n_rows: int, seed: int = 0) -> list:
    return [
        make_run_result(
            model_name=model, fold_index=fold, n_pairs=n_rows, n_samples=N_SAMPLES, n_drugs=N_DRUGS, seed=seed + fold
        )
        for fold in range(2)
    ]


def _experiment(n_rows: int) -> ExperimentResult:
    return ExperimentResult([run for i in range(N_MODELS) for run in _runs(f"Model_{i}", n_rows, seed=i * 100)])


def _model_result(n_rows: int) -> ModelResult:
    return ModelResult(model_name="Model_0", dataset_name=DEFAULT_DATASET_NAME, runs=_runs("Model_0", n_rows))


def _payload_bytes(plot, result) -> int:
    return sum(len(section.content or "") for section in plot.sections(result))


def _assert_bounded(small: int, large: int) -> None:
    assert large <= small * GROWTH_TOLERANCE + FIXED_ALLOWANCE_BYTES, (
        f"grew from {small:,} to {large:,} bytes for a {LARGE_ROWS // SMALL_ROWS}x increase in predictions"
    )


def test_comparison_scatter_payload_does_not_grow_with_predictions():
    plot = ComparisonScatterPlot()
    payloads = {n_rows: _payload_bytes(plot, _experiment(n_rows)) for n_rows in (SMALL_ROWS, LARGE_ROWS)}
    _assert_bounded(payloads[SMALL_ROWS], payloads[LARGE_ROWS])

    matrices = _matrices(_experiment(LARGE_ROWS))
    assert matrices["drug"].values.shape == (N_MODELS, N_DRUGS)
    assert matrices["sample"].values.shape == (N_MODELS, N_SAMPLES)
    assert vars(plot) == {}


def test_regression_scatter_payload_does_not_grow_with_predictions():
    plot = RegressionScatterPlot()
    payloads = {n_rows: _payload_bytes(plot, _model_result(n_rows)) for n_rows in (SMALL_ROWS, LARGE_ROWS)}
    _assert_bounded(payloads[SMALL_ROWS], payloads[LARGE_ROWS])

    for array in _pooled_predictions(_model_result(LARGE_ROWS)):
        assert array.dtype == np.float64
        assert array.ndim == 1
    assert vars(plot) == {}


def test_the_bound_rejects_a_per_point_payload():
    with pytest.raises(AssertionError):
        _assert_bounded(240 * SMALL_ROWS * 2, 240 * LARGE_ROWS * 2)
    _assert_bounded(10_000, 10_000)
