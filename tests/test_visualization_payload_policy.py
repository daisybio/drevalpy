"""Regression guard: visualization payloads must not scale with the prediction count.

``ComparisonScatterVisualization`` once retained a point dict per shared prediction per model pair
(~250 GB at 96 models), and ``RegressionScatterVisualization`` a milder version of the same. The
tests grow the number of *predictions* while holding models and groups fixed and require the
retained bytes and the report payload not to follow.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable

import numpy as np
import pytest

from drevalpy.types.results.experiment import ExperimentResult
from drevalpy.types.results.model import ModelResult
from drevalpy.visualization.plots.comparison_scatter import ComparisonScatterVisualization
from drevalpy.visualization.plots.regression_scatter import RegressionScatterVisualization
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


def _deep_size(obj: object, seen: set[int] | None = None) -> int:
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        return obj.nbytes + sys.getsizeof(obj) - obj.nbytes if obj.base is None else obj.nbytes
    total = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for key, value in obj.items():
            total += _deep_size(key, seen) + _deep_size(value, seen)
    elif isinstance(obj, str | bytes | bytearray):
        return total
    elif isinstance(obj, Iterable):
        for item in obj:
            total += _deep_size(item, seen)
    return total


def _retained_bytes(plot: object) -> int:
    return sum(_deep_size(value) for name, value in vars(plot).items() if name != "_result")


def _payload_bytes(plot) -> int:
    return sum(len(section.content or "") for section in plot.to_multiqc())


def _assert_bounded(small: int, large: int) -> None:
    assert large <= small * GROWTH_TOLERANCE + FIXED_ALLOWANCE_BYTES, (
        f"grew from {small:,} to {large:,} bytes for a {LARGE_ROWS // SMALL_ROWS}x increase in predictions"
    )


def test_comparison_scatter_state_and_payload_do_not_grow_with_predictions():
    sizes = {}
    for n_rows in (SMALL_ROWS, LARGE_ROWS):
        plot = ComparisonScatterVisualization()
        plot.compute(_experiment(n_rows))
        sizes[n_rows] = (_retained_bytes(plot), _payload_bytes(plot))
    _assert_bounded(sizes[SMALL_ROWS][0], sizes[LARGE_ROWS][0])
    _assert_bounded(sizes[SMALL_ROWS][1], sizes[LARGE_ROWS][1])

    assert plot._matrices["drug"].values.shape == (N_MODELS, N_DRUGS)
    assert plot._matrices["sample"].values.shape == (N_MODELS, N_SAMPLES)
    assert not hasattr(plot, "_pair_data")


def test_regression_scatter_payload_does_not_grow_with_predictions():
    payloads = {}
    for n_rows in (SMALL_ROWS, LARGE_ROWS):
        plot = RegressionScatterVisualization()
        plot.compute(_model_result(n_rows))
        payloads[n_rows] = _payload_bytes(plot)
    _assert_bounded(payloads[SMALL_ROWS], payloads[LARGE_ROWS])

    for array in (plot._ground_truth, plot._predictions):
        assert array.dtype == np.float64
        assert array.ndim == 1


def test_the_bound_rejects_a_per_point_payload():
    with pytest.raises(AssertionError):
        _assert_bounded(240 * SMALL_ROWS * 2, 240 * LARGE_ROWS * 2)
    _assert_bounded(10_000, 10_000)
