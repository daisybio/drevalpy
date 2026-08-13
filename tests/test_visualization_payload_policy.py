"""Policy test: visualization payloads must not scale with the prediction count.

The OOM this guard exists to prevent came from
``ComparisonScatterVisualization.compute`` retaining one ``{"x": …, "y": …}``
dict per shared prediction *per model pair*: at 96 models that is
``C(96,2) x 231,080`` dicts, roughly 250 GB, and the process was killed inside the
first plot. ``RegressionScatterVisualization`` had the milder version of the same
shape, 5.33 GB of point dicts across 96 models.

Both are now bounded by ``models x groups`` and by the rendered image
respectively. The tests below assert that empirically rather than by inspecting
the source: they grow the number of *predictions* while holding models and groups
fixed, and require the retained bytes and the report payload not to follow. A
reintroduced per-sample payload fails here even if it is spelled differently.

This lives at the ``tests/`` root alongside the other cross-cutting guards
(``test_architecture_policy.py``, ``test_layering_policy.py``) because it pins a
property of the visualization layer as a whole rather than one module's
behaviour.
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

#: Held fixed across the two sizes so only the prediction count varies.
N_MODELS = 4
N_DRUGS = 5
N_CELL_LINES = 4

#: An 8x growth in predictions. Anything retaining per-sample state grows with
#: this; the plots under test must not.
SMALL_ROWS = 100
LARGE_ROWS = 800

#: The per-sample implementations cost ~240 B per point, so an 8x row increase
#: moved ~1.3 MB in this fixture. Allowing 2x plus a fixed 64 kB is far below
#: that and far above the noise of a float32 matrix or a PNG re-render.
GROWTH_TOLERANCE = 2.0
FIXED_ALLOWANCE_BYTES = 64_000


def _experiment(n_rows: int) -> ExperimentResult:
    """Build an experiment whose only varying dimension is the row count."""
    return ExperimentResult(
        [
            make_run_result(
                model_name=name,
                fold_index=fold,
                n_pairs=n_rows,
                n_cell_lines=N_CELL_LINES,
                n_drugs=N_DRUGS,
                seed=index * 100 + fold,
            )
            for index, name in enumerate(f"Model_{i}" for i in range(N_MODELS))
            for fold in range(2)
        ]
    )


def _model_result(n_rows: int) -> ModelResult:
    return ModelResult(
        model_name="Model_0",
        dataset_name=DEFAULT_DATASET_NAME,
        runs=[
            make_run_result(
                model_name="Model_0",
                fold_index=fold,
                n_pairs=n_rows,
                n_cell_lines=N_CELL_LINES,
                n_drugs=N_DRUGS,
            )
            for fold in range(2)
        ],
    )


def _deep_size(obj: object, seen: set[int] | None = None) -> int:
    """Recursively size a Python object graph, counting each object once.

    ``sys.getsizeof`` alone reports a container's own overhead and misses the
    per-element dicts that caused the original blow-up, so the walk is the point.

    :param obj: Object to size.
    :param seen: Ids already counted, used for the recursive calls.
    :returns: Total bytes reachable from ``obj``.
    """
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
    """Size everything a computed visualization holds, excluding the input."""
    excluded = {"_result"}
    return sum(_deep_size(value) for name, value in vars(plot).items() if name not in excluded)


def _payload_bytes(plot) -> int:
    """Size the report payload a computed visualization emits."""
    return sum(len(section.content or "") for section in plot.to_multiqc())


@pytest.fixture(scope="module")
def comparison_scatter_sizes() -> dict[int, tuple[int, int]]:
    sizes = {}
    for n_rows in (SMALL_ROWS, LARGE_ROWS):
        plot = ComparisonScatterVisualization()
        plot.compute(_experiment(n_rows))
        sizes[n_rows] = (_retained_bytes(plot), _payload_bytes(plot))
    return sizes


@pytest.fixture(scope="module")
def regression_scatter_payloads() -> dict[int, int]:
    payloads = {}
    for n_rows in (SMALL_ROWS, LARGE_ROWS):
        plot = RegressionScatterVisualization()
        plot.compute(_model_result(n_rows))
        payloads[n_rows] = _payload_bytes(plot)
    return payloads


def _assert_bounded(small: int, large: int, label: str) -> None:
    limit = small * GROWTH_TOLERANCE + FIXED_ALLOWANCE_BYTES
    ratio = LARGE_ROWS / SMALL_ROWS
    assert large <= limit, (
        f"{label} grew from {small:,} to {large:,} bytes for a {ratio:.0f}x increase in predictions "
        f"({SMALL_ROWS} -> {LARGE_ROWS} rows per fold) with models and groups held fixed. "
        "Visualization payloads must scale with models x groups, not with the number of samples - "
        "see the docstring of this module."
    )


class TestComparisonScatter:
    def test_retained_state_does_not_grow_with_predictions(self, comparison_scatter_sizes):
        small, _ = comparison_scatter_sizes[SMALL_ROWS]
        large, _ = comparison_scatter_sizes[LARGE_ROWS]

        _assert_bounded(small, large, "ComparisonScatterVisualization retained state")

    def test_report_payload_does_not_grow_with_predictions(self, comparison_scatter_sizes):
        _, small = comparison_scatter_sizes[SMALL_ROWS]
        _, large = comparison_scatter_sizes[LARGE_ROWS]

        _assert_bounded(small, large, "ComparisonScatterVisualization report payload")

    def test_retained_state_is_a_models_by_groups_matrix(self):
        plot = ComparisonScatterVisualization()

        plot.compute(_experiment(SMALL_ROWS))

        assert plot._matrices["drug"].values.shape == (N_MODELS, N_DRUGS)
        assert plot._matrices["cell_line"].values.shape == (N_MODELS, N_CELL_LINES)

    def test_nothing_is_retained_per_model_pair(self):
        """The original bug stored ``C(n_models, 2)`` point clouds."""
        plot = ComparisonScatterVisualization()

        plot.compute(_experiment(SMALL_ROWS))

        assert not hasattr(plot, "_pair_data")

    def test_the_matrix_is_float32(self):
        plot = ComparisonScatterVisualization()

        plot.compute(_experiment(SMALL_ROWS))

        assert all(matrix.values.dtype == np.float32 for matrix in plot._matrices.values())


class TestRegressionScatter:
    def test_report_payload_does_not_grow_with_predictions(self, regression_scatter_payloads):
        _assert_bounded(
            regression_scatter_payloads[SMALL_ROWS],
            regression_scatter_payloads[LARGE_ROWS],
            "RegressionScatterVisualization report payload",
        )

    def test_retained_arrays_are_flat_floats_not_objects(self):
        plot = RegressionScatterVisualization()

        plot.compute(_model_result(SMALL_ROWS))

        for array in (plot._ground_truth, plot._predictions):
            assert array.dtype == np.float64
            assert array.ndim == 1

    def test_retained_arrays_cost_two_floats_per_prediction(self):
        """A per-row dict cost ~240 B; two float64 arrays cost 16 B."""
        plot = RegressionScatterVisualization()

        plot.compute(_model_result(SMALL_ROWS))

        assert plot._ground_truth.nbytes + plot._predictions.nbytes == 16 * 2 * SMALL_ROWS


class TestGuardIsMeaningful:
    def test_the_bound_would_reject_a_per_sample_payload(self):
        """Guard the guard: 240 B per point at both sizes must fail the check."""
        per_point = 240
        small = per_point * SMALL_ROWS * 2
        large = per_point * LARGE_ROWS * 2

        with pytest.raises(AssertionError, match="must scale with models x groups"):
            _assert_bounded(small, large, "hypothetical per-sample payload")

    def test_the_bound_accepts_a_constant_payload(self):
        _assert_bounded(10_000, 10_000, "hypothetical constant payload")

    def test_deep_size_counts_per_element_dicts(self):
        """The walk must see through a list, or it cannot detect the old shape."""
        points = [{"x": float(i), "y": float(i)} for i in range(50)]

        assert _deep_size(points) > 50 * sys.getsizeof({"x": 1.0, "y": 1.0}) * 0.5

    def test_deep_size_counts_numpy_buffers(self):
        assert _deep_size(np.zeros(1000, dtype=np.float64)) >= 8000
