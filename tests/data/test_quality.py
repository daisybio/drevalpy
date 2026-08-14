"""Tests for :mod:`drevalpy.data.quality`.

The threshold checks are parametrized over :data:`~drevalpy.data.quality._RULES`
rather than written out one per option. That is deliberate: it means a new rule
cannot be added to the table without inheriting a boundary test and an
off-by-default test, so the fifteen options cannot drift apart in coverage.
"""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.curation._anndata import _REGULATION_ENCODING
from drevalpy.data.quality import _RESPONSE_MATRIX, _RULES, curve_quality_mask

_SHAPE = (2, 3)

#: A metric value that passes each rule comfortably, per option.
_PASSING: dict[str, float] = {
    "min_relevance_score": 9.0,
    "min_abs_fold_change": -2.0,
    "max_p_value": 1e-9,
    "min_log_p_value": 9.0,
    "min_f_value": 400.0,
    "min_f_value_sam": 80.0,
    "min_r2": 0.99,
    "max_rmse": 0.02,
    "min_signal_quality": 1.0,
    "min_abs_slope": 3.0,
    "max_abs_slope": 3.0,
    "min_front": 1.0,
    "max_back": 0.05,
    "min_pec50": 6.0,
    "max_pec50": 6.0,
}

#: A threshold that the matching :data:`_PASSING` value satisfies, and a second
#: one it fails, for every option. The passing threshold is set to exactly the
#: value under test wherever the comparison allows, which pins inclusivity.
_THRESHOLDS: dict[str, tuple[float, float]] = {
    #                     (satisfied, violated)
    "min_relevance_score": (9.0, 9.5),
    "min_abs_fold_change": (2.0, 2.5),
    "max_p_value": (1e-9, 1e-12),
    "min_log_p_value": (9.0, 9.5),
    "min_f_value": (400.0, 500.0),
    "min_f_value_sam": (80.0, 100.0),
    "min_r2": (0.99, 0.995),
    "max_rmse": (0.02, 0.01),
    "min_signal_quality": (1.0, 1.5),
    "min_abs_slope": (3.0, 3.5),
    "max_abs_slope": (3.0, 2.5),
    "min_front": (1.0, 1.5),
    "max_back": (0.05, 0.01),
    "min_pec50": (6.0, 6.5),
    "max_pec50": (6.0, 5.5),
}


class FakeDataset:
    """Minimal ``MuDataLike`` whose layers are set per test.

    Only the members the quality filter touches are implemented; the splitters'
    axis accessors are irrelevant here.
    """

    def __init__(self, layers: dict[str, np.ndarray], response: np.ndarray | None = None) -> None:
        """Store the layers and the response matrix backing ``pEC50``."""
        self._layers = layers
        self._response = response if response is not None else np.full(_SHAPE, 6.0)

    @property
    def cell_line_ids(self) -> np.ndarray:
        """Row identifiers."""
        return np.array([f"CL_{i}" for i in range(_SHAPE[0])])

    @property
    def drug_ids(self) -> np.ndarray:
        """Column identifiers."""
        return np.array([f"D_{i}" for i in range(_SHAPE[1])])

    @property
    def response_matrix(self) -> np.ndarray:
        """Response matrix, which is where ``pEC50`` lives."""
        return self._response

    def get_tissue(self, ids: np.ndarray) -> np.ndarray:
        """One tissue for every cell line."""
        return np.array(["Lung"] * len(ids))

    def response_layer_names(self) -> list[str]:
        """Names of the layers this fake carries."""
        return list(self._layers)

    def get_response_layer(self, name: str) -> np.ndarray:
        """Return a layer, raising like ``Dataset`` does when it is absent."""
        if name not in self._layers:
            raise KeyError(f"Response layer '{name}' not found. Available: {list(self._layers)}")
        return self._layers[name]


def _dataset_for(option: str, value: float) -> FakeDataset:
    """Build a dataset where the layer *option* reads carries *value* throughout."""
    layer, _comparison, _transform = _RULES[option]
    matrix = np.full(_SHAPE, value)
    if layer == _RESPONSE_MATRIX:
        return FakeDataset({}, response=matrix)
    return FakeDataset({layer: matrix})


def _only(option: str, threshold: float) -> dict[str, float | None]:
    """Request exactly one check, disabling the two that are on by default."""
    return {"min_relevance_score": None, "min_abs_fold_change": None, option: threshold}


def _passing_dataset() -> FakeDataset:
    """A dataset that passes every rule in the table."""
    layers: dict[str, np.ndarray] = {}
    for option, value in _PASSING.items():
        layer, _comparison, _transform = _RULES[option]
        if layer != _RESPONSE_MATRIX:
            layers[layer] = np.full(_SHAPE, value)
    return FakeDataset(layers)


class TestEveryOptionIsWiredUp:
    """One boundary and one off-by-default case per entry in ``_RULES``."""

    def test_the_table_and_the_expectations_cover_the_same_options(self) -> None:
        """Guards the guard: a new rule must arrive with its test values."""
        assert sorted(_RULES) == sorted(_PASSING)
        assert sorted(_RULES) == sorted(_THRESHOLDS)

    @pytest.mark.parametrize("option", sorted(_RULES))
    def test_a_satisfied_threshold_keeps_every_pair(self, option: str) -> None:
        satisfied, _violated = _THRESHOLDS[option]
        dataset = _dataset_for(option, _PASSING[option])

        assert curve_quality_mask(dataset, **_only(option, satisfied)).all()

    @pytest.mark.parametrize("option", sorted(_RULES))
    def test_a_violated_threshold_drops_every_pair(self, option: str) -> None:
        _satisfied, violated = _THRESHOLDS[option]
        dataset = _dataset_for(option, _PASSING[option])

        assert not curve_quality_mask(dataset, **_only(option, violated)).any()

    @pytest.mark.parametrize("option", sorted(_RULES))
    def test_the_threshold_is_inclusive(self, option: str) -> None:
        """``>=``/``<=``, never ``>``/``<``: a value exactly at the cut passes."""
        satisfied, _violated = _THRESHOLDS[option]
        _layer, _comparison, transform = _RULES[option]
        # The satisfied threshold equals the metric under test, modulo the
        # absolute-value transform, so this is the boundary itself.
        value = _PASSING[option]
        expected = abs(value) if transform is not None else value
        assert satisfied == pytest.approx(expected)

        dataset = _dataset_for(option, value)
        assert curve_quality_mask(dataset, **_only(option, satisfied)).all()

    @pytest.mark.parametrize("option", sorted(_RULES))
    def test_an_option_left_at_none_is_not_checked(self, option: str) -> None:
        """A metric bad enough to fail its own rule is ignored while it is off."""
        _satisfied, violated = _THRESHOLDS[option]
        dataset = _dataset_for(option, _PASSING[option])

        # Sanity: this threshold really would reject every pair if requested.
        assert not curve_quality_mask(dataset, **_only(option, violated)).any()

        assert curve_quality_mask(
            dataset,
            min_relevance_score=None,
            min_abs_fold_change=None,
        ).all()


class TestDefaults:
    def test_the_default_rule_uses_relevance_score_and_fold_change(self) -> None:
        """Nothing else may be consulted by default, or unrelated data breaks."""
        dataset = FakeDataset(
            {
                "relevance_score": np.full(_SHAPE, 9.0),
                "fold_change": np.full(_SHAPE, -2.0),
            }
        )

        assert curve_quality_mask(dataset).all()

    def test_the_default_relevance_threshold_is_minus_log10_alpha(self) -> None:
        """``alpha = 0.05`` in ``drevalpy/curation/_fit.py``."""
        cut = -np.log10(0.05)
        dataset = FakeDataset(
            {
                "relevance_score": np.array([[cut, cut * 0.999, 0.0]] * 2),
                "fold_change": np.full(_SHAPE, -2.0),
            }
        )

        mask = curve_quality_mask(dataset)

        assert mask[:, 0].all()
        assert not mask[:, 1].any()
        assert not mask[:, 2].any()

    def test_the_default_fold_change_threshold_is_fc_lim(self) -> None:
        """``fc_lim = 0.45``, applied to the magnitude of an already-log2 layer."""
        dataset = FakeDataset(
            {
                "relevance_score": np.full(_SHAPE, 9.0),
                "fold_change": np.array([[0.45, -0.45, 0.44]] * 2),
            }
        )

        mask = curve_quality_mask(dataset)

        assert mask[:, 0].all()
        assert mask[:, 1].all()
        assert not mask[:, 2].any()


class TestNanFailsClosed:
    @pytest.mark.parametrize("option", sorted(_RULES))
    def test_a_nan_metric_never_passes(self, option: str) -> None:
        """A curve CurveCurator could not score is not a curve worth keeping."""
        satisfied, _violated = _THRESHOLDS[option]
        dataset = _dataset_for(option, np.nan)

        assert not curve_quality_mask(dataset, **_only(option, satisfied)).any()

    def test_a_nan_in_one_metric_does_not_condemn_the_others(self) -> None:
        dataset = FakeDataset(
            {
                "relevance_score": np.array([[9.0, np.nan, 9.0]] * 2),
                "fold_change": np.full(_SHAPE, -2.0),
            }
        )

        mask = curve_quality_mask(dataset)

        assert mask[:, 0].all()
        assert not mask[:, 1].any()
        assert mask[:, 2].all()


class TestCombiningOptions:
    def test_two_options_are_anded(self) -> None:
        dataset = FakeDataset(
            {
                "relevance_score": np.full(_SHAPE, 9.0),
                "fold_change": np.full(_SHAPE, -2.0),
                "R2": np.array([[0.99, 0.5, 0.99]] * 2),
            }
        )

        mask = curve_quality_mask(dataset, min_r2=0.9)

        # Column 1 passes the default rule but fails the added R2 floor.
        assert mask[:, 0].all()
        assert not mask[:, 1].any()
        assert mask[:, 2].all()

    def test_every_option_at_once_still_keeps_a_passing_dataset(self) -> None:
        dataset = _passing_dataset()
        options = {option: _THRESHOLDS[option][0] for option in _RULES}

        assert curve_quality_mask(dataset, **options).all()

    def test_disabling_everything_keeps_every_pair(self) -> None:
        """Including pairs whose metrics are missing entirely."""
        dataset = FakeDataset({})

        mask = curve_quality_mask(dataset, min_relevance_score=None, min_abs_fold_change=None)

        assert mask.shape == _SHAPE
        assert mask.all()


class TestMissingLayers:
    def test_a_requested_layer_that_is_absent_raises(self) -> None:
        """No capability check and no silent fallback: the format guarantees it."""
        dataset = FakeDataset({"relevance_score": np.full(_SHAPE, 9.0)})

        with pytest.raises(KeyError, match="fold_change"):
            curve_quality_mask(dataset)

    def test_an_unused_layer_may_be_absent(self) -> None:
        dataset = FakeDataset(
            {
                "relevance_score": np.full(_SHAPE, 9.0),
                "fold_change": np.full(_SHAPE, -2.0),
            }
        )

        assert curve_quality_mask(dataset).all()


class TestRegulation:
    def _dataset(self) -> FakeDataset:
        return FakeDataset(
            {
                "relevance_score": np.full(_SHAPE, 9.0),
                "fold_change": np.full(_SHAPE, -2.0),
                "regulation": np.array(
                    [
                        [_REGULATION_ENCODING["up"], _REGULATION_ENCODING["down"], _REGULATION_ENCODING["not"]],
                        [np.nan, _REGULATION_ENCODING["down"], _REGULATION_ENCODING["not"]],
                    ],
                    dtype=float,
                ),
            }
        )

    @pytest.mark.parametrize(
        ("labels", "expected"),
        [
            (["up"], [[True, False, False], [False, False, False]]),
            (["down"], [[False, True, False], [False, True, False]]),
            (["not"], [[False, False, True], [False, False, True]]),
            (["up", "down"], [[True, True, False], [False, True, False]]),
        ],
    )
    def test_labels_select_the_matching_pairs(self, labels: list[str], expected: list[list[bool]]) -> None:
        mask = curve_quality_mask(
            self._dataset(),
            min_relevance_score=None,
            min_abs_fold_change=None,
            regulation=labels,
        )

        assert mask.tolist() == expected

    def test_an_undetermined_curve_is_in_no_category(self) -> None:
        """NaN regulation means CurveCurator reached no verdict, so it is dropped."""
        mask = curve_quality_mask(
            self._dataset(),
            min_relevance_score=None,
            min_abs_fold_change=None,
            regulation=["up", "down", "not"],
        )

        assert not mask[1, 0]

    def test_an_unknown_label_raises(self) -> None:
        with pytest.raises(ValueError, match="sideways"):
            curve_quality_mask(self._dataset(), regulation=["sideways"])

    def test_the_error_lists_the_valid_labels(self) -> None:
        with pytest.raises(ValueError, match="down.*not.*up"):
            curve_quality_mask(self._dataset(), regulation=["bogus"])

    def test_the_default_rule_reproduces_curvecurators_verdict(self) -> None:
        """The whole reason the defaults are what they are.

        With ``quality_min = -inf`` and ``pEC50_filter = [-inf, inf]`` - the
        resolved config in ``drevalpy/curation/_fit.py`` - CurveCurator's
        ``regulation`` label reduces exactly to relevance-score-plus-fold-change,
        so recomputing it must agree pair for pair.
        """
        rng = np.random.default_rng(20260814)
        shape = (16, 8)
        relevance = rng.uniform(0.0, 5.0, size=shape)
        fold_change = rng.uniform(-3.0, 1.0, size=shape)

        significant = relevance >= -np.log10(0.05)
        large_effect = np.abs(fold_change) >= 0.45
        regulated = significant & large_effect
        regulation = np.where(regulated, np.sign(fold_change), 0.0)

        dataset = FakeDataset(
            {
                "relevance_score": relevance,
                "fold_change": fold_change,
                "regulation": regulation,
            },
            response=np.full(shape, 6.0),
        )

        mask = curve_quality_mask(dataset)

        assert mask.tolist() == regulated.tolist()
        assert mask.tolist() == (regulation != 0).tolist()


class TestMaskShape:
    def test_the_mask_matches_the_response_matrix(self) -> None:
        dataset = _passing_dataset()

        mask = curve_quality_mask(dataset)

        assert mask.shape == dataset.response_matrix.shape
        assert mask.dtype == np.bool_

    def test_the_inverse_blanks_a_response_matrix_without_reshaping_it(self) -> None:
        """The idiom every splitter uses."""
        dataset = FakeDataset(
            {
                "relevance_score": np.array([[9.0, 0.0, 9.0]] * 2),
                "fold_change": np.full(_SHAPE, -2.0),
            }
        )

        response = dataset.response_matrix.copy()
        response[~curve_quality_mask(dataset)] = np.nan

        assert response.shape == _SHAPE
        assert np.isnan(response[:, 1]).all()
        assert not np.isnan(response[:, [0, 2]]).any()

    def test_the_dataset_is_not_mutated(self) -> None:
        dataset = _passing_dataset()
        before = {name: dataset.get_response_layer(name).copy() for name in dataset.response_layer_names()}

        curve_quality_mask(dataset)

        for name, values in before.items():
            assert np.array_equal(dataset.get_response_layer(name), values)
