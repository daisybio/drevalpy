"""Tests for :mod:`drevalpy.data.quality`; threshold checks are parametrized over ``_RULES``."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from drevalpy.curation._anndata import _REGULATION_ENCODING
from drevalpy.data import synthetic
from drevalpy.data.quality import _RESPONSE_MATRIX, _RULES, CurveQuality, curve_quality_mask

_SHAPE = (2, 3)

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

_THRESHOLDS: dict[str, tuple[float, float]] = {
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
    """Minimal ``MuDataLike`` whose layers are set per test."""

    def __init__(self, layers: dict[str, np.ndarray], response: np.ndarray | None = None) -> None:
        """Store the layers and the response matrix backing ``pEC50``."""
        self._layers = layers
        self._response = response if response is not None else np.full(_SHAPE, 6.0)

    @property
    def sample_ids(self) -> np.ndarray:
        return np.array([f"CL_{i}" for i in range(_SHAPE[0])])

    @property
    def drug_ids(self) -> np.ndarray:
        return np.array([f"D_{i}" for i in range(_SHAPE[1])])

    @property
    def response_matrix(self) -> np.ndarray:
        return self._response

    def get_tissue(self, ids: np.ndarray) -> np.ndarray:
        return np.array(["Lung"] * len(ids))

    def sample_column(self, column: str, ids: np.ndarray) -> np.ndarray:
        if column != "tissue":
            raise KeyError(f"Sample column '{column}' not found. Available: ['tissue']")
        return self.get_tissue(ids)

    def response_layer_names(self) -> list[str]:
        return list(self._layers)

    def get_response_layer(self, name: str) -> np.ndarray:
        if name not in self._layers:
            raise KeyError(f"Response layer '{name}' not found. Available: {list(self._layers)}")
        return self._layers[name]


def _dataset_for(option: str, value: float) -> FakeDataset:
    layer, _comparison, _transform = _RULES[option]
    matrix = np.full(_SHAPE, value)
    if layer == _RESPONSE_MATRIX:
        return FakeDataset({}, response=matrix)
    return FakeDataset({layer: matrix})


def _only(option: str, threshold: float) -> dict[str, float | None]:
    return {"min_relevance_score": None, "min_abs_fold_change": None, option: threshold}


def _passing_dataset() -> FakeDataset:
    layers: dict[str, np.ndarray] = {}
    for option, value in _PASSING.items():
        layer, _comparison, _transform = _RULES[option]
        if layer != _RESPONSE_MATRIX:
            layers[layer] = np.full(_SHAPE, value)
    return FakeDataset(layers)


def _both(extra: dict[str, np.ndarray] | None = None, relevance: float | np.ndarray = 9.0) -> FakeDataset:
    layers = {"relevance_score": np.broadcast_to(relevance, _SHAPE), "fold_change": np.full(_SHAPE, -2.0)}
    return FakeDataset({**layers, **(extra or {})})


def test_the_table_and_the_expectations_cover_the_same_options() -> None:
    assert sorted(_RULES) == sorted(_PASSING) == sorted(_THRESHOLDS)


@pytest.mark.parametrize("option", sorted(_RULES))
def test_every_option_keeps_at_the_boundary_drops_when_violated_is_off_by_default_and_fails_closed_on_nan(
    option: str,
) -> None:
    satisfied, violated = _THRESHOLDS[option]
    dataset = _dataset_for(option, _PASSING[option])
    off = {"min_relevance_score": None, "min_abs_fold_change": None}

    assert curve_quality_mask(dataset, **_only(option, satisfied)).all()
    assert not curve_quality_mask(dataset, **_only(option, violated)).any()
    assert curve_quality_mask(dataset, **off).all()
    assert not curve_quality_mask(_dataset_for(option, np.nan), **_only(option, satisfied)).any()


def test_the_default_rule_consults_only_relevance_score_and_fold_change() -> None:
    assert curve_quality_mask(_both()).all()


def test_the_default_thresholds_are_minus_log10_alpha_and_fc_lim() -> None:
    cut = -np.log10(0.05)
    relevance = curve_quality_mask(_both(relevance=np.array([[cut, cut * 0.999, 0.0]] * 2)))
    fold = curve_quality_mask(
        FakeDataset({"relevance_score": np.full(_SHAPE, 9.0), "fold_change": np.array([[0.45, -0.45, 0.44]] * 2)})
    )

    assert relevance.tolist() == [[True, False, False]] * 2
    assert fold.tolist() == [[True, True, False]] * 2


def test_a_nan_in_one_metric_does_not_condemn_the_others() -> None:
    mask = curve_quality_mask(_both(relevance=np.array([[9.0, np.nan, 9.0]] * 2)))

    assert mask.tolist() == [[True, False, True]] * 2


def test_options_are_anded_and_a_passing_dataset_survives_all_of_them() -> None:
    mask = curve_quality_mask(_both({"R2": np.array([[0.99, 0.5, 0.99]] * 2)}), min_r2=0.9)

    assert mask.tolist() == [[True, False, True]] * 2
    assert curve_quality_mask(_passing_dataset(), **{option: _THRESHOLDS[option][0] for option in _RULES}).all()


def test_disabling_everything_keeps_every_pair_even_with_no_layers() -> None:
    mask = curve_quality_mask(FakeDataset({}), min_relevance_score=None, min_abs_fold_change=None)

    assert mask.shape == _SHAPE
    assert mask.all()


def test_a_requested_layer_that_is_absent_raises_and_an_unused_one_may_be_absent() -> None:
    with pytest.raises(KeyError, match="fold_change"):
        curve_quality_mask(FakeDataset({"relevance_score": np.full(_SHAPE, 9.0)}))
    with pytest.raises(TypeError):
        curve_quality_mask(FakeDataset({}), min_nonsense=1.0)


class TestRegulation:
    def _dataset(self) -> FakeDataset:
        up, down, not_ = (_REGULATION_ENCODING[label] for label in ("up", "down", "not"))
        return _both({"regulation": np.array([[up, down, not_], [np.nan, down, not_]], dtype=float)})

    @pytest.mark.parametrize(
        ("labels", "expected"),
        [
            (["up"], [[True, False, False], [False, False, False]]),
            (["down"], [[False, True, False], [False, True, False]]),
            (["not"], [[False, False, True], [False, False, True]]),
            (["up", "down"], [[True, True, False], [False, True, False]]),
        ],
    )
    def test_labels_select_the_matching_pairs_and_an_undetermined_curve_is_in_no_category(
        self, labels: list[str], expected: list[list[bool]]
    ) -> None:
        mask = curve_quality_mask(
            self._dataset(), min_relevance_score=None, min_abs_fold_change=None, regulation=labels
        )

        assert mask.tolist() == expected

    def test_an_unknown_label_raises_listing_the_valid_ones(self) -> None:
        with pytest.raises(ValueError, match="sideways"):
            curve_quality_mask(self._dataset(), regulation=["sideways"])
        with pytest.raises(ValueError, match="down.*not.*up"):
            curve_quality_mask(self._dataset(), regulation=["bogus"])

    def test_the_default_rule_reproduces_curvecurators_verdict(self) -> None:
        rng = np.random.default_rng(20260814)
        shape = (16, 8)
        relevance = rng.uniform(0.0, 5.0, size=shape)
        fold_change = rng.uniform(-3.0, 1.0, size=shape)
        regulated = (relevance >= -np.log10(0.05)) & (np.abs(fold_change) >= 0.45)
        regulation = np.where(regulated, np.sign(fold_change), 0.0)
        dataset = FakeDataset(
            {"relevance_score": relevance, "fold_change": fold_change, "regulation": regulation},
            response=np.full(shape, 6.0),
        )

        mask = curve_quality_mask(dataset)

        assert mask.tolist() == regulated.tolist()
        assert mask.tolist() == (regulation != 0).tolist()


def test_the_mask_matches_the_response_matrix_blanks_it_in_place_and_does_not_mutate_the_dataset() -> None:
    dataset = _passing_dataset()
    before = {name: dataset.get_response_layer(name).copy() for name in dataset.response_layer_names()}
    mask = curve_quality_mask(dataset)
    uneven = _both(relevance=np.array([[9.0, 0.0, 9.0]] * 2))
    response = uneven.response_matrix.copy()
    response[~curve_quality_mask(uneven)] = np.nan

    assert mask.shape == dataset.response_matrix.shape
    assert mask.dtype == np.bool_
    assert all(np.array_equal(dataset.get_response_layer(name), values) for name, values in before.items())
    assert np.isnan(response[:, 1]).all()
    assert not np.isnan(response[:, [0, 2]]).any()


class TestCurveQuality:
    def test_default_mask_equals_the_hand_written_rule_on_a_synthetic_dataset(self) -> None:
        dataset = synthetic(low_quality_fraction=0.3)
        relevance = dataset.get_response_layer("relevance_score").astype(np.float64)
        fold_change = dataset.get_response_layer("fold_change").astype(np.float64)
        expected = (relevance >= -np.log10(0.05)) & (np.abs(fold_change) >= 0.45)

        mask = CurveQuality.default().mask(dataset)

        assert not mask.all()
        assert mask.dtype == bool
        np.testing.assert_array_equal(mask, expected)
        np.testing.assert_array_equal(mask, curve_quality_mask(dataset))
        assert CurveQuality.default() == CurveQuality()

    def test_off_keeps_exactly_the_finite_responses_even_after_the_quality_layers_are_removed(self) -> None:
        explicit = FakeDataset({}, response=np.array([[6.0, np.nan, 5.0], [np.inf, 7.0, 6.5]]))
        dataset = synthetic(low_quality_fraction=0.3)
        for name in dataset.response_layer_names():
            if name != "LN_IC50":
                del dataset.response.layers[name]

        np.testing.assert_array_equal(CurveQuality.off().mask(explicit), [[True, False, True], [False, True, True]])
        with pytest.raises(KeyError):
            CurveQuality.default().mask(dataset)
        np.testing.assert_array_equal(CurveQuality.off().mask(dataset), np.isfinite(dataset.response_matrix))
        assert CurveQuality.off().to_metadata() == {}

    def test_custom_thresholds_are_applied_on_top_of_the_defaults(self) -> None:
        dataset = _both({"R2": np.array([[0.99, 0.5, 0.99]] * 2)})

        np.testing.assert_array_equal(CurveQuality(min_r2=0.9).mask(dataset), [[True, False, True]] * 2)

    def test_the_rule_is_frozen_hashable_and_stores_regulation_sorted_and_deduplicated(self) -> None:
        rule = CurveQuality(regulation=("up",))

        with pytest.raises(dataclasses.FrozenInstanceError):
            rule.min_r2 = 0.5  # type: ignore[misc]
        assert hash(rule) == hash(CurveQuality(regulation=("up",)))
        assert CurveQuality(regulation={"up", "down"}).regulation == ("down", "up")  # type: ignore[arg-type]
        assert CurveQuality(regulation=["up", "up"]).regulation == ("up",)  # type: ignore[arg-type]


class TestMetadata:
    def test_default_metadata_holds_only_the_two_default_thresholds(self) -> None:
        assert CurveQuality.default().to_metadata() == {
            "min_relevance_score": 1.3010299956639813,
            "min_abs_fold_change": 0.45,
        }

    @pytest.mark.parametrize(
        "rule",
        [
            CurveQuality.default(),
            CurveQuality.off(),
            CurveQuality(min_relevance_score=None, max_rmse=0.1, regulation=("up", "down")),
            CurveQuality(**{option: _THRESHOLDS[option][0] for option in _RULES}),
        ],
        ids=["default", "off", "partial", "all"],
    )
    def test_metadata_round_trips_through_json(self, rule: CurveQuality) -> None:
        assert CurveQuality.from_metadata(json.loads(json.dumps(rule.to_metadata()))) == rule

    def test_empty_metadata_is_off_not_default_and_unknown_names_raise(self) -> None:
        assert CurveQuality.from_metadata({}) == CurveQuality.off()
        with pytest.raises(ValueError, match="min_nonsense"):
            CurveQuality.from_metadata({"min_nonsense": 1.0})
