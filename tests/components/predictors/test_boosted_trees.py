"""Tests for :mod:`drevalpy.components.predictors._boosted_trees`.

Mirrors the private module with the underscore stripped. The two shipped
subclasses are asserted in their own modules; what is pinned here is the sharing
mechanism itself - default resolution, coercion and space assembly - on throwaway
subclasses, so neither library has to be installed.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from drevalpy.components.predictors._boosted_trees import SHARED_DEFAULTS, SHARED_SPACE, BoostedTreesPredictor
from drevalpy.components.predictors.sklearn_tabular import SklearnTabularPredictor


class _Bare(BoostedTreesPredictor):
    """Neither overrides nor tunes anything."""

    def _make_estimator(self) -> Any:
        return object()


class _Tweaked(BoostedTreesPredictor):
    """Overrides one default, adds one library-only knob, narrows one bound."""

    boosting_default_overrides: ClassVar[dict[str, Any]] = {"subsample": 0.8}
    boosting_extra_defaults: ClassVar[dict[str, Any]] = {"num_leaves": 63}
    boosting_space_overrides: ClassVar[dict[str, dict[str, Any]]] = {"max_depth": {"high": 8}}
    tuned_hyperparameters: ClassVar[tuple[str, ...]] = ("max_depth", "num_leaves")

    def _make_estimator(self) -> Any:
        return object()


class TestSharedTables:
    def test_every_default_is_also_a_declarable_spec(self):
        assert set(SHARED_DEFAULTS) - {"random_state"} <= set(SHARED_SPACE)

    def test_learning_rate_carries_no_log_flag(self):
        """The two libraries disagree, so each declares it rather than the table."""
        assert "log" not in SHARED_SPACE["learning_rate"]

    def test_every_spec_brackets_its_default(self):
        assert all(spec["low"] <= spec["default"] <= spec["high"] for spec in SHARED_SPACE.values())


class TestEstimatorParams:
    def test_defaults_to_the_shared_table(self):
        assert _Bare()._estimator_params() == SHARED_DEFAULTS

    def test_applies_the_library_override(self):
        assert _Tweaked()._estimator_params()["subsample"] == pytest.approx(0.8)

    def test_leaves_the_shared_table_untouched(self):
        _Tweaked()._estimator_params()

        assert SHARED_DEFAULTS["subsample"] == pytest.approx(1.0)

    def test_includes_the_library_only_knobs(self):
        assert _Tweaked()._estimator_params()["num_leaves"] == 63

    def test_forwards_a_hyperparameter_override(self):
        assert _Bare(hyperparameters={"n_estimators": 7})._estimator_params()["n_estimators"] == 7

    def test_coerces_to_the_type_of_the_default(self):
        params = _Bare(hyperparameters={"n_estimators": "7", "learning_rate": "0.05"})._estimator_params()

        assert params["n_estimators"] == 7
        assert params["learning_rate"] == pytest.approx(0.05)
        assert isinstance(params["n_estimators"], int)

    def test_ignores_hyperparameters_the_library_does_not_take(self):
        assert "num_leaves" not in _Bare(hyperparameters={"num_leaves": 5})._estimator_params()


class TestHyperparameterSpace:
    def test_is_empty_when_nothing_is_declared_as_tunable(self):
        assert _Bare.get_hyperparameter_space() == {}

    def test_exposes_exactly_the_declared_names(self):
        assert set(_Tweaked.get_hyperparameter_space()) == {"max_depth", "num_leaves"}

    def test_merges_the_override_onto_the_shared_spec(self):
        spec = _Tweaked.get_hyperparameter_space()["max_depth"]

        assert spec == {**SHARED_SPACE["max_depth"], "high": 8}

    def test_does_not_mutate_the_shared_spec(self):
        _Tweaked.get_hyperparameter_space()

        assert SHARED_SPACE["max_depth"]["high"] == 12

    def test_reuses_the_shared_spec_verbatim_without_an_override(self):
        assert _Tweaked.get_hyperparameter_space()["num_leaves"] == SHARED_SPACE["num_leaves"]


def test_keeps_the_sklearn_tabular_lifecycle() -> None:
    assert issubclass(BoostedTreesPredictor, SklearnTabularPredictor)
