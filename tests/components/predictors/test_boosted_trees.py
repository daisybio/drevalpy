"""Tests for the shared boosted-trees predictor base, on throwaway subclasses."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from drevalpy.components.predictors._boosted_trees import SHARED_DEFAULTS, SHARED_SPACE, BoostedTreesPredictor
from drevalpy.components.predictors.sklearn_tabular import SklearnTabularPredictor


class _Bare(BoostedTreesPredictor):
    def _make_estimator(self) -> Any:
        return object()


class _Tweaked(BoostedTreesPredictor):
    boosting_default_overrides: ClassVar[dict[str, Any]] = {"subsample": 0.8}
    boosting_extra_defaults: ClassVar[dict[str, Any]] = {"num_leaves": 63}
    boosting_space_overrides: ClassVar[dict[str, dict[str, Any]]] = {"max_depth": {"high": 8}}
    tuned_hyperparameters: ClassVar[tuple[str, ...]] = ("max_depth", "num_leaves")

    def _make_estimator(self) -> Any:
        return object()


def test_shared_tables_are_consistent() -> None:
    assert set(SHARED_DEFAULTS) - {"random_state"} <= set(SHARED_SPACE)
    assert "log" not in SHARED_SPACE["learning_rate"]
    assert all(spec["low"] <= spec["default"] <= spec["high"] for spec in SHARED_SPACE.values())
    assert issubclass(BoostedTreesPredictor, SklearnTabularPredictor)


def test_estimator_params_resolve_defaults_overrides_and_coercion() -> None:
    assert _Bare()._estimator_params() == SHARED_DEFAULTS
    tweaked = _Tweaked()._estimator_params()
    assert tweaked["subsample"] == pytest.approx(0.8)
    assert tweaked["num_leaves"] == 63
    assert SHARED_DEFAULTS["subsample"] == pytest.approx(1.0)
    assert "num_leaves" not in _Bare(hyperparameters={"num_leaves": 5})._estimator_params()
    params = _Bare(hyperparameters={"n_estimators": "7", "learning_rate": "0.05"})._estimator_params()
    assert params["n_estimators"] == 7
    assert isinstance(params["n_estimators"], int)
    assert params["learning_rate"] == pytest.approx(0.05)


def test_hyperparameter_space_exposes_declared_names_merged_with_overrides() -> None:
    assert _Bare.get_hyperparameter_space() == {}
    space = _Tweaked.get_hyperparameter_space()
    assert set(space) == {"max_depth", "num_leaves"}
    assert space["max_depth"] == {**SHARED_SPACE["max_depth"], "high": 8}
    assert space["num_leaves"] == SHARED_SPACE["num_leaves"]
    assert SHARED_SPACE["max_depth"]["high"] == 12
