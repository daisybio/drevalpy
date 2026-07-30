"""Tests that every built-in predictor belongs to exactly one input interface."""

from __future__ import annotations

import pytest

from drevalpy.components.predictors.feature_free import FeatureFreePredictor
from drevalpy.components.predictors.matrix import MatrixPredictor
from drevalpy.components.predictors.structured import BlockPredictor
from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.components.registry import get_predictor, list_predictors

EXPECTED = {
    "feature_free": {"naiveMean"},
    "matrix": {
        "elasticNet",
        "singleDrugElasticNet",
        "lasso",
        "ridge",
        "randomForest",
        "singleDrugRandomForest",
        "svr",
        "gradientBoosting",
        "adaboost",
        "knn",
        "xgboost",
        "lightgbm",
        "neuralNetwork",
    },
    "block": {
        "naiveDrugMean",
        "naiveCellLineMean",
        "naiveTissueMean",
        "naiveTissueDrugMean",
        "naiveMeanEffects",
        "precily",
        "srmf",
        "drugGNN",
        "dipk",
        "pharmaFormer",
        "sparsego",
        "molir",
        "superfeltr",
    },
}


@pytest.fixture(autouse=True)
def _register() -> None:
    register_builtin_components()


def _interface_name(cls: type) -> str:
    matches = []
    if issubclass(cls, FeatureFreePredictor):
        matches.append("feature_free")
    if issubclass(cls, MatrixPredictor):
        matches.append("matrix")
    if issubclass(cls, BlockPredictor):
        matches.append("block")
    assert len(matches) == 1, (cls, matches)
    return matches[0]


def test_builtin_predictor_interfaces_partition() -> None:
    observed: dict[str, set[str]] = {
        "feature_free": set(),
        "matrix": set(),
        "block": set(),
    }
    for name in list_predictors():
        observed[_interface_name(get_predictor(name))].add(name)
    assert observed == EXPECTED
