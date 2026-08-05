"""Tests that every built-in predictor belongs to exactly one input interface."""

from __future__ import annotations

import pytest

from drevalpy.components.predictors.block import BlockPredictor
from drevalpy.components.predictors.feature_free import FeatureFreePredictor
from drevalpy.components.predictors.matrix import MatrixPredictor
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
LEAF_BASES = (FeatureFreePredictor, MatrixPredictor, BlockPredictor)


@pytest.fixture(autouse=True)
def _register() -> None:
    register_builtin_components()


def test_interface_bases_declare_input_interface() -> None:
    assert FeatureFreePredictor.input_interface == "feature_free"
    assert MatrixPredictor.input_interface == "matrix"
    assert BlockPredictor.input_interface == "block"


def test_builtin_predictor_interfaces_partition() -> None:
    observed: dict[str, set[str]] = {
        "feature_free": set(),
        "matrix": set(),
        "block": set(),
    }
    for name in list_predictors():
        cls = get_predictor(name)
        matches = [base for base in LEAF_BASES if issubclass(cls, base)]
        assert len(matches) == 1, (name, matches)
        assert cls.input_interface == matches[0].input_interface
        observed[cls.input_interface].add(name)
    assert observed == EXPECTED
