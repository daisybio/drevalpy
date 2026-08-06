"""Tests for drevalpy.models.config._predictor_traits."""

from __future__ import annotations

import pytest

from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.models.config import PredictorConfig
from drevalpy.models.config._predictor_traits import (
    routing_drug_featurizer_for_slot,
    scope_for_predictor,
)
from drevalpy.types.model_scope import ModelScope


@pytest.fixture(autouse=True)
def _register_builtins() -> None:
    register_builtin_components()


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("elasticNet", ModelScope.MULTI_DRUG),
        ("naiveMean", ModelScope.MULTI_DRUG),
        ("singleDrugElasticNet", ModelScope.SINGLE_DRUG),
        ("singleDrugRandomForest", ModelScope.SINGLE_DRUG),
    ],
)
def test_scope_for_predictor_reads_the_class_declaration(name: str, expected: ModelScope) -> None:
    assert scope_for_predictor(name) is expected


def test_scope_for_predictor_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown Predictor"):
        scope_for_predictor("noSuchPredictor")


@pytest.mark.parametrize(
    "slot",
    [
        "singleDrugElasticNet",
        {"singleDrugElasticNet": None},
        {"name": "singleDrugElasticNet"},
        {"singleDrugElasticNet": {"alpha": 0.5}},
        PredictorConfig(name="singleDrugElasticNet"),
    ],
)
def test_routing_drug_featurizer_reads_every_predictor_spelling(slot: object) -> None:
    assert routing_drug_featurizer_for_slot(slot) == "identity"


def test_routing_drug_featurizer_is_none_for_a_multi_drug_predictor() -> None:
    assert routing_drug_featurizer_for_slot("elasticNet") is None


@pytest.mark.parametrize(
    "slot",
    [
        None,
        42,
        [],
        "noSuchPredictor",
        {"name": "noSuchPredictor"},
        {"first": None, "second": None},
        {"name": "singleDrugElasticNet", "unexpected": 1},
    ],
)
def test_routing_drug_featurizer_returns_none_for_unusable_slots(slot: object) -> None:
    assert routing_drug_featurizer_for_slot(slot) is None
