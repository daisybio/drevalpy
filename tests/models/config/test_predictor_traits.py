"""Tests for drevalpy.models.config._predictor_traits."""

from __future__ import annotations

import pytest

from drevalpy.components.core.plugins.register_builtins import register_builtin_components
from drevalpy.models.config import PredictorConfig
from drevalpy.models.config._predictor_traits import (
    needs_identity_drug_routing,
    scope,
)
from drevalpy.types.enums.model_scope import ModelScope


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
    assert scope(name) is expected


def test_scope_for_predictor_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown Predictor"):
        scope("noSuchPredictor")


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
def test_routing_is_needed_for_every_predictor_spelling(slot: object) -> None:
    assert needs_identity_drug_routing(slot) is True


def test_routing_is_not_needed_for_a_multi_drug_predictor() -> None:
    assert needs_identity_drug_routing("elasticNet") is False


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
def test_routing_is_not_needed_for_unusable_slots(slot: object) -> None:
    assert needs_identity_drug_routing(slot) is False
