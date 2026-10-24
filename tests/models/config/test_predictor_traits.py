"""Tests for predictor traits."""

from __future__ import annotations

import pytest

from drevalpy.models.config import PredictorConfig
from drevalpy.models.config._predictor_traits import needs_identity_drug_routing, scope
from drevalpy.types.enums.model_scope import ModelScope


@pytest.mark.parametrize(
    ("name", "expected"),
    [("elasticNet", ModelScope.MULTI_DRUG), ("singleDrugElasticNet", ModelScope.SINGLE_DRUG)],
)
def test_scope_reads_the_class_declaration(name: str, expected: ModelScope) -> None:
    assert scope(name) is expected


def test_scope_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown Predictor"):
        scope("noSuchPredictor")


@pytest.mark.parametrize(
    ("slot", "expected"),
    [
        ("singleDrugElasticNet", True),
        ({"singleDrugElasticNet": None}, True),
        ({"name": "singleDrugElasticNet"}, True),
        ({"singleDrugElasticNet": {"alpha": 0.5}}, True),
        (PredictorConfig(name="singleDrugElasticNet"), True),
        ("elasticNet", False),
        (None, False),
        (42, False),
        ({"name": "noSuchPredictor"}, False),
        ({"first": None, "second": None}, False),
        ({"name": "singleDrugElasticNet", "unexpected": 1}, False),
    ],
)
def test_identity_drug_routing(slot: object, expected: bool) -> None:
    assert needs_identity_drug_routing(slot) is expected
