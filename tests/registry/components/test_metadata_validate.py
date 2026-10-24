"""Tests for registry class-state validation and role checks."""

from __future__ import annotations

import pytest

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.registry.components._metadata_validate import validate_registered_class
from drevalpy.registry.components._registration_metadata import (
    apply_registration_metadata,
    normalize_registration_metadata,
)
from drevalpy.registry.featurizer import FeaturizerRegistry
from drevalpy.registry.predictor import PredictorRegistry
from drevalpy.types.enums.literature_reference import LiteratureReference


def _component(**attrs) -> type:
    return type("Native", (), {"description": "native", "tags": frozenset(), "reference": None} | attrs)


def test_literature_reference_is_accepted() -> None:
    contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
    lit = type("Lit", (), {"sample_contract": contract, "drug_contract": contract})
    reference = LiteratureReference(
        repo_url="https://github.com/example/repo", citation_doi="10.1234/example", deviations="none"
    )

    apply_registration_metadata(lit, normalize_registration_metadata("lit model", reference=reference))

    validate_registered_class("predictor", "lit", lit, required_fields=PredictorRegistry._required_fields)


@pytest.mark.parametrize(
    ("role", "required_fields", "attrs", "message"),
    [
        ("drug_featurizer", FeaturizerRegistry._required_fields, {}, r"missing=\['contract'\]"),
        (
            "predictor",
            PredictorRegistry._required_fields,
            {},
            r"missing=\['sample_contract', 'drug_contract'\]",
        ),
        ("predictor", ("description",), {"description": None}, r"missing=\['description'\]"),
        (
            "drug_featurizer",
            FeaturizerRegistry._required_fields,
            {"contract": "numeric_matrix"},
            r"invalid=\['contract'\]",
        ),
    ],
    ids=["featurizer-contract", "predictor-contracts", "description", "wrong-type-contract"],
)
def test_role_validation_reports_missing_and_invalid_fields(role, required_fields, attrs, message) -> None:
    cls = _component(**attrs)
    if attrs.get("description", "x") is None:
        del cls.description  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match=message):
        validate_registered_class(role, "native", cls, required_fields=required_fields)


def test_register_rejects_a_blank_description_and_a_bad_contract_before_the_class_body() -> None:
    registry = FeaturizerRegistry("test", "Test", "tests")

    with pytest.raises(ValueError, match="description must be a non-empty string"):
        registry.register("blank", description="  ", contract=FeatureFormat.NUMERIC_MATRIX)
    with pytest.raises(Exception, match="FeatureContract|FeatureFormat|did not match any element in the union"):
        registry.register("badContract", description="demo", contract="numeric_matrix")  # type: ignore[arg-type]
