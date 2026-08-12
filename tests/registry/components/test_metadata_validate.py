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


def test_literature_reference_is_accepted() -> None:
    class Lit:
        cell_line_contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
        drug_contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)

    apply_registration_metadata(
        Lit,
        normalize_registration_metadata(
            "lit model",
            reference=LiteratureReference(
                repo_url="https://github.com/example/repo",
                citation_doi="10.1234/example",
                deviations="none",
            ),
        ),
    )
    validate_registered_class(
        "predictor",
        "lit",
        Lit,
        required_fields=PredictorRegistry._required_fields,
    )


def test_literature_reference_missing_fields_fails_on_normalize() -> None:
    with pytest.raises(ValueError, match="invalid fields"):
        normalize_registration_metadata(
            "lit model",
            reference=LiteratureReference(repo_url="https://github.com/example/repo"),
        )


def test_featurizer_role_validation_requires_contract() -> None:
    class Native:
        description = "native"
        tags: frozenset[str] = frozenset()
        reference = None

    with pytest.raises(ValueError, match="missing=\\['contract'\\]"):
        validate_registered_class(
            "drug_featurizer",
            "native",
            Native,
            required_fields=FeaturizerRegistry._required_fields,
        )


def test_predictor_role_validation_requires_contracts() -> None:
    class Native:
        description = "native"
        tags: frozenset[str] = frozenset()
        reference = None

    with pytest.raises(ValueError, match="missing=\\['cell_line_contract', 'drug_contract'\\]"):
        validate_registered_class(
            "predictor",
            "native",
            Native,
            required_fields=PredictorRegistry._required_fields,
        )


def test_missing_description_fails() -> None:
    class Empty:
        tags: frozenset[str] = frozenset()

    with pytest.raises(ValueError, match="missing=\\['description'\\]"):
        validate_registered_class(
            "predictor",
            "empty",
            Empty,
            required_fields=("description",),
        )


def test_wrong_type_contract_fails() -> None:
    class BadContract:
        description = "bad"
        tags: frozenset[str] = frozenset()
        reference = None
        contract = "numeric_matrix"

    with pytest.raises(ValueError, match="invalid=\\['contract'\\]"):
        validate_registered_class(
            "drug_featurizer",
            "bad",
            BadContract,
            required_fields=FeaturizerRegistry._required_fields,
        )


def test_register_rejects_blank_description_before_class_body() -> None:
    registry = FeaturizerRegistry("test", "Test", "tests")
    with pytest.raises(ValueError, match="description must be a non-empty string"):
        registry.register(
            "blank",
            description="  ",
            contract=FeatureFormat.NUMERIC_MATRIX,
        )


def test_register_rejects_bad_contract_before_class_body() -> None:
    registry = FeaturizerRegistry("test", "Test", "tests")
    with pytest.raises(
        (TypeError, Exception), match="FeatureContract|FeatureFormat|did not match any element in the union"
    ):
        registry.register(
            "badContract",
            description="demo",
            contract="numeric_matrix",  # type: ignore[arg-type]
        )
