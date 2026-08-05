"""Tests for registry metadata validation and role checks."""

from __future__ import annotations

import pytest

from drevalpy.components.contracts import FeatureContract, FeatureFormat
from drevalpy.components.registry._metadata_validate import validate_shared_registration_metadata
from drevalpy.components.registry.base import apply_shared_registration_metadata
from drevalpy.components.registry.featurizer import DrugFeaturizerRegistry
from drevalpy.components.registry.predictor import PredictorRegistry
from drevalpy.types.literature_reference import LiteratureReference
from tests._trusted_subprocess import run_trusted_python


def test_literature_reference_is_accepted() -> None:
    class Lit:
        cell_line_contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
        drug_contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)

    apply_shared_registration_metadata(
        Lit,
        description="lit model",
        reference=LiteratureReference(
            repo_url="https://github.com/example/repo",
            citation_doi="10.1234/example",
            deviations="none",
        ),
    )
    validate_shared_registration_metadata("predictor", "lit", Lit)


def test_literature_reference_missing_fields_fails() -> None:
    class Lit:
        cell_line_contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
        drug_contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)

    apply_shared_registration_metadata(
        Lit,
        description="lit model",
        reference=LiteratureReference(repo_url="https://github.com/example/repo"),
    )
    with pytest.raises(ValueError, match="metadata validation failed"):
        validate_shared_registration_metadata("predictor", "lit", Lit)


def test_featurizer_role_validation_requires_contract() -> None:
    registry = DrugFeaturizerRegistry()

    class Native:
        description = "native"
        tags = frozenset()
        reference = None

    with pytest.raises(ValueError, match="missing=\\['contract'\\]"):
        registry._validate_role(Native, "native")


def test_predictor_role_validation_requires_contracts() -> None:
    registry = PredictorRegistry()

    class Native:
        description = "native"
        tags = frozenset()
        reference = None

    with pytest.raises(ValueError, match="missing=\\['cell_line_contract', 'drug_contract'\\]"):
        registry._validate_role(Native, "native")


def test_missing_description_fails() -> None:
    class Empty:
        tags: frozenset[str] = frozenset()

    with pytest.raises(ValueError, match="missing=\\['description'\\]"):
        validate_shared_registration_metadata("predictor", "empty", Empty)


def test_fresh_process_discovery_returns_all_builtins() -> None:
    script = """
from drevalpy.components.registry import (
    list_cell_line_featurizer_metadata,
    list_drug_featurizer_metadata,
    list_predictor_metadata,
)
assert len(list_cell_line_featurizer_metadata()) == 17
assert len(list_drug_featurizer_metadata()) == 9
assert len(list_predictor_metadata()) == 27
print("ok")
"""
    completed = run_trusted_python(script)
    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout
