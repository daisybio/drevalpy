"""Tests for registry metadata validation."""

from __future__ import annotations

import subprocess
import sys

import pytest

from drevalpy.components.registry._metadata_validate import validate_registered_class_metadata
from drevalpy.components.registry.common import apply_registration_metadata
from drevalpy.types.literature_reference import LiteratureReference


def test_literature_reference_is_accepted() -> None:
    class Lit:
        pass

    apply_registration_metadata(
        Lit,
        description="lit model",
        reference=LiteratureReference(
            repo_url="https://github.com/example/repo",
            citation_doi="10.1234/example",
            deviations="none",
        ),
    )
    validate_registered_class_metadata("predictor", "lit", Lit)


def test_literature_reference_missing_fields_fails() -> None:
    class Lit:
        pass

    apply_registration_metadata(
        Lit,
        description="lit model",
        reference=LiteratureReference(repo_url="https://github.com/example/repo"),
    )
    with pytest.raises(ValueError, match="metadata validation failed"):
        validate_registered_class_metadata("predictor", "lit", Lit)


def test_featurizer_metadata_requires_explicit_contract() -> None:
    class Native:
        pass

    apply_registration_metadata(
        Native,
        description="native",
    )
    with pytest.raises(ValueError, match="missing=\\['contract'\\]"):
        validate_registered_class_metadata("drug_featurizer", "native", Native)


def test_missing_description_fails() -> None:
    class Empty:
        tags = frozenset()

    with pytest.raises(ValueError, match="missing=\\['description'\\]"):
        validate_registered_class_metadata("predictor", "empty", Empty)


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
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout
