"""Tests for registry metadata validation."""

from __future__ import annotations

import pytest

from drevalpy.components.registry.common import apply_registration_metadata
from drevalpy.components.registry._metadata_validate import validate_registered_class_metadata


def test_literature_metadata_requires_citation_and_repo() -> None:
    class Lit:
        pass

    apply_registration_metadata(
        Lit,
        description="lit model",
        category="literature",
        template_repo_url="https://github.com/example/repo",
        citation_doi="10.1234/example",
        deviations="none",
    )
    validate_registered_class_metadata("predictor", "lit", Lit)


def test_literature_metadata_missing_fields_fails() -> None:
    class Lit:
        description = "lit model"
        category = "literature"

    with pytest.raises(ValueError, match="metadata validation failed"):
        validate_registered_class_metadata("predictor", "lit", Lit)


def test_native_metadata_rejects_citation_fields_on_class() -> None:
    class Native:
        description = "native"
        category = "native"
        citation = "should not be here"

    with pytest.raises(ValueError, match="metadata validation failed"):
        validate_registered_class_metadata("cell_line_featurizer", "native", Native)
