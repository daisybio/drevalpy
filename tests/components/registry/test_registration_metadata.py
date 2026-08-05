"""Tests for normalized registration metadata helpers."""

from __future__ import annotations

import pytest

from drevalpy.components.registry._registration_metadata import (
    RegistrationMetadata,
    apply_registration_metadata,
    normalize_registration_metadata,
)
from drevalpy.types.literature_reference import LiteratureReference


def test_normalize_registration_metadata_strips_tags_and_description() -> None:
    metadata = normalize_registration_metadata(
        "  demo  ",
        tags=("  baseline ", "", "omics"),
        reference=LiteratureReference(
            repo_url="https://github.com/example/repo",
            citation_doi="10.1234/example",
            deviations="none",
        ),
    )
    assert metadata == RegistrationMetadata(
        description="demo",
        tags=frozenset({"baseline", "omics"}),
        reference=LiteratureReference(
            repo_url="https://github.com/example/repo",
            citation_doi="10.1234/example",
            deviations="none",
        ),
    )


def test_normalize_registration_metadata_rejects_blank_description() -> None:
    with pytest.raises(ValueError, match="description must be a non-empty string"):
        normalize_registration_metadata("  ")


def test_normalize_registration_metadata_rejects_bare_string_tags() -> None:
    with pytest.raises(TypeError, match="iterable of strings"):
        normalize_registration_metadata("demo", tags="baseline")  # type: ignore[arg-type]


def test_normalize_registration_metadata_rejects_non_string_tags() -> None:
    with pytest.raises(TypeError, match="tags must contain strings"):
        normalize_registration_metadata("demo", tags=("ok", 1))  # type: ignore[arg-type]


def test_normalize_registration_metadata_rejects_bad_reference_type() -> None:
    with pytest.raises(TypeError, match="LiteratureReference"):
        normalize_registration_metadata("demo", reference="not-a-reference")  # type: ignore[arg-type]


def test_normalize_registration_metadata_rejects_incomplete_reference() -> None:
    with pytest.raises(ValueError, match="invalid fields"):
        normalize_registration_metadata(
            "demo",
            reference=LiteratureReference(repo_url="https://github.com/example/repo"),
        )


def test_apply_registration_metadata_assigns_fields() -> None:
    class Target:
        pass

    metadata = RegistrationMetadata(
        description="demo",
        tags=frozenset({"baseline"}),
        reference=None,
    )
    apply_registration_metadata(Target, metadata)
    assert Target.description == "demo"
    assert Target.tags == frozenset({"baseline"})
    assert Target.reference is None
