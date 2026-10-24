"""Tests for normalized registration metadata helpers."""

from __future__ import annotations

import pytest

from drevalpy.registry.components._registration_metadata import (
    RegistrationMetadata,
    apply_registration_metadata,
    normalize_registration_metadata,
)
from drevalpy.types.enums.literature_reference import LiteratureReference


def test_normalize_strips_tags_and_description() -> None:
    reference = LiteratureReference(
        repo_url="https://github.com/example/repo", citation_doi="10.1234/example", deviations="none"
    )

    metadata = normalize_registration_metadata("  demo  ", tags=("  baseline ", "", "omics"), reference=reference)

    assert metadata == RegistrationMetadata(
        description="demo", tags=frozenset({"baseline", "omics"}), reference=reference
    )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"description": "  "}, ValueError, "description must be a non-empty string"),
        ({"description": "demo", "tags": "baseline"}, TypeError, "iterable of strings"),
        ({"description": "demo", "tags": ("ok", 1)}, TypeError, "tags must contain strings"),
        (
            {"description": "demo", "reference": "not-a-reference"},
            Exception,
            "LiteratureReference|did not match any element in the union",
        ),
        (
            {"description": "demo", "reference": LiteratureReference(repo_url="https://github.com/example/repo")},
            ValueError,
            "invalid fields",
        ),
    ],
    ids=["blank-description", "bare-string-tags", "non-string-tags", "bad-reference-type", "incomplete-reference"],
)
def test_normalize_rejects_bad_input(kwargs: dict, error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        normalize_registration_metadata(**kwargs)


def test_apply_registration_metadata_assigns_fields() -> None:
    target = type("Target", (), {})

    apply_registration_metadata(
        target, RegistrationMetadata(description="demo", tags=frozenset({"baseline"}), reference=None)
    )

    assert (target.description, target.tags, target.reference) == ("demo", frozenset({"baseline"}), None)  # type: ignore[attr-defined]
