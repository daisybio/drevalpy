"""Tests for shared registry base helpers."""

from __future__ import annotations

import pytest

from drevalpy.components.registry.base import apply_shared_registration_metadata
from drevalpy.types.literature_reference import LiteratureReference


def test_apply_shared_registration_metadata_rejects_bad_reference() -> None:
    class Target:
        pass

    with pytest.raises(TypeError, match="LiteratureReference"):
        apply_shared_registration_metadata(
            Target,
            description="demo",
            reference="not-a-reference",  # type: ignore[arg-type]
        )


def test_apply_shared_registration_metadata_normalizes_tags() -> None:
    class Target:
        pass

    apply_shared_registration_metadata(
        Target,
        description="demo",
        tags=("  baseline ", "", "omics"),
        reference=LiteratureReference(
            repo_url="https://github.com/example/repo",
            citation_doi="10.1234/example",
            deviations="none",
        ),
    )
    assert Target.description == "demo"
    assert Target.tags == frozenset({"baseline", "omics"})
