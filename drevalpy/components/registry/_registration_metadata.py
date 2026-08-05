"""Normalized shared registration metadata for fresh decorator registration."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from drevalpy.components.registry._metadata_validate import validate_literature_reference
from drevalpy.types.literature_reference import LiteratureReference


@dataclass(frozen=True)
class RegistrationMetadata:
    """Validated description, tags, and optional literature reference."""

    description: str
    tags: frozenset[str]
    reference: LiteratureReference | None


def _normalize_description(description: str) -> str:
    stripped = str(description).strip()
    if not stripped:
        msg = "description must be a non-empty string"
        raise ValueError(msg)
    return stripped


def _normalize_tags(tags: Iterable[str] | None) -> frozenset[str]:
    if tags is None:
        return frozenset()
    if isinstance(tags, str):
        msg = "tags must be an iterable of strings, not a bare string"
        raise TypeError(msg)
    cleaned: list[str] = []
    for tag in tags:
        if not isinstance(tag, str):
            msg = f"tags must contain strings, got {type(tag).__name__}"
            raise TypeError(msg)
        stripped = tag.strip()
        if stripped:
            cleaned.append(stripped)
    return frozenset(cleaned)


def _normalize_reference(reference: LiteratureReference | None) -> LiteratureReference | None:
    if reference is None:
        return None
    if not isinstance(reference, LiteratureReference):
        msg = f"reference must be LiteratureReference, got {type(reference).__name__}"
        raise TypeError(msg)
    invalid = validate_literature_reference(reference)
    if invalid:
        msg = f"reference has invalid fields: {invalid}"
        raise ValueError(msg)
    return reference


def normalize_registration_metadata(
    description: str,
    tags: Iterable[str] | None = None,
    reference: LiteratureReference | None = None,
) -> RegistrationMetadata:
    """Validate and normalize shared registration kwargs.

    :param description: Short human-readable summary.
    :param tags: Optional discovery tags.
    :param reference: Optional literature citation metadata.
    :returns: Normalized registration metadata.
    """
    return RegistrationMetadata(
        description=_normalize_description(description),
        tags=_normalize_tags(tags),
        reference=_normalize_reference(reference),
    )


def apply_registration_metadata(cls: type[Any], metadata: RegistrationMetadata) -> None:
    """Attach already-normalized registration metadata to *cls*.

    :param cls: Class receiving registration metadata.
    :param metadata: Normalized description, tags, and optional reference.
    """
    cls.description = metadata.description
    cls.tags = metadata.tags
    cls.reference = metadata.reference
