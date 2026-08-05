"""Shared registration metadata validation (description, tags, literature)."""

from __future__ import annotations

from typing import Any

from drevalpy.types.literature_reference import LiteratureReference


def _is_valid_url(url: str) -> bool:
    return url.startswith(("http://", "https://"))


def validate_literature_reference(reference: LiteratureReference) -> list[str]:
    """Return invalid-field names for a literature reference, or an empty list.

    :param reference: reference.
    :returns: Result.
    """
    invalid: list[str] = []
    if not reference.repo_url or not _is_valid_url(reference.repo_url):
        invalid.append("repo_url")
    if not (reference.citation_text or reference.citation_doi):
        invalid.append("citation")
    if not reference.deviations:
        invalid.append("deviations")
    return invalid


def _has_invalid_tags(tags: object) -> bool:
    """Return ``True`` if ``tags`` fails the registration invariant.

    Valid tags are a ``frozenset`` of non-empty strings (use an empty
    ``frozenset`` when there are no tags). Anything else is invalid.

    :param tags: Candidate tag collection from a registered class.
    :returns: ``True`` when *tags* is not a valid ``frozenset`` of non-empty strings.
    """
    is_frozenset = isinstance(tags, frozenset)
    if not is_frozenset:
        return True
    all_nonempty_strings = all(isinstance(tag, str) and tag.strip() for tag in tags)
    return not all_nonempty_strings


def _missing_shared_fields(cls: type[Any]) -> list[str]:
    """Return required shared metadata fields missing from ``cls``.

    :param cls: Registered component class.
    :returns: Names of required metadata fields that are absent or empty.
    """
    required = ("description",)
    missing: list[str] = []
    for field in required:
        if not str(getattr(cls, field, "") or "").strip():
            missing.append(field)
    return missing


def _invalid_shared_fields(cls: type[Any]) -> list[str]:
    """Return shared metadata fields with invalid values on ``cls``.

    :param cls: Registered component class.
    :returns: Names of metadata fields with invalid values.
    """
    invalid: list[str] = []
    if _has_invalid_tags(getattr(cls, "tags", frozenset())):
        invalid.append("tags")

    reference = getattr(cls, "reference", None)
    if reference is None:
        return invalid
    if not isinstance(reference, LiteratureReference):
        invalid.append("reference")
        return invalid
    invalid.extend(validate_literature_reference(reference))
    return invalid


def _format_validation_error(
    registry_id: str,
    name: str,
    *,
    missing: list[str],
    invalid: list[str],
) -> str:
    """Format a registry metadata validation error.

    :param registry_id: Registry identifier.
    :param name: Component registry name.
    :param missing: Required metadata fields that are absent.
    :param invalid: Metadata fields with invalid values.
    :returns: Human-readable validation error message.
    """
    parts: list[str] = []
    if missing:
        parts.append(f"missing={missing}")
    if invalid:
        parts.append(f"invalid={invalid}")
    return f"{registry_id} '{name}' metadata validation failed: " + ", ".join(parts)


def validate_shared_registration_metadata(
    registry_id: str,
    name: str,
    cls: type[Any],
) -> None:
    """Raise ``ValueError`` if shared class metadata is inconsistent or incomplete.

    Validates description, tags, and literature reference. Role-specific contract
    checks live on ``FeaturizerRegistry`` / ``PredictorRegistry``.

    :param registry_id: registry id.
    :param name: name.
    :param cls: Registered component class.
    :raises ValueError: Raised on invalid input.
    """
    missing = _missing_shared_fields(cls)
    invalid = _invalid_shared_fields(cls)
    if missing or invalid:
        raise ValueError(_format_validation_error(registry_id, name, missing=missing, invalid=invalid))
