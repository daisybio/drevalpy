"""Post-decorator validation for registry class metadata."""

from __future__ import annotations

from typing import Any

from drevalpy.types.literature_reference import LiteratureReference

_FEATURIZER_REGISTRY_IDS = frozenset({"cell_line_featurizer", "drug_featurizer"})


def _is_valid_url(url: str) -> bool:
    return url.startswith(("http://", "https://"))


def validate_literature_reference(reference: LiteratureReference) -> list[str]:
    """Return invalid-field names for a literature reference, or an empty list."""
    invalid: list[str] = []
    if not reference.repo_url:
        invalid.append("repo_url")
    elif not _is_valid_url(reference.repo_url):
        invalid.append("repo_url")
    if not (reference.citation_text or reference.citation_doi):
        invalid.append("citation")
    if not reference.deviations:
        invalid.append("deviations")
    return invalid


def _missing_metadata_fields(registry_id: str, cls: type[Any]) -> list[str]:
    """Return required metadata fields missing from ``cls``."""
    missing: list[str] = []
    description = str(getattr(cls, "description", "") or "").strip()
    if not description:
        missing.append("description")
    if registry_id in _FEATURIZER_REGISTRY_IDS and "contract" not in cls.__dict__:
        missing.append("contract")
    return missing


def _has_invalid_tags(tags: object) -> bool:
    """Return whether ``tags`` is not a collection of non-empty strings."""
    if tags is None:
        return False
    if not isinstance(tags, (frozenset, set, list, tuple)):
        return True
    return any(not isinstance(tag, str) or not tag.strip() for tag in tags)


def _invalid_metadata_fields(cls: type[Any]) -> list[str]:
    """Return metadata fields with invalid values on ``cls``."""
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
    """Format a registry metadata validation error."""
    parts: list[str] = []
    if missing:
        parts.append(f"missing={missing}")
    if invalid:
        parts.append(f"invalid={invalid}")
    return f"{registry_id} '{name}' metadata validation failed: " + ", ".join(parts)


def validate_registered_class_metadata(
    registry_id: str,
    name: str,
    cls: type[Any],
) -> None:
    """Raise ``ValueError`` if class metadata is inconsistent or incomplete."""
    missing = _missing_metadata_fields(registry_id, cls)
    invalid = _invalid_metadata_fields(cls)
    if missing or invalid:
        raise ValueError(_format_validation_error(registry_id, name, missing=missing, invalid=invalid))
