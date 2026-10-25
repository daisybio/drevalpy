"""Class-state validation for restored registered components."""

from __future__ import annotations

from typing import Any

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.registry.components._metadata import LiteratureReference

_FORMAT_FIELDS = frozenset({"output_format", "sample_input_format", "drug_input_format"})


def _is_valid_url(url: str) -> bool:
    return url.startswith(("http://", "https://"))


def validate_literature_reference(reference: LiteratureReference) -> list[str]:
    """Return invalid-field names for a literature reference, or an empty list.

    :param reference: Literature reference attached to a registered class.
    :returns: Names of the invalid fields (``repo_url``, ``citation``, ``deviations``).
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
    if not isinstance(tags, frozenset):
        return True
    return any(not isinstance(tag, str) or not tag for tag in tags)


def _missing_fields(cls: type[Any], required_fields: tuple[str, ...]) -> list[str]:
    """Return required fields missing from ``cls`` or present as blank strings.

    :param cls: Registered component class.
    :param required_fields: Field names that must appear on the class body.
    :returns: Names of required fields that are absent or empty.
    """
    missing: list[str] = []
    for field in required_fields:
        if field not in cls.__dict__:
            missing.append(field)
            continue
        value = cls.__dict__[field]
        if isinstance(value, str) and not value.strip():
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


def _invalid_format_fields(cls: type[Any], required_fields: tuple[str, ...]) -> list[str]:
    """Return required format fields that are not ``FeatureFormat`` members.

    :param cls: Registered component class.
    :param required_fields: Field names required on the class body.
    :returns: Format field names with the wrong runtime type.
    """
    invalid: list[str] = []
    for field in required_fields:
        if field not in _FORMAT_FIELDS or field not in cls.__dict__:
            continue
        if not isinstance(cls.__dict__[field], FeatureFormat):
            invalid.append(field)
    return invalid


def _format_validation_error(
    registry_id: str,
    name: str,
    *,
    missing: list[str],
    invalid: list[str],
) -> str:
    """Format a registry metadata validation error.

    :param registry_id: Identifier of the registry, named in the message.
    :param name: Registry name of the component, named in the message.
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


def validate_registered_class(
    registry_id: str,
    name: str,
    cls: type[Any],
    *,
    required_fields: tuple[str, ...],
) -> None:
    """Raise ``ValueError`` if a restored class lacks valid registration state.

    Used by ``register_existing`` when decorator kwargs are unavailable.

    :param registry_id: Registry identifier, used in the error message.
    :param name: Component registry name.
    :param cls: Previously registered component class.
    :param required_fields: Field names required on the class body for this registry.
    :raises ValueError: If a required field is missing or a shared or format field is invalid.
    """
    missing = _missing_fields(cls, required_fields)
    invalid = _invalid_shared_fields(cls)
    invalid.extend(_invalid_format_fields(cls, required_fields))
    if missing or invalid:
        raise ValueError(_format_validation_error(registry_id, name, missing=missing, invalid=invalid))
