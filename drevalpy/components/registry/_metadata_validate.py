"""Post-decorator validation for registry class metadata."""

from __future__ import annotations

from typing import Any

_VALID_CATEGORIES = frozenset({"literature", "baseline", "general_purpose", "native"})
_FORBIDDEN_ON_BASELINE_NATIVE: frozenset[str] = frozenset(
    (
        "template_repo_url",
        "citation",
        "citation_doi",
        "citation_text",
        "deviations",
    )
)


def _is_valid_url(url: str) -> bool:
    return url.startswith(("http://", "https://"))


def _metadata_fields_from_class(cls: type[Any]) -> dict[str, str]:
    return {
        "description": str(getattr(cls, "description", "") or "").strip(),
        "category": str(getattr(cls, "category", "") or "").strip(),
        "template_repo_url": str(getattr(cls, "template_repo_url", "") or "").strip(),
        "citation": str(getattr(cls, "citation", "") or "").strip(),
        "citation_doi": str(getattr(cls, "citation_doi", "") or "").strip(),
        "citation_text": str(getattr(cls, "citation_text", "") or "").strip(),
        "deviations": str(getattr(cls, "deviations", "") or "").strip(),
    }


def _validate_required_base(metadata: dict[str, str]) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    invalid: list[str] = []
    if not metadata["description"]:
        missing.append("description")
    if not metadata["category"]:
        missing.append("category")
    elif metadata["category"] not in _VALID_CATEGORIES:
        invalid.append("category")
    return missing, invalid


def _validate_literature(metadata: dict[str, str]) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    invalid: list[str] = []
    if not metadata["template_repo_url"]:
        missing.append("template_repo_url")
    elif not _is_valid_url(metadata["template_repo_url"]):
        invalid.append("template_repo_url")
    has_cite = metadata["citation"] or metadata["citation_doi"] or metadata["citation_text"]
    if not has_cite:
        missing.append("citation")
    if not metadata["deviations"]:
        missing.append("deviations")
    return missing, invalid


def _validate_non_literature_explicit_fields(cls: type[Any], *, category: str) -> list[str]:
    if category in {"baseline", "native"}:
        bad = [field for field in _FORBIDDEN_ON_BASELINE_NATIVE if field in cls.__dict__]
        return [f"non_literature_explicit={bad}"] if bad else []
    if category == "general_purpose":
        if "deviations" in cls.__dict__:
            return ["deviations_only_for_literature"]
        return []
    return []


def validate_registered_class_metadata(
    registry_id: str,
    name: str,
    cls: type[Any],
) -> None:
    """Raise ``ValueError`` if class metadata is inconsistent or incomplete."""
    meta = _metadata_fields_from_class(cls)
    missing, invalid = _validate_required_base(meta)
    category = meta["category"]
    if category == "literature":
        lit_m, lit_i = _validate_literature(meta)
        missing.extend(lit_m)
        invalid.extend(lit_i)
    elif category in {"baseline", "general_purpose", "native"}:
        invalid.extend(_validate_non_literature_explicit_fields(cls, category=category))

    if missing or invalid:
        parts: list[str] = []
        if missing:
            parts.append(f"missing={missing}")
        if invalid:
            parts.append(f"invalid={invalid}")
        msg = f"{registry_id} '{name}' metadata validation failed: " + ", ".join(parts)
        raise ValueError(msg)
