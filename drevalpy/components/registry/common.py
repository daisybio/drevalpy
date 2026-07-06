"""Shared metadata validation for component registries."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from drevalpy.components.registry._metadata_validate import (
    _VALID_CATEGORIES,
    validate_registered_class_metadata,
)


def apply_registration_metadata(
    cls: type[Any],
    *,
    description: str,
    category: str,
    template_repo_url: str = "",
    citation: str = "",
    citation_doi: str = "",
    citation_text: str = "",
    deviations: str = "",
) -> None:
    """Attach ``description`` / ``category`` and optional reference fields to *cls*."""
    if category not in _VALID_CATEGORIES:
        msg = f"category must be one of {sorted(_VALID_CATEGORIES)}, got {category!r}"
        raise ValueError(msg)

    cite_any = bool(
        (template_repo_url or "").strip()
        or (citation or "").strip()
        or (citation_doi or "").strip()
        or (citation_text or "").strip()
    )
    dev_any = bool((deviations or "").strip())

    if category == "literature":
        cls.description = description
        cls.category = category
        cls.template_repo_url = template_repo_url
        cls.citation = citation
        cls.citation_doi = citation_doi
        cls.citation_text = citation_text
        cls.deviations = deviations
        return

    if category == "general_purpose":
        if dev_any:
            msg = "deviations is only allowed when category='literature', not 'general_purpose'"
            raise ValueError(msg)
        cls.description = description
        cls.category = category
        cls.template_repo_url = template_repo_url
        cls.citation = citation
        cls.citation_doi = citation_doi
        cls.citation_text = citation_text
        return

    if cite_any or dev_any:
        msg = (
            "template_repo_url, citation, citation_doi, citation_text, and deviations "
            f"are not allowed when category is {category!r}"
        )
        raise ValueError(msg)
    cls.description = description
    cls.category = category


def make_registration_decorator(
    registry: dict[str, type[Any]],
    lock: threading.Lock,
    registry_id: str,
    name: str,
    *,
    description: str,
    category: str,
    template_repo_url: str = "",
    citation: str = "",
    citation_doi: str = "",
    citation_text: str = "",
    deviations: str = "",
    already_registered_label: str | None = None,
) -> Callable[[type[Any]], type[Any]]:
    """Build a class decorator that applies metadata, validates, and registers."""
    dup = already_registered_label or registry_id

    def decorator(cls: type[Any]) -> type[Any]:
        with lock:
            if name in registry:
                msg = f"{dup} {name!r} already registered"
                raise ValueError(msg)
            apply_registration_metadata(
                cls,
                description=description,
                category=category,
                template_repo_url=template_repo_url,
                citation=citation,
                citation_doi=citation_doi,
                citation_text=citation_text,
                deviations=deviations,
            )
            validate_registered_class_metadata(registry_id, name, cls)
            registry[name] = cls
            cls.registry_name = name  # type: ignore[attr-defined]
        return cls

    return decorator
