"""Shared metadata attachment for component registries."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from typing import Any

from drevalpy.components.contracts import FeatureContract, FeatureFormat
from drevalpy.components.registry._contracts import apply_registration_contracts
from drevalpy.components.registry._metadata_validate import validate_registered_class_metadata
from drevalpy.types.literature_reference import LiteratureReference


def apply_registration_metadata(
    cls: type[Any],
    *,
    description: str,
    tags: Iterable[str] | None = None,
    reference: LiteratureReference | None = None,
) -> None:
    """Attach ``description``, optional ``tags``, and optional literature ``reference``."""
    cls.description = description
    normalized_tags = frozenset(str(tag).strip() for tag in (tags or ()) if str(tag).strip())
    cls.tags = normalized_tags
    if reference is not None and not isinstance(reference, LiteratureReference):
        msg = f"reference must be LiteratureReference, got {type(reference).__name__}"
        raise TypeError(msg)
    cls.reference = reference


def make_registration_decorator(
    registry: dict[str, type[Any]],
    lock: threading.Lock,
    registry_id: str,
    name: str,
    *,
    description: str,
    tags: Iterable[str] | None = None,
    reference: LiteratureReference | None = None,
    contract: FeatureContract | FeatureFormat | None = None,
    cell_line_contract: FeatureContract | FeatureFormat | None = None,
    drug_contract: FeatureContract | FeatureFormat | None = None,
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
                tags=tags,
                reference=reference,
            )
            apply_registration_contracts(
                cls,
                contract=contract,
                cell_line_contract=cell_line_contract,
                drug_contract=drug_contract,
            )
            validate_registered_class_metadata(registry_id, name, cls)
            registry[name] = cls
            cls.registry_name = name  # type: ignore[attr-defined]
        return cls

    return decorator
