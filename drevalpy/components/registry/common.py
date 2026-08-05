"""Shared metadata attachment for component registries."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from typing import Any, Protocol

from drevalpy.components.contracts import FeatureContract, FeatureFormat
from drevalpy.components.registry._contracts import _set_class_attribute, apply_registration_contracts
from drevalpy.components.registry._metadata_validate import validate_registered_class_metadata
from drevalpy.types.literature_reference import LiteratureReference


class RegistrationMetadataAttributes(Protocol):
    """Class variables attached to every registered component."""

    registry_name: str
    description: str
    tags: frozenset[str]
    reference: LiteratureReference | None


def apply_registration_metadata(
    cls: type[Any],
    *,
    description: str,
    tags: Iterable[str] | None = None,
    reference: LiteratureReference | None = None,
) -> None:
    """Attach ``description``, optional ``tags``, and optional literature ``reference``.

    :param cls: Class receiving registration metadata.
    :param description: description.
    :param tags: tags.
    :param reference: reference.
    :raises TypeError: Raised on invalid input.
    """
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
    """Build a class decorator that applies metadata, validates, and registers.

    :param registry: Mutable name-to-class store for the target registry.
    :param lock: Thread lock guarding registry mutations.
    :param registry_id: Registry identifier used for validation messages.
    :param name: Registry name under which decorated classes are stored.
    :param description: Short human-readable component summary.
    :param tags: Optional discovery tags attached to the class.
    :param reference: Optional literature citation metadata.
    :param contract: Featurizer feature-format contract override.
    :param cell_line_contract: Predictor cell-line contract override.
    :param drug_contract: Predictor drug contract override.
    :param already_registered_label: Label used in duplicate-registration errors.
    :returns: Class decorator that registers and returns the decorated class.
    """
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
            _set_class_attribute(cls, "registry_name", name)
        return cls

    return decorator
