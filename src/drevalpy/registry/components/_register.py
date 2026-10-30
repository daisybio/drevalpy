"""Store a featurizer or predictor class in its registry after the shared checks.

The predictor and featurizer decorators first stamp what their own kind needs onto
the class (feature formats, side); :func:`register_component` then adds what every
component shares: the registration metadata on the class, the abstract-member check,
the kind's own invariants, the registry entry and ``registry_name``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from drevalpy.registry._registry import Registry
from drevalpy.registry.components._abstract import validate_no_abstract_methods
from drevalpy.registry.components._registration_metadata import RegistrationMetadata, apply_registration_metadata


def register_component(
    registry: Registry[type[Any]],
    registry_id: str,
    name: str,
    cls: type[Any],
    metadata: RegistrationMetadata,
    validate: Callable[[str, type[Any]], None],
) -> type[Any]:
    """Validate *cls* and register it under *name*.

    A duplicate *name* makes :meth:`Registry.register` raise ``ValueError``.

    :param registry: Registry of the component kind.
    :param registry_id: Identifier named in validation messages, such as ``"predictor"``.
    :param name: Registry name of the component.
    :param cls: Component class with its kind-specific attributes already stamped.
    :param metadata: Normalized description, tags and reference.
    :param validate: Invariants of the component kind, called as ``validate(name, cls)``.
    :returns: *cls*, so the call can end a decorator.
    """
    apply_registration_metadata(cls, metadata)
    validate_no_abstract_methods(registry_id, name, cls)
    validate(name, cls)
    registry.register(name, cls, metadata={"description": metadata.description, "tags": metadata.tags})
    cls.registry_name = name
    return cls
