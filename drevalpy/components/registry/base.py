"""Shared registry base and registration helpers."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

from drevalpy.components.registry._metadata_validate import validate_shared_registration_metadata
from drevalpy.types.literature_reference import LiteratureReference


def apply_shared_registration_metadata(
    cls: type[Any],
    *,
    description: str,
    tags: Iterable[str] | None = None,
    reference: LiteratureReference | None = None,
) -> None:
    """Attach ``description``, optional ``tags``, and optional literature ``reference``.

    :param cls: Class receiving registration metadata.
    :param description: Short human-readable summary.
    :param tags: Optional discovery tags.
    :param reference: Optional literature citation metadata.
    :raises TypeError: If *reference* is not a ``LiteratureReference``.
    """
    cls.description = description
    normalized_tags = frozenset(str(tag).strip() for tag in (tags or ()) if str(tag).strip())
    cls.tags = normalized_tags
    if reference is not None and not isinstance(reference, LiteratureReference):
        msg = f"reference must be LiteratureReference, got {type(reference).__name__}"
        raise TypeError(msg)
    cls.reference = reference


class Registry(ABC):
    """Thread-safe name-to-class registry with shared store and metadata listing."""

    def __init__(self, registry_id: str, label: str, display_name: str) -> None:
        """Initialize an empty registry store.

        :param registry_id: Stable identifier used in validation messages.
        :param label: Human-readable label for unknown/duplicate errors.
        :param display_name: Catalog registry name written into component metadata.
        """
        self._registry_id = registry_id
        self._label = label
        self._display_name = display_name
        self._store: dict[str, type[Any]] = {}
        self._lock = threading.Lock()

    def get(self, name: str) -> type[Any]:
        """Return the class registered under *name*.

        :param name: Registry name of the component.

        :returns: Registered component class.

        :raises ValueError: If *name* is not registered.
        """
        with self._lock:
            if name not in self._store:
                available = list(self._store.keys())
                raise ValueError(f"Unknown {self._label}: {name!r}. Available: {available}")
            return self._store[name]

    def list_names(self) -> list[str]:
        """Return all registered names.

        :returns: Sorted list of registry names currently stored.
        """
        with self._lock:
            return list(self._store.keys())

    def get_metadata(self, name: str) -> dict[str, Any]:
        """Return the metadata record for the component registered under *name*.

        :param name: Registry name of the component.

        :returns: Metadata dict for catalog listings.
        """
        cls = self.get(name)
        return self._component_metadata(name, cls)

    def list_metadata(self, *, tag: str | None = None) -> list[dict[str, Any]]:
        """Return metadata for all components, optionally filtered by discovery tag.

        :param tag: When set, keep only components whose ``tags`` contain *tag*.

        :returns: List of metadata dicts.
        """
        rows = [self.get_metadata(name) for name in self.list_names()]
        if tag is None:
            return rows
        needle = tag.strip()
        return [row for row in rows if needle in row.get("tags", frozenset())]

    def clear(self) -> None:
        """Remove all entries (primarily for testing)."""
        with self._lock:
            self._store.clear()

    def retain_only(self, names: frozenset[str]) -> None:
        """Drop entries whose names are not in *names*.

        :param names: Registry names to keep after rollback or partial unload.
        """
        with self._lock:
            for registered_name in list(self._store):
                if registered_name not in names:
                    del self._store[registered_name]

    def register_existing(self, name: str, cls: type[Any]) -> None:
        """Register a class that was previously decorated but removed via ``clear``.

        :param name: Registry name under which *cls* should be restored.
        :param cls: Component class with registration metadata attributes.
        """
        with self._lock:
            if name in self._store:
                return
            validate_shared_registration_metadata(self._registry_id, name, cls)
            self._validate_role(cls, name)
            self._store[name] = cls
            cls.registry_name = name

    @abstractmethod
    def _validate_role(self, cls: type[Any], name: str) -> None:
        """Raise ``ValueError`` when role-specific registration attributes are missing.

        :param cls: Class being validated.
        :param name: Registry name used in error messages.
        """

    @abstractmethod
    def _component_metadata(self, name: str, cls: type[Any]) -> dict[str, Any]:
        """Return component metadata for a registered class.

        :param name: Registry name of the component.
        :param cls: Registered component class.
        :returns: Metadata dict.
        """
