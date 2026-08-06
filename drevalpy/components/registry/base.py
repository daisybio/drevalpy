"""Shared registry base and registration helpers."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from drevalpy.components.contracts import FeatureContract
from drevalpy.components.registry._metadata_validate import validate_registered_class


class Registry(ABC):
    """Thread-safe name-to-class registry with shared store and metadata listing."""

    _required_fields: ClassVar[tuple[str, ...]] = ("description",)

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

    def _apply_contract(self, cls: type[Any], attr_name: str, contract: FeatureContract) -> None:
        """Assign a normalized contract attribute, rejecting class-body definitions.

        :param cls: Class being registered.
        :param attr_name: Attribute name such as ``contract`` or ``cell_line_contract``.
        :param contract: Already-normalized feature contract to attach.
        :raises ValueError: If *attr_name* is already defined on the class body.
        """
        if attr_name in cls.__dict__:
            msg = (
                f"{cls.__name__}: do not set {attr_name} on the class body; "
                "pass it to the registration decorator instead"
            )
            raise ValueError(msg)
        setattr(cls, attr_name, contract)

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
            validate_registered_class(
                self._registry_id,
                name,
                cls,
                required_fields=self._required_fields,
            )
            self._validate_registration(name, cls)
            self._store[name] = cls
            cls.registry_name = name

    def _validate_registration(self, name: str, cls: type[Any]) -> None:
        """Run registry-specific class invariants after metadata validation.

        :param name: Registry name under which *cls* is being registered.
        :param cls: Component class being registered or restored.
        """
        return

    @abstractmethod
    def _component_metadata(self, name: str, cls: type[Any]) -> dict[str, Any]:
        """Return component metadata for a registered class.

        :param name: Registry name of the component.
        :param cls: Registered component class.
        :returns: Metadata dict.
        """
