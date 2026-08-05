"""Thread-safe registries for cell-line featurizers, drug featurizers, and predictors."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from typing import Any

from drevalpy.components.contracts import FeatureContract, FeatureFormat
from drevalpy.components.registry._contracts import _set_class_attribute
from drevalpy.components.registry._metadata_validate import validate_registered_class_metadata
from drevalpy.components.registry.common import make_registration_decorator
from drevalpy.components.registry.metadata import (
    featurizer_component_metadata,
    predictor_component_metadata,
)
from drevalpy.types.literature_reference import LiteratureReference


class Registry:
    """Thread-safe name-to-class registry with metadata validation."""

    def __init__(
        self,
        registry_id: str,
        label: str,
        display_name: str,
        metadata_fn: Callable[[str, str, type[Any]], dict[str, str]],
    ) -> None:
        """Initialize instance state.

        :param registry_id: registry id.
        :param label: label.
        :param display_name: display name.
        :param metadata_fn: metadata fn.
        """
        self._registry_id = registry_id
        self._label = label
        self._display_name = display_name
        self._metadata_fn = metadata_fn
        self._store: dict[str, type[Any]] = {}
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        *,
        description: str,
        tags: Iterable[str] | None = None,
        reference: LiteratureReference | None = None,
        contract: FeatureContract | FeatureFormat | None = None,
        cell_line_contract: FeatureContract | FeatureFormat | None = None,
        drug_contract: FeatureContract | FeatureFormat | None = None,
    ) -> Callable[[type[Any]], type[Any]]:
        """Return a class decorator that registers the decorated class under *name*.

        :param name: Registry name used in model configs and discovery listings.
        :param description: Short human-readable summary.
        :param tags: Optional discovery tags.
        :param reference: Optional literature citation metadata.
        :param contract: Feature format contract for featurizer registries.
        :param cell_line_contract: Expected cell-line feature format for predictors.
        :param drug_contract: Expected drug feature format for predictors.

        :returns: Class decorator that registers and returns the decorated class.
        """
        return make_registration_decorator(
            self._store,
            self._lock,
            self._registry_id,
            name,
            description=description,
            tags=tags,
            reference=reference,
            contract=contract,
            cell_line_contract=cell_line_contract,
            drug_contract=drug_contract,
            already_registered_label=self._label,
        )

    def get(self, name: str) -> type[Any]:
        """Return the class registered under *name*.

        :param name: Registry name of the component.

        :returns: Registered component class.

        :raises ValueError: If *name* is not registered.
        """
        with self._lock:
            if name not in self._store:
                available = list(self._store.keys())
                msg = f"Unknown {self._label}: {name!r}. Available: {available}"
                raise ValueError(msg)
            return self._store[name]

    def list_names(self) -> list[str]:
        """Return all registered names.

        :returns: Sorted list of registry names currently stored.
        """
        with self._lock:
            return list(self._store.keys())

    def get_metadata(self, name: str) -> dict[str, str]:
        """Return the metadata record for the component registered under *name*.

        :param name: Registry name of the component.

        :returns: Flattened metadata dict for catalog listings.
        """
        cls = self.get(name)
        return self._metadata_fn(self._display_name, name, cls)

    def list_metadata(self, *, tag: str | None = None) -> list[dict[str, str]]:
        """Return metadata for all components, optionally filtered by discovery tag.

        :param tag: When set, keep only components whose ``tags`` field contains *tag*.

        :returns: List of flattened metadata dicts.
        """
        rows = [self.get_metadata(name) for name in self.list_names()]
        if tag is None:
            return rows
        needle = tag.strip()
        return [row for row in rows if needle in {part for part in row.get("tags", "").split(",") if part}]

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
            validate_registered_class_metadata(self._registry_id, name, cls)
            self._store[name] = cls
            _set_class_attribute(cls, "registry_name", name)


cell_line_featurizer_registry = Registry(
    "cell_line_featurizer",
    "Cell line featurizer",
    "cell_line_featurizers",
    featurizer_component_metadata,
)
drug_featurizer_registry = Registry(
    "drug_featurizer",
    "Drug featurizer",
    "drug_featurizers",
    featurizer_component_metadata,
)
predictor_registry = Registry(
    "predictor",
    "Predictor",
    "predictors",
    predictor_component_metadata,
)
