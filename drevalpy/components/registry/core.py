"""Thread-safe registries for cell-line featurizers, drug featurizers, and predictors."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any

from drevalpy.components.contracts import FeatureContract, FeatureFormat, normalize_feature_contract
from drevalpy.components.registry._metadata_validate import (
    format_validation_error,
    validate_shared_registration_metadata,
)
from drevalpy.components.registry.metadata import (
    featurizer_component_metadata,
    predictor_component_metadata,
)
from drevalpy.types.literature_reference import LiteratureReference


def _set_class_attribute(cls: type[Any], name: str, value: object) -> None:
    """Assign a registration attribute on *cls* via ``setattr``.

    :param cls: Component class receiving the attribute.
    :param name: Attribute name to set.
    :param value: Attribute value to assign.
    """
    setattr(cls, name, value)


def _contract_defined_on_class(cls: type[Any], *attr_names: str) -> bool:
    return any(name in cls.__dict__ for name in attr_names)


def _apply_shared_registration_metadata(
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
        """Initialize instance state.

        :param registry_id: Stable identifier used in validation messages.
        :param label: Human-readable label for unknown/duplicate errors.
        :param display_name: Catalog registry name written into metadata rows.
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
        return self._metadata_row(name, cls)

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
            validate_shared_registration_metadata(self._registry_id, name, cls)
            self._validate_role(cls, name)
            self._store[name] = cls
            _set_class_attribute(cls, "registry_name", name)

    @abstractmethod
    def _validate_role(self, cls: type[Any], name: str) -> None:
        """Raise ``ValueError`` when role-specific registration attributes are missing.

        :param cls: Class being validated.
        :param name: Registry name used in error messages.
        """

    @abstractmethod
    def _metadata_row(self, name: str, cls: type[Any]) -> dict[str, str]:
        """Return the flattened metadata row for a registered class.

        :param name: Registry name of the component.
        :param cls: Registered component class.
        :returns: Flattened metadata dict.
        """


class FeaturizerRegistry(Registry):
    """Registry for cell-line or drug featurizers that emit one feature contract."""

    def register(
        self,
        name: str,
        *,
        description: str,
        contract: FeatureContract | FeatureFormat,
        tags: Iterable[str] | None = None,
        reference: LiteratureReference | None = None,
    ) -> Callable[[type[Any]], type[Any]]:
        """Return a class decorator that registers a featurizer under *name*.

        :param name: Registry name used in model configs and discovery listings.
        :param description: Short human-readable summary.
        :param contract: Feature format contract for predictor matching.
        :param tags: Optional discovery tags.
        :param reference: Optional literature citation metadata.

        :returns: Class decorator that registers and returns the decorated class.
        """

        def decorator(cls: type[Any]) -> type[Any]:
            with self._lock:
                if name in self._store:
                    msg = f"{self._label} {name!r} already registered"
                    raise ValueError(msg)
                self._apply_contract(cls, contract)
                _apply_shared_registration_metadata(
                    cls,
                    description=description,
                    tags=tags,
                    reference=reference,
                )
                validate_shared_registration_metadata(self._registry_id, name, cls)
                self._validate_role(cls, name)
                self._store[name] = cls
                _set_class_attribute(cls, "registry_name", name)
            return cls

        return decorator

    def _apply_contract(self, cls: type[Any], contract: FeatureContract | FeatureFormat) -> None:
        if _contract_defined_on_class(cls, "contract"):
            msg = f"{cls.__name__!r} already defines a featurizer contract on the class body"
            raise ValueError(msg)
        _set_class_attribute(cls, "contract", normalize_feature_contract(contract))

    def _validate_role(self, cls: type[Any], name: str) -> None:
        if "contract" in cls.__dict__:
            return
        raise ValueError(format_validation_error(self._registry_id, name, missing=["contract"], invalid=[]))

    def _metadata_row(self, name: str, cls: type[Any]) -> dict[str, str]:
        return featurizer_component_metadata(self._display_name, name, cls)


class PredictorRegistry(Registry):
    """Registry for predictors that declare cell-line and drug input contracts."""

    def register(
        self,
        name: str,
        *,
        description: str,
        cell_line_contract: FeatureContract | FeatureFormat,
        drug_contract: FeatureContract | FeatureFormat,
        tags: Iterable[str] | None = None,
        reference: LiteratureReference | None = None,
    ) -> Callable[[type[Any]], type[Any]]:
        """Return a class decorator that registers a predictor under *name*.

        :param name: Registry name used in model configs and discovery listings.
        :param description: Short human-readable summary.
        :param cell_line_contract: Expected cell-line feature format.
        :param drug_contract: Expected drug feature format.
        :param tags: Optional discovery tags.
        :param reference: Optional literature citation metadata.

        :returns: Class decorator that registers and returns the decorated class.
        """

        def decorator(cls: type[Any]) -> type[Any]:
            with self._lock:
                if name in self._store:
                    msg = f"{self._label} {name!r} already registered"
                    raise ValueError(msg)
                self._apply_contracts(cls, cell_line_contract, drug_contract)
                _apply_shared_registration_metadata(
                    cls,
                    description=description,
                    tags=tags,
                    reference=reference,
                )
                validate_shared_registration_metadata(self._registry_id, name, cls)
                self._validate_role(cls, name)
                self._store[name] = cls
                _set_class_attribute(cls, "registry_name", name)
            return cls

        return decorator

    def _apply_contracts(
        self,
        cls: type[Any],
        cell_line_contract: FeatureContract | FeatureFormat,
        drug_contract: FeatureContract | FeatureFormat,
    ) -> None:
        if _contract_defined_on_class(cls, "cell_line_contract"):
            msg = f"{cls.__name__!r} already defines a cell-line contract on the class body"
            raise ValueError(msg)
        if _contract_defined_on_class(cls, "drug_contract"):
            msg = f"{cls.__name__!r} already defines a drug contract on the class body"
            raise ValueError(msg)
        _set_class_attribute(cls, "cell_line_contract", normalize_feature_contract(cell_line_contract))
        _set_class_attribute(cls, "drug_contract", normalize_feature_contract(drug_contract))

    def _validate_role(self, cls: type[Any], name: str) -> None:
        missing: list[str] = []
        if "cell_line_contract" not in cls.__dict__:
            missing.append("cell_line_contract")
        if "drug_contract" not in cls.__dict__:
            missing.append("drug_contract")
        if missing:
            raise ValueError(format_validation_error(self._registry_id, name, missing=missing, invalid=[]))

    def _metadata_row(self, name: str, cls: type[Any]) -> dict[str, str]:
        return predictor_component_metadata(self._display_name, name, cls)


cell_line_featurizer_registry = FeaturizerRegistry(
    "cell_line_featurizer",
    "Cell line featurizer",
    "cell_line_featurizers",
)
drug_featurizer_registry = FeaturizerRegistry(
    "drug_featurizer",
    "Drug featurizer",
    "drug_featurizers",
)
predictor_registry = PredictorRegistry(
    "predictor",
    "Predictor",
    "predictors",
)
