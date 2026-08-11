"""Featurizer registry class and module singletons."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, ClassVar

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat, normalize_feature_contract
from drevalpy.components.registry._registration_metadata import (
    apply_registration_metadata,
    normalize_registration_metadata,
)
from drevalpy.components.registry.base import Registry
from drevalpy.components.registry.metadata import featurizer_component_metadata
from drevalpy.types.enums.literature_reference import LiteratureReference


class FeaturizerRegistry(Registry):
    """Registry for featurizers that emit one feature contract."""

    _required_fields: ClassVar[tuple[str, ...]] = ("description", "contract")
    _side: str = ""

    def __init__(self, registry_id: str, label: str, display_name: str, *, side: str = "") -> None:
        """Initialize with an optional side designation.

        :param registry_id: Stable identifier.
        :param label: Human-readable label.
        :param display_name: Catalog name.
        :param side: Entity side ("cell_line" or "drug").
        """
        super().__init__(registry_id, label, display_name)
        self._side = side

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
        metadata = normalize_registration_metadata(description, tags, reference)
        normalized_contract = normalize_feature_contract(contract)

        def decorator(cls: type[Any]) -> type[Any]:
            with self._lock:
                if name in self._store:
                    msg = f"{self._label} {name!r} already registered"
                    raise ValueError(msg)
                self._apply_contract(cls, "contract", normalized_contract)
                apply_registration_metadata(cls, metadata)
                self._validate_registration(name, cls)
                self._store[name] = cls
                cls.registry_name = name
                if not getattr(cls, "storage_key", ""):
                    cls.storage_key = name
                if self._side:
                    cls.side = self._side
            return cls

        return decorator

    def _validate_registration(self, name: str, cls: type[Any]) -> None:
        """Enforce featurizer class invariants at registration time.

        :param name: Registry name under which *cls* is being registered.
        :param cls: Featurizer class with contract metadata already attached.
        """
        from drevalpy.components.contracts.hyperparameter_space import validate_component_hyperparameter_space
        from drevalpy.components.registry._featurizer_validate import validate_featurizer_input_views

        validate_component_hyperparameter_space(name, cls)
        validate_featurizer_input_views(self._registry_id, name, cls)

    def _component_metadata(self, name: str, cls: type[Any]) -> dict[str, Any]:
        return featurizer_component_metadata(self._display_name, name, cls)


cell_line_featurizer_registry = FeaturizerRegistry(
    "cell_line_featurizer",
    "Cell line featurizer",
    "cell_line_featurizers",
    side="cell_line",
)
drug_featurizer_registry = FeaturizerRegistry(
    "drug_featurizer",
    "Drug featurizer",
    "drug_featurizers",
    side="drug",
)
