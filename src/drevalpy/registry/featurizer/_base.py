"""Featurizer registries: the shared class, one instance per entity side, and the lookup by side."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.contracts.hyperparameter_space import validate_component_hyperparameter_space
from drevalpy.errors import UnknownNameError
from drevalpy.registry.components._base import ComponentRegistry
from drevalpy.registry.components._format_assignment import assign_format
from drevalpy.registry.components._metadata import LiteratureReference, featurizer_component_metadata
from drevalpy.registry.components._registration_metadata import (
    apply_registration_metadata,
    normalize_registration_metadata,
)
from drevalpy.registry.featurizer._validate import validate_featurizer_input_views

if TYPE_CHECKING:
    from drevalpy.components.featurizers.base import Featurizer

#: The entity side a featurizer is registered for.
Side = Literal["sample", "drug"]


class FeaturizerRegistry(ComponentRegistry):
    """Registry for featurizers that emit one feature format."""

    _required_fields: ClassVar[tuple[str, ...]] = ("description", "output_format")
    _side: str = ""

    def __init__(self, registry_id: str, label: str, display_name: str, *, side: str = "") -> None:
        """Initialize with an optional side designation.

        :param registry_id: Stable identifier used in validation messages.
        :param label: Human-readable label for unknown/duplicate errors.
        :param display_name: Catalog registry name written into component metadata.
        :param side: Entity side ("sample" or "drug").
        """
        super().__init__(registry_id, label, display_name)
        self._side = side

    def register(
        self,
        name: str,
        *,
        description: str,
        output_format: FeatureFormat | None = None,
        tags: Iterable[str] | None = None,
        reference: LiteratureReference | None = None,
    ) -> Callable[[type[Any]], type[Any]]:
        """Return a class decorator that registers a featurizer under *name*.

        :param name: Registry name used in model configs and discovery listings.
        :param description: Short human-readable summary.
        :param output_format: Feature format for predictor matching. When
            omitted, the ``output_format`` declared on the class body is used.
        :param tags: Optional discovery tags.
        :param reference: Optional literature citation metadata.
        :returns: Class decorator that registers and returns the decorated class.
        """
        metadata = normalize_registration_metadata(description, tags, reference)

        def decorator(cls: type[Any]) -> type[Any]:
            with self._lock:
                if name in self._store:
                    msg = f"{self._label} {name!r} already registered"
                    raise ValueError(msg)
                assign_format(cls, "output_format", output_format)
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
        :param cls: Featurizer class with its output format already attached.
        """
        super()._validate_registration(name, cls)
        validate_component_hyperparameter_space(name, cls)
        validate_featurizer_input_views(self._registry_id, name, cls)

    def _component_metadata(self, name: str, cls: type[Any]) -> dict[str, Any]:
        return featurizer_component_metadata(self._display_name, name, cls)


class SampleFeaturizerRegistry(FeaturizerRegistry):
    """Registry for sample featurizers."""

    def __init__(self) -> None:
        """Initialize with the fixed sample identity."""
        super().__init__("sample_featurizer", "Sample featurizer", "sample_featurizers", side="sample")


class DrugFeaturizerRegistry(FeaturizerRegistry):
    """Registry for drug featurizers."""

    def __init__(self) -> None:
        """Initialize with the fixed drug identity."""
        super().__init__("drug_featurizer", "Drug featurizer", "drug_featurizers", side="drug")


sample_featurizer_registry = SampleFeaturizerRegistry()
drug_featurizer_registry = DrugFeaturizerRegistry()

_REGISTRIES: dict[str, FeaturizerRegistry] = {
    "sample": sample_featurizer_registry,
    "drug": drug_featurizer_registry,
}


def get_featurizer(side: Side, name: str) -> type[Featurizer]:
    """Return the featurizer class registered under *name* for one entity side.

    :param side: ``"sample"`` or ``"drug"``.
    :param name: Registry name of the featurizer.
    :returns: Featurizer class registered under *name* on that side.
    :raises UnknownNameError: If *side* is not ``"sample"`` or ``"drug"``, or *name* is not registered on it.
    """
    registry = _REGISTRIES.get(side)
    if registry is None:
        raise UnknownNameError(f"Unknown featurizer side {side!r}. Available: {list(_REGISTRIES)}")
    return registry.get(name)
