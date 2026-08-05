"""Featurizer registry classes and module singletons."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, ClassVar

from drevalpy.components.contracts import FeatureContract, FeatureFormat, normalize_feature_contract
from drevalpy.components.registry._metadata_validate import (
    format_validation_error,
    validate_shared_registration_metadata,
)
from drevalpy.components.registry.base import (
    Registry,
    apply_shared_registration_metadata,
    contract_defined_on_class,
    set_class_attribute,
)
from drevalpy.components.registry.metadata import featurizer_component_metadata
from drevalpy.types.literature_reference import LiteratureReference


class FeaturizerRegistry(Registry):
    """Registry for featurizers that emit one feature contract."""

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
                apply_shared_registration_metadata(
                    cls,
                    description=description,
                    tags=tags,
                    reference=reference,
                )
                validate_shared_registration_metadata(self._registry_id, name, cls)
                self._validate_role(cls, name)
                self._store[name] = cls
                set_class_attribute(cls, "registry_name", name)
            return cls

        return decorator

    def _apply_contract(self, cls: type[Any], contract: FeatureContract | FeatureFormat) -> None:
        if contract_defined_on_class(cls, "contract"):
            msg = f"{cls.__name__!r} already defines a featurizer contract on the class body"
            raise ValueError(msg)
        set_class_attribute(cls, "contract", normalize_feature_contract(contract))

    def _validate_role(self, cls: type[Any], name: str) -> None:
        if "contract" in cls.__dict__:
            return
        raise ValueError(format_validation_error(self._registry_id, name, missing=["contract"], invalid=[]))

    def _metadata_row(self, name: str, cls: type[Any]) -> dict[str, str]:
        return featurizer_component_metadata(self._display_name, name, cls)


class CellLineFeaturizerRegistry(FeaturizerRegistry):
    """Registry for cell-line featurizers."""

    _registry_id: ClassVar[str] = "cell_line_featurizer"
    _label: ClassVar[str] = "Cell line featurizer"
    _display_name: ClassVar[str] = "cell_line_featurizers"


class DrugFeaturizerRegistry(FeaturizerRegistry):
    """Registry for drug featurizers."""

    _registry_id: ClassVar[str] = "drug_featurizer"
    _label: ClassVar[str] = "Drug featurizer"
    _display_name: ClassVar[str] = "drug_featurizers"


cell_line_featurizer_registry = CellLineFeaturizerRegistry()
drug_featurizer_registry = DrugFeaturizerRegistry()
