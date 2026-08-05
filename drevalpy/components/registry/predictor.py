"""Predictor registry class and module singleton."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

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
from drevalpy.components.registry.metadata import predictor_component_metadata
from drevalpy.types.literature_reference import LiteratureReference


class PredictorRegistry(Registry):
    """Registry for predictors that declare cell-line and drug input contracts."""

    def __init__(self) -> None:
        """Initialize the predictor registry with its fixed identity."""
        super().__init__("predictor", "Predictor", "predictors")

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

    def _apply_contracts(
        self,
        cls: type[Any],
        cell_line_contract: FeatureContract | FeatureFormat,
        drug_contract: FeatureContract | FeatureFormat,
    ) -> None:
        if contract_defined_on_class(cls, "cell_line_contract"):
            msg = f"{cls.__name__!r} already defines a cell-line contract on the class body"
            raise ValueError(msg)
        if contract_defined_on_class(cls, "drug_contract"):
            msg = f"{cls.__name__!r} already defines a drug contract on the class body"
            raise ValueError(msg)
        set_class_attribute(cls, "cell_line_contract", normalize_feature_contract(cell_line_contract))
        set_class_attribute(cls, "drug_contract", normalize_feature_contract(drug_contract))

    def _validate_role(self, cls: type[Any], name: str) -> None:
        missing: list[str] = []
        if "cell_line_contract" not in cls.__dict__:
            missing.append("cell_line_contract")
        if "drug_contract" not in cls.__dict__:
            missing.append("drug_contract")
        if missing:
            raise ValueError(format_validation_error(self._registry_id, name, missing=missing, invalid=[]))

    def _metadata_row(self, name: str, cls: type[Any]) -> dict[str, Any]:
        return predictor_component_metadata(self._display_name, name, cls)


predictor_registry = PredictorRegistry()
