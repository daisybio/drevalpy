"""Predictor registry class and module singleton."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, ClassVar

from drevalpy.components.core.contracts.contracts import FeatureContract, FeatureFormat, normalize_feature_contract
from drevalpy.components.registry._registration_metadata import (
    apply_registration_metadata,
    normalize_registration_metadata,
)
from drevalpy.components.registry.base import Registry
from drevalpy.components.registry.metadata import predictor_component_metadata
from drevalpy.types.literature_reference import LiteratureReference


class PredictorRegistry(Registry):
    """Registry for predictors that declare cell-line and drug input contracts."""

    _required_fields: ClassVar[tuple[str, ...]] = (
        "description",
        "cell_line_contract",
        "drug_contract",
    )

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
        metadata = normalize_registration_metadata(description, tags, reference)
        normalized_cell_line_contract = normalize_feature_contract(cell_line_contract)
        normalized_drug_contract = normalize_feature_contract(drug_contract)

        def decorator(cls: type[Any]) -> type[Any]:
            with self._lock:
                if name in self._store:
                    msg = f"{self._label} {name!r} already registered"
                    raise ValueError(msg)
                self._apply_contract(cls, "cell_line_contract", normalized_cell_line_contract)
                self._apply_contract(cls, "drug_contract", normalized_drug_contract)
                apply_registration_metadata(cls, metadata)
                self._validate_registration(name, cls)
                self._store[name] = cls
                cls.registry_name = name
            return cls

        return decorator

    def _validate_registration(self, name: str, cls: type[Any]) -> None:
        """Enforce predictor leaf-interface and capability invariants.

        :param name: Registry name under which *cls* is being registered.
        :param cls: Predictor class with contracts already attached.
        """
        from drevalpy.components.registry._predictor_validate import validate_predictor_registration

        validate_predictor_registration(name, cls)

    def _component_metadata(self, name: str, cls: type[Any]) -> dict[str, Any]:
        return predictor_component_metadata(self._display_name, name, cls)


predictor_registry = PredictorRegistry()
