"""Predictor registry class and module singleton."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, ClassVar

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.registry.components._base import ComponentRegistry
from drevalpy.registry.components._format_assignment import assign_format
from drevalpy.registry.components._metadata import LiteratureReference, predictor_component_metadata
from drevalpy.registry.components._registration_metadata import (
    apply_registration_metadata,
    normalize_registration_metadata,
)
from drevalpy.registry.predictor._validate import validate_predictor_registration


class PredictorRegistry(ComponentRegistry):
    """Registry for predictors that declare sample and drug input formats."""

    _required_fields: ClassVar[tuple[str, ...]] = (
        "description",
        "sample_input_format",
        "drug_input_format",
    )

    def __init__(self) -> None:
        """Initialize the predictor registry with its fixed identity."""
        super().__init__("predictor", "Predictor", "predictors")

    def register(
        self,
        name: str,
        *,
        description: str,
        sample_input_format: FeatureFormat | None = None,
        drug_input_format: FeatureFormat | None = None,
        tags: Iterable[str] | None = None,
        reference: LiteratureReference | None = None,
    ) -> Callable[[type[Any]], type[Any]]:
        """Return a class decorator that registers a predictor under *name*.

        :param name: Registry name used in model configs and discovery listings.
        :param description: Short human-readable summary.
        :param sample_input_format: Expected sample feature format. When omitted,
            the ``sample_input_format`` declared on the class body is used.
        :param drug_input_format: Expected drug feature format. When omitted, the
            ``drug_input_format`` declared on the class body is used.
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
                assign_format(cls, "sample_input_format", sample_input_format)
                assign_format(cls, "drug_input_format", drug_input_format)
                apply_registration_metadata(cls, metadata)
                self._validate_registration(name, cls)
                self._store[name] = cls
                cls.registry_name = name
            return cls

        return decorator

    def _validate_registration(self, name: str, cls: type[Any]) -> None:
        """Enforce predictor leaf-interface and capability invariants.

        :param name: Registry name under which *cls* is being registered.
        :param cls: Predictor class with its input formats already attached.
        """
        super()._validate_registration(name, cls)
        validate_predictor_registration(name, cls)

    def _component_metadata(self, name: str, cls: type[Any]) -> dict[str, Any]:
        return predictor_component_metadata(self._display_name, name, cls)


predictor_registry = PredictorRegistry()
