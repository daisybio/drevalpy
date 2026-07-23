"""Thread-safe registries for cell-line featurizers, drug featurizers, and predictors."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from drevalpy.components.contracts import FeatureContract, FeatureKind
from drevalpy.components.registry._metadata_validate import validate_registered_class_metadata
from drevalpy.components.registry.common import make_registration_decorator
from drevalpy.components.registry.metadata import (
    featurizer_component_metadata,
    predictor_component_metadata,
)


class Registry:
    """Thread-safe name-to-class registry with metadata validation."""

    def __init__(
        self,
        registry_id: str,
        label: str,
        display_name: str,
        metadata_fn: Callable[[str, str, type[Any]], dict[str, str]],
    ) -> None:
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
        category: str,
        template_repo_url: str = "",
        citation: str = "",
        citation_doi: str = "",
        citation_text: str = "",
        deviations: str = "",
        contract: FeatureContract | FeatureKind | None = None,
        cell_line_contract: FeatureContract | FeatureKind | None = None,
        drug_contract: FeatureContract | FeatureKind | None = None,
    ) -> Callable[[type[Any]], type[Any]]:
        """Return a class decorator that registers the decorated class under *name*."""
        return make_registration_decorator(
            self._store,
            self._lock,
            self._registry_id,
            name,
            description=description,
            category=category,
            template_repo_url=template_repo_url,
            citation=citation,
            citation_doi=citation_doi,
            citation_text=citation_text,
            deviations=deviations,
            contract=contract,
            cell_line_contract=cell_line_contract,
            drug_contract=drug_contract,
            already_registered_label=self._label,
        )

    def get(self, name: str) -> type[Any]:
        """Return the class registered under *name*, or raise ``ValueError``."""
        with self._lock:
            if name not in self._store:
                available = list(self._store.keys())
                msg = f"Unknown {self._label}: {name!r}. Available: {available}"
                raise ValueError(msg)
            return self._store[name]

    def list_names(self) -> list[str]:
        """Return all registered names."""
        with self._lock:
            return list(self._store.keys())

    def get_metadata(self, name: str) -> dict[str, str]:
        """Return the metadata record for the component registered under *name*."""
        cls = self.get(name)
        return self._metadata_fn(self._display_name, name, cls)

    def list_metadata(self, category: str | None = None) -> list[dict[str, str]]:
        """Return metadata for all components, optionally filtered by taxonomy category."""
        rows = [self.get_metadata(name) for name in self.list_names()]
        if category is None:
            return rows
        return [row for row in rows if row["category"] == category]

    def clear(self) -> None:
        """Remove all entries (primarily for testing)."""
        with self._lock:
            self._store.clear()

    def retain_only(self, names: frozenset[str]) -> None:
        """Drop entries whose names are not in *names*."""
        with self._lock:
            for registered_name in list(self._store):
                if registered_name not in names:
                    del self._store[registered_name]

    def register_existing(self, name: str, cls: type[Any]) -> None:
        """Register a class that was previously decorated but removed via `clear`."""
        with self._lock:
            if name in self._store:
                return
            validate_registered_class_metadata(self._registry_id, name, cls)
            self._store[name] = cls
            cls.registry_name = name  # type: ignore[attr-defined]


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
