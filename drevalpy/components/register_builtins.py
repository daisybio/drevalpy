"""Register built-in featurizers and predictors."""

from __future__ import annotations

import importlib
import inspect
from types import ModuleType

from drevalpy.components.registry.core import (
    cell_line_featurizer_registry,
    drug_featurizer_registry,
    predictor_registry,
)


def _restore_registry_from_module(registry, module: ModuleType) -> None:
    for value in vars(module).values():
        if not inspect.isclass(value):
            continue
        registry_name = getattr(value, "registry_name", None)
        if registry_name:
            registry.register_existing(registry_name, value)


def _restore_module_registrations(module_path: str, registry) -> None:
    module = importlib.import_module(module_path)
    _restore_registry_from_module(registry, module)


def register_native_components() -> None:
    """Register native cell-line featurizers, drug featurizers, and tabular predictors."""
    for module_path, registry in (
        ("drevalpy.components.featurizers.cell_line.view", cell_line_featurizer_registry),
        ("drevalpy.components.featurizers.cell_line.multi_concat", cell_line_featurizer_registry),
        ("drevalpy.components.featurizers.drug.view", drug_featurizer_registry),
        ("drevalpy.components.predictors.naive", predictor_registry),
        ("drevalpy.components.predictors.sklearn_models", predictor_registry),
        ("drevalpy.components.predictors.xgboost_pred", predictor_registry),
    ):
        _restore_module_registrations(module_path, registry)


def register_builtin_components(*, include_legacy: bool = True) -> None:
    """Register native components and optionally legacy DRPModel predictor wrappers."""
    register_native_components()
    if include_legacy:
        from drevalpy.models.component_registration import register_legacy_component_predictors

        register_legacy_component_predictors()


def ensure_components_registered(*, include_legacy: bool = True) -> None:
    """Ensure built-in component registries are populated."""
    register_builtin_components(include_legacy=include_legacy)
