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
        ("drevalpy.components.featurizers.cell_line.omics.gene_expression", cell_line_featurizer_registry),
        ("drevalpy.components.featurizers.cell_line.omics.scaled_gene_expression", cell_line_featurizer_registry),
        ("drevalpy.components.featurizers.cell_line.omics.methylation", cell_line_featurizer_registry),
        ("drevalpy.components.featurizers.cell_line.omics.mutations", cell_line_featurizer_registry),
        ("drevalpy.components.featurizers.cell_line.omics.copy_number_variation", cell_line_featurizer_registry),
        ("drevalpy.components.featurizers.cell_line.pca", cell_line_featurizer_registry),
        ("drevalpy.components.featurizers.cell_line.omics.proteomics", cell_line_featurizer_registry),
        ("drevalpy.components.featurizers.cell_line.concat", cell_line_featurizer_registry),
        ("drevalpy.components.featurizers.cell_line.landmark", cell_line_featurizer_registry),
        ("drevalpy.components.featurizers.cell_line.pathways", cell_line_featurizer_registry),
        ("drevalpy.components.featurizers.cell_line.bionic", cell_line_featurizer_registry),
        ("drevalpy.components.featurizers.drug.view", drug_featurizer_registry),
        ("drevalpy.components.featurizers.drug.fingerprints", drug_featurizer_registry),
        ("drevalpy.components.featurizers.drug.one_hot", drug_featurizer_registry),
        ("drevalpy.components.featurizers.drug.drug_graph", drug_featurizer_registry),
        ("drevalpy.components.featurizers.drug.molgnet", drug_featurizer_registry),
        ("drevalpy.components.featurizers.drug.bpe_pharmaformer", drug_featurizer_registry),
        ("drevalpy.components.featurizers.drug.smilesvec", drug_featurizer_registry),
        ("drevalpy.components.predictors.naive", predictor_registry),
        ("drevalpy.components.predictors.sklearn_models", predictor_registry),
        ("drevalpy.components.predictors.xgboost_pred", predictor_registry),
        ("drevalpy.components.predictors.literature.neural_network", predictor_registry),
    ):
        _restore_module_registrations(module_path, registry)


def register_literature_components() -> None:
    """Register literature predictors that depend on drevalpy.models DRPModel stacks."""
    for module_path in (
        "drevalpy.components.predictors.literature.druggnn",
        "drevalpy.components.predictors.literature.structured_predictors",
    ):
        _restore_module_registrations(module_path, predictor_registry)


def register_builtin_components(*, include_legacy: bool = False) -> None:
    """Register native components and optional legacy DRPModel predictor wrappers."""
    register_native_components()
    register_literature_components()


def ensure_components_registered(*, include_legacy: bool = False) -> None:
    """Ensure built-in component registries are populated."""
    _ = include_legacy
    register_builtin_components()
