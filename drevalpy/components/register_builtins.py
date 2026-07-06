"""Register built-in featurizers and predictors."""

from __future__ import annotations

import importlib
import inspect
from types import ModuleType

from drevalpy.components.registry.core import (
    cell_line_featurizer_registry,
    drug_featurizer_registry,
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


def register_native_featurizers() -> None:
    """Register native cell-line and drug featurizers."""
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
        ("drevalpy.components.featurizers.cell_line.identity", cell_line_featurizer_registry),
        ("drevalpy.components.featurizers.cell_line.tissue", cell_line_featurizer_registry),
        ("drevalpy.components.featurizers.drug.concat", drug_featurizer_registry),
        ("drevalpy.components.featurizers.drug.view", drug_featurizer_registry),
        ("drevalpy.components.featurizers.drug.fingerprints", drug_featurizer_registry),
        ("drevalpy.components.featurizers.drug.one_hot", drug_featurizer_registry),
        ("drevalpy.components.featurizers.drug.drug_graph", drug_featurizer_registry),
        ("drevalpy.components.featurizers.drug.molgnet", drug_featurizer_registry),
        ("drevalpy.components.featurizers.drug.bpe_pharmaformer", drug_featurizer_registry),
        ("drevalpy.components.featurizers.drug.smilesvec", drug_featurizer_registry),
        ("drevalpy.components.featurizers.drug.identity", drug_featurizer_registry),
    ):
        _restore_module_registrations(module_path, registry)


def register_builtin_components(*, include_legacy: bool = False) -> None:
    """Register native featurizers."""
    _ = include_legacy
    register_native_featurizers()


def ensure_components_registered(*, include_legacy: bool = False) -> None:
    """Ensure built-in component registries are populated."""
    _ = include_legacy
    register_builtin_components()
