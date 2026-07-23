"""Lazily register built-in featurizers and predictors."""

from __future__ import annotations

import importlib
import inspect
from types import ModuleType

from drevalpy.components.registry.core import (
    cell_line_featurizer_registry,
    drug_featurizer_registry,
    predictor_registry,
)

_CELL_LINE_MODULES = {
    "scaledGeneExpression": "drevalpy.components.featurizers.cell_line.scaled_gene_expression",
    "normalizedProteomics": "drevalpy.components.featurizers.cell_line.normalized_proteomics",
    "pca": "drevalpy.components.featurizers.cell_line.pca",
    "raw": "drevalpy.components.featurizers.cell_line.raw",
    "concatFeaturizers": "drevalpy.components.featurizers.cell_line.concat",
    "landmarkGenes": "drevalpy.components.featurizers.cell_line.landmark",
    "landmarkGenesReduced": "drevalpy.components.featurizers.cell_line.landmark",
    "pathways": "drevalpy.components.featurizers.cell_line.pathways",
    "bionic": "drevalpy.components.featurizers.cell_line.bionic",
    "identity": "drevalpy.components.featurizers.cell_line.identity",
    "tissue": "drevalpy.components.featurizers.cell_line.tissue",
}

_DRUG_MODULES = {
    "concatFeaturizers": "drevalpy.components.featurizers.drug.concat",
    "view": "drevalpy.components.featurizers.drug.view",
    "fingerprints": "drevalpy.components.featurizers.drug.fingerprints",
    "drugGraph": "drevalpy.components.featurizers.drug.drug_graph",
    "molgnet": "drevalpy.components.featurizers.drug.molgnet",
    "bpePharmaformer": "drevalpy.components.featurizers.drug.bpe_pharmaformer",
    "smilesvec": "drevalpy.components.featurizers.drug.smilesvec",
    "identity": "drevalpy.components.featurizers.drug.identity",
}

_PREDICTOR_MODULES = {
    **{
        name: "drevalpy.components.predictors.naive"
        for name in (
            "naiveMean",
            "naiveDrugMean",
            "naiveCellLineMean",
            "naiveTissueMean",
            "naiveTissueDrugMean",
            "naiveMeanEffects",
        )
    },
    **{
        name: "drevalpy.components.predictors.sklearn_models"
        for name in (
            "elasticNet",
            "singleDrugElasticNet",
            "lasso",
            "ridge",
            "randomForest",
            "singleDrugRandomForest",
            "svr",
            "gradientBoosting",
            "adaboost",
            "knn",
        )
    },
    "xgboost": "drevalpy.components.predictors.xgboost_pred",
    "lightgbm": "drevalpy.components.predictors.lightgbm_pred",
    "neuralNetwork": "drevalpy.components.predictors.literature.neural_network",
    "drugGNN": "drevalpy.components.predictors.literature.druggnn",
    **{
        name: "drevalpy.components.predictors.literature.structured_predictors"
        for name in ("precily", "srmf", "molir", "superfeltr", "pharmaFormer", "dipk", "sparsego")
    },
}


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


def _register_named(name: str, modules: dict[str, str], registry) -> None:
    module_path = modules.get(name)
    if module_path is None:
        return
    _restore_module_registrations(module_path, registry)


def ensure_cell_line_featurizer_registered(name: str) -> None:
    """Import only the module that provides one cell-line featurizer."""
    _register_named(name, _CELL_LINE_MODULES, cell_line_featurizer_registry)


def ensure_drug_featurizer_registered(name: str) -> None:
    """Import only the module that provides one drug featurizer."""
    _register_named(name, _DRUG_MODULES, drug_featurizer_registry)


def ensure_predictor_registered(name: str) -> None:
    """Import only the module that provides one predictor."""
    _register_named(name, _PREDICTOR_MODULES, predictor_registry)


def register_native_components() -> None:
    """Register dependency-light native components."""
    modules = {
        *(_CELL_LINE_MODULES.values()),
        *(_DRUG_MODULES.values()),
        "drevalpy.components.predictors.naive",
        "drevalpy.components.predictors.sklearn_models",
    }
    for module_path in sorted(modules):
        if ".cell_line." in module_path:
            registry = cell_line_featurizer_registry
        elif ".featurizers.drug." in module_path:
            registry = drug_featurizer_registry
        else:
            registry = predictor_registry
        _restore_module_registrations(module_path, registry)


def register_optional_components() -> None:
    """Register predictors whose estimators require optional dependencies."""
    for name in ("xgboost", "lightgbm"):
        ensure_predictor_registered(name)


def register_literature_components() -> None:
    """Register literature predictors and their neural dependencies."""
    for name in ("neuralNetwork", "drugGNN", "precily"):
        ensure_predictor_registered(name)


def register_builtin_components(*, include_legacy: bool = False) -> None:
    """Register every built-in component for discovery and compatibility tests."""
    _ = include_legacy
    register_native_components()
    register_optional_components()
    register_literature_components()


def ensure_components_registered(*, include_legacy: bool = False) -> None:
    """Compatibility helper that registers every built-in component."""
    register_builtin_components(include_legacy=include_legacy)
