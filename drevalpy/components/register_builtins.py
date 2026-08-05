"""Lazily register built-in featurizers and predictors."""

from __future__ import annotations

import importlib
import inspect
from types import ModuleType

from drevalpy.components.registry.featurizer import (
    cell_line_featurizer_registry,
    drug_featurizer_registry,
)
from drevalpy.components.registry.predictor import predictor_registry

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
    "dipkGeneExpression": "drevalpy.components.featurizers.cell_line.dipk_gene_expression",
    "pharmaFormerGeneExpression": "drevalpy.components.featurizers.cell_line.pharmaformer_gene_expression",
    "sparsegoOntology": "drevalpy.components.featurizers.cell_line.sparsego_ontology",
    "molirOmics": "drevalpy.components.featurizers.cell_line.molir_omics",
    "superfeltrOmics": "drevalpy.components.featurizers.cell_line.superfeltr_omics",
    "identity": "drevalpy.components.featurizers.cell_line.identity",
    "constant": "drevalpy.components.featurizers.cell_line.constant",
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
    "constant": "drevalpy.components.featurizers.drug.constant",
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
    "neuralNetwork": "drevalpy.components.predictors.neural_network.predictor",
    "drugGNN": "drevalpy.components.predictors.literature.druggnn.predictor",
    "precily": "drevalpy.components.predictors.literature.precily.predictor",
    "srmf": "drevalpy.components.predictors.literature.srmf.predictor",
    "molir": "drevalpy.components.predictors.literature.molir.predictor",
    "superfeltr": "drevalpy.components.predictors.literature.superfeltr.predictor",
    "pharmaFormer": "drevalpy.components.predictors.literature.pharmaformer.predictor",
    "dipk": "drevalpy.components.predictors.literature.dipk.predictor",
    "sparsego": "drevalpy.components.predictors.literature.sparsego.predictor",
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
    """Import only the module that provides one cell-line featurizer.

    :param name: Built-in registry name to lazy-load.
    """
    _register_named(name, _CELL_LINE_MODULES, cell_line_featurizer_registry)


def ensure_drug_featurizer_registered(name: str) -> None:
    """Import only the module that provides one drug featurizer.

    :param name: Built-in registry name to lazy-load.
    """
    _register_named(name, _DRUG_MODULES, drug_featurizer_registry)


def ensure_predictor_registered(name: str) -> None:
    """Import only the module that provides one predictor.

    :param name: Built-in registry name to lazy-load.
    """
    _register_named(name, _PREDICTOR_MODULES, predictor_registry)


def is_known_builtin_cell_line_featurizer(name: str) -> bool:
    """Return whether *name* maps to a built-in cell-line featurizer module.

    :param name: Registry name to check.

    :returns: ``True`` when *name* is listed in the built-in catalog.
    """
    return name in _CELL_LINE_MODULES


def is_known_builtin_drug_featurizer(name: str) -> bool:
    """Return whether *name* maps to a built-in drug featurizer module.

    :param name: Registry name to check.

    :returns: ``True`` when *name* is listed in the built-in catalog.
    """
    return name in _DRUG_MODULES


def is_known_builtin_predictor(name: str) -> bool:
    """Return whether *name* maps to a built-in predictor module.

    :param name: Registry name to check.

    :returns: ``True`` when *name* is listed in the built-in catalog.
    """
    return name in _PREDICTOR_MODULES


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
    for name in (
        "neuralNetwork",
        "drugGNN",
        "precily",
        "srmf",
        "molir",
        "superfeltr",
        "pharmaFormer",
        "dipk",
        "sparsego",
    ):
        ensure_predictor_registered(name)


def register_builtin_components() -> None:
    """Register every built-in component for discovery and compatibility tests."""
    register_native_components()
    register_optional_components()
    register_literature_components()


def ensure_components_registered() -> None:
    """Compatibility helper that registers every built-in component."""
    register_builtin_components()
