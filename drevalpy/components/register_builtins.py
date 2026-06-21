"""Register built-in featurizers and predictors."""

from __future__ import annotations

import importlib
import inspect
from types import ModuleType
from typing import Any

from drevalpy.components.predictors.literature import register_legacy_predictor
from drevalpy.components.registry.core import (
    cell_line_featurizer_registry,
    drug_featurizer_registry,
    predictor_registry,
)
from drevalpy.models.DIPK.dipk import DIPKModel
from drevalpy.models.DrugGNN.drug_gnn import DrugGNN
from drevalpy.models.MOLIR.molir import MOLIR
from drevalpy.models.PharmaFormer.pharmaformer import PharmaFormerModel
from drevalpy.models.Precily.precily import PrecilyModel
from drevalpy.models.SRMF.srmf import SRMF
from drevalpy.models.SimpleNeuralNetwork.multi_view_neural_network import MultiViewNeuralNetwork
from drevalpy.models.SimpleNeuralNetwork.simple_neural_network import SimpleNeuralNetwork
from drevalpy.models.SuperFELTR.superfeltr import SuperFELTR


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


def register_builtin_components() -> None:
    """Register built-in featurizers, predictors, and literature stacks."""
    for module_path, registry in (
        ("drevalpy.components.featurizers.cell_line.view", cell_line_featurizer_registry),
        ("drevalpy.components.featurizers.cell_line.multi_concat", cell_line_featurizer_registry),
        ("drevalpy.components.featurizers.drug.view", drug_featurizer_registry),
        ("drevalpy.components.predictors.naive", predictor_registry),
        ("drevalpy.components.predictors.sklearn_models", predictor_registry),
        ("drevalpy.components.predictors.xgboost_pred", predictor_registry),
    ):
        _restore_module_registrations(module_path, registry)

    legacy_predictors = {
        "dipk": (DIPKModel, "DIPK literature model stack."),
        "drugGNN": (DrugGNN, "DrugGNN literature model stack."),
        "molir": (MOLIR, "MOLIR single-drug literature model stack."),
        "superfeltr": (SuperFELTR, "SuperFELTR single-drug literature model stack."),
        "pharmaFormer": (PharmaFormerModel, "PharmaFormer literature model stack."),
        "precily": (PrecilyModel, "Precily literature model stack."),
        "srmf": (SRMF, "SRMF matrix-factorization model stack."),
        "simpleNeuralNetwork": (SimpleNeuralNetwork, "Simple neural network baseline stack."),
        "multiViewNeuralNetwork": (
            MultiViewNeuralNetwork,
            "Multi-view neural network baseline stack.",
        ),
    }
    for name, (model_cls, description) in legacy_predictors.items():
        if name not in predictor_registry.list_names():
            register_legacy_predictor(name, model_cls, description=description)


def ensure_components_registered() -> None:
    """Ensure built-in component registries are populated."""
    register_builtin_components()
