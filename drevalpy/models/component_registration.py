"""Register literature and neural DRPModel stacks as component predictors."""

from __future__ import annotations

from drevalpy.components.predictors.literature import register_legacy_predictor
from drevalpy.components.registry.core import predictor_registry
from drevalpy.models.DIPK.dipk import DIPKModel
from drevalpy.models.DrugGNN.drug_gnn import DrugGNN
from drevalpy.models.MOLIR.molir import MOLIR
from drevalpy.models.PharmaFormer.pharmaformer import PharmaFormerModel
from drevalpy.models.Precily.precily import PrecilyModel
from drevalpy.models.SRMF.srmf import SRMF
from drevalpy.models.SimpleNeuralNetwork.multi_view_neural_network import MultiViewNeuralNetwork
from drevalpy.models.SimpleNeuralNetwork.simple_neural_network import SimpleNeuralNetwork
from drevalpy.models.SuperFELTR.superfeltr import SuperFELTR

_LEGACY_PREDICTORS: dict[str, tuple[type, str]] = {
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


def register_legacy_component_predictors() -> None:
    """Register monolithic DRPModel stacks as legacy component predictors."""
    for name, (model_cls, description) in _LEGACY_PREDICTORS.items():
        if name not in predictor_registry.list_names():
            register_legacy_predictor(name, model_cls, description=description)


def ensure_legacy_components_registered() -> None:
    """Alias for :func:`register_legacy_component_predictors`."""
    register_legacy_component_predictors()
