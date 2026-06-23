"""Literature modular predictors."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from drevalpy.components.predictors.literature.druggnn import DrugGNNPredictor
    from drevalpy.components.predictors.literature.neural_network import NeuralNetworkPredictor
    from drevalpy.components.predictors.literature.public_models import (
        DIPKModel,
        DrugGNN,
        MOLIR,
        MultiViewNeuralNetwork,
        PharmaFormerModel,
        PrecilyModel,
        SRMF,
        SimpleNeuralNetwork,
        SuperFELTR,
    )
    from drevalpy.components.predictors.literature.structured_predictors import (
        DIPKPredictor,
        MOLIRPredictor,
        PharmaFormerPredictor,
        PrecilyPredictor,
        SRMFPredictor,
        SuperFELTRPredictor,
    )

__all__ = [
    "DIPKModel",
    "DIPKPredictor",
    "DrugGNN",
    "DrugGNNPredictor",
    "MOLIR",
    "MOLIRPredictor",
    "MultiViewNeuralNetwork",
    "NeuralNetworkPredictor",
    "PharmaFormerModel",
    "PharmaFormerPredictor",
    "PrecilyModel",
    "PrecilyPredictor",
    "SRMF",
    "SRMFPredictor",
    "SimpleNeuralNetwork",
    "SuperFELTR",
    "SuperFELTRPredictor",
]

_LAZY_EXPORTS = {
    "DIPKModel": ("drevalpy.components.predictors.literature.public_models", "DIPKModel"),
    "DIPKPredictor": ("drevalpy.components.predictors.literature.structured_predictors", "DIPKPredictor"),
    "DrugGNN": ("drevalpy.components.predictors.literature.public_models", "DrugGNN"),
    "DrugGNNPredictor": ("drevalpy.components.predictors.literature.druggnn", "DrugGNNPredictor"),
    "MOLIR": ("drevalpy.components.predictors.literature.public_models", "MOLIR"),
    "MOLIRPredictor": ("drevalpy.components.predictors.literature.structured_predictors", "MOLIRPredictor"),
    "MultiViewNeuralNetwork": (
        "drevalpy.components.predictors.literature.public_models",
        "MultiViewNeuralNetwork",
    ),
    "NeuralNetworkPredictor": (
        "drevalpy.components.predictors.literature.neural_network",
        "NeuralNetworkPredictor",
    ),
    "PharmaFormerModel": ("drevalpy.components.predictors.literature.public_models", "PharmaFormerModel"),
    "PharmaFormerPredictor": (
        "drevalpy.components.predictors.literature.structured_predictors",
        "PharmaFormerPredictor",
    ),
    "PrecilyModel": ("drevalpy.components.predictors.literature.public_models", "PrecilyModel"),
    "PrecilyPredictor": ("drevalpy.components.predictors.literature.structured_predictors", "PrecilyPredictor"),
    "SRMF": ("drevalpy.components.predictors.literature.public_models", "SRMF"),
    "SRMFPredictor": ("drevalpy.components.predictors.literature.structured_predictors", "SRMFPredictor"),
    "SimpleNeuralNetwork": (
        "drevalpy.components.predictors.literature.public_models",
        "SimpleNeuralNetwork",
    ),
    "SuperFELTR": ("drevalpy.components.predictors.literature.public_models", "SuperFELTR"),
    "SuperFELTRPredictor": (
        "drevalpy.components.predictors.literature.structured_predictors",
        "SuperFELTRPredictor",
    ),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    module_path, attr = _LAZY_EXPORTS[name]
    import importlib

    module = importlib.import_module(module_path)
    value = getattr(module, attr)
    globals()[name] = value
    return value
