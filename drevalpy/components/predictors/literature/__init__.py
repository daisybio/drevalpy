"""Literature modular predictors (algorithm components only)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from drevalpy.components.predictors.literature.dipk_predictor import DIPKPredictor
    from drevalpy.components.predictors.literature.druggnn import DrugGNNPredictor
    from drevalpy.components.predictors.literature.molir_predictor import MOLIRPredictor
    from drevalpy.components.predictors.literature.neural_network import NeuralNetworkPredictor
    from drevalpy.components.predictors.literature.pharmaformer_predictor import PharmaFormerPredictor
    from drevalpy.components.predictors.literature.precily_predictor import PrecilyPredictor
    from drevalpy.components.predictors.literature.sparsego_predictor import SparseGOPredictor
    from drevalpy.components.predictors.literature.srmf_predictor import SRMFPredictor
    from drevalpy.components.predictors.literature.superfeltr_predictor import SuperFELTRPredictor

__all__ = [
    "DIPKPredictor",
    "DrugGNNPredictor",
    "MOLIRPredictor",
    "NeuralNetworkPredictor",
    "PharmaFormerPredictor",
    "PrecilyPredictor",
    "SRMFPredictor",
    "SparseGOPredictor",
    "SuperFELTRPredictor",
]

_LAZY_EXPORTS = {
    "DIPKPredictor": ("drevalpy.components.predictors.literature.dipk_predictor", "DIPKPredictor"),
    "DrugGNNPredictor": ("drevalpy.components.predictors.literature.druggnn", "DrugGNNPredictor"),
    "MOLIRPredictor": ("drevalpy.components.predictors.literature.molir_predictor", "MOLIRPredictor"),
    "NeuralNetworkPredictor": (
        "drevalpy.components.predictors.literature.neural_network",
        "NeuralNetworkPredictor",
    ),
    "PharmaFormerPredictor": (
        "drevalpy.components.predictors.literature.pharmaformer_predictor",
        "PharmaFormerPredictor",
    ),
    "PrecilyPredictor": ("drevalpy.components.predictors.literature.precily_predictor", "PrecilyPredictor"),
    "SRMFPredictor": ("drevalpy.components.predictors.literature.srmf_predictor", "SRMFPredictor"),
    "SparseGOPredictor": ("drevalpy.components.predictors.literature.sparsego_predictor", "SparseGOPredictor"),
    "SuperFELTRPredictor": (
        "drevalpy.components.predictors.literature.superfeltr_predictor",
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
