"""Literature modular predictors (algorithm components only)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from drevalpy.components.predictors.literature.druggnn import DrugGNNPredictor
    from drevalpy.components.predictors.literature.neural_network import NeuralNetworkPredictor
    from drevalpy.components.predictors.literature.structured_predictors import (
        DIPKPredictor,
        MOLIRPredictor,
        PharmaFormerPredictor,
        PrecilyPredictor,
        SparseGOPredictor,
        SRMFPredictor,
        SuperFELTRPredictor,
    )

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
    "DIPKPredictor": ("drevalpy.components.predictors.literature.structured_predictors", "DIPKPredictor"),
    "DrugGNNPredictor": ("drevalpy.components.predictors.literature.druggnn", "DrugGNNPredictor"),
    "MOLIRPredictor": ("drevalpy.components.predictors.literature.structured_predictors", "MOLIRPredictor"),
    "NeuralNetworkPredictor": (
        "drevalpy.components.predictors.literature.neural_network",
        "NeuralNetworkPredictor",
    ),
    "PharmaFormerPredictor": (
        "drevalpy.components.predictors.literature.structured_predictors",
        "PharmaFormerPredictor",
    ),
    "PrecilyPredictor": ("drevalpy.components.predictors.literature.structured_predictors", "PrecilyPredictor"),
    "SRMFPredictor": ("drevalpy.components.predictors.literature.structured_predictors", "SRMFPredictor"),
    "SparseGOPredictor": ("drevalpy.components.predictors.literature.structured_predictors", "SparseGOPredictor"),
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
