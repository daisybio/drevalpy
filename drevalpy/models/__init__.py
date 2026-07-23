"""Public drug response prediction models.

Root exports (`MODEL_FACTORY`, named facade classes, `construct_model`) are
generated from zoo presets and backed by a single `NativeDRPModel` facade.
Component registration plus `ModelConfig` is the supported extension path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .drp_model import DRPModel

__all__ = [
    "DRPModel",
    "construct_model",
    "MULTI_DRUG_MODEL_FACTORY",
    "SINGLE_DRUG_MODEL_FACTORY",
    "MODEL_FACTORY",
    "NaivePredictor",
    "NaiveDrugMeanPredictor",
    "NaiveCellLineMeanPredictor",
    "NaiveTissueMeanPredictor",
    "NaiveTissueDrugMeanPredictor",
    "NaiveMeanEffectsPredictor",
    "ElasticNetModel",
    "RandomForest",
    "SVMRegressor",
    "SimpleNeuralNetwork",
    "MultiViewNeuralNetwork",
    "MultiViewRandomForest",
    "SingleDrugRandomForest",
    "SingleDrugElasticNet",
    "SRMF",
    "GradientBoosting",
    "MOLIR",
    "SuperFELTR",
    "DIPKModel",
    "DrugGNN",
    "PharmaFormerModel",
    "PrecilyModel",
    "KNNRegressor",
    "AdaBoostDecisionTree",
    "LassoModel",
    "MultiViewXGBoost",
    "MultiViewLightGBM",
    "SparseGO",
]

_LAZY_LOADED = False

if TYPE_CHECKING:
    from drevalpy.models._native_drp_model import NativeDRPModel

    AdaBoostDecisionTree: type[NativeDRPModel]
    ElasticNetModel: type[NativeDRPModel]
    GradientBoosting: type[NativeDRPModel]
    KNNRegressor: type[NativeDRPModel]
    LassoModel: type[NativeDRPModel]
    MultiViewRandomForest: type[NativeDRPModel]
    MultiViewXGBoost: type[NativeDRPModel]
    MultiViewLightGBM: type[NativeDRPModel]
    NaiveCellLineMeanPredictor: type[NativeDRPModel]
    NaiveDrugMeanPredictor: type[NativeDRPModel]
    NaiveMeanEffectsPredictor: type[NativeDRPModel]
    NaivePredictor: type[NativeDRPModel]
    NaiveTissueDrugMeanPredictor: type[NativeDRPModel]
    NaiveTissueMeanPredictor: type[NativeDRPModel]
    RandomForest: type[NativeDRPModel]
    SingleDrugElasticNet: type[NativeDRPModel]
    SingleDrugRandomForest: type[NativeDRPModel]
    SVMRegressor: type[NativeDRPModel]
    DIPKModel: type[NativeDRPModel]
    DrugGNN: type[NativeDRPModel]
    MOLIR: type[NativeDRPModel]
    MultiViewNeuralNetwork: type[NativeDRPModel]
    PharmaFormerModel: type[NativeDRPModel]
    PrecilyModel: type[NativeDRPModel]
    SRMF: type[NativeDRPModel]
    SimpleNeuralNetwork: type[NativeDRPModel]
    SparseGO: type[NativeDRPModel]
    SuperFELTR: type[NativeDRPModel]

    SINGLE_DRUG_MODEL_FACTORY: dict[str, type[DRPModel]]
    MULTI_DRUG_MODEL_FACTORY: dict[str, type[DRPModel]]
    MODEL_FACTORY: dict[str, type[DRPModel]]


def _lazy_load_public_models() -> None:
    global _LAZY_LOADED
    if _LAZY_LOADED:
        return
    from drevalpy.models._factory_classes import populate_public_model_namespace

    populate_public_model_namespace(globals())
    _LAZY_LOADED = True


def __getattr__(name: str) -> Any:
    if name == "construct_model":
        from ._construct_model_api import construct_model

        return construct_model
    if name in __all__ and name != "DRPModel":
        _lazy_load_public_models()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
