"""Public drug response prediction models and legacy experiment adapters."""

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
    from drevalpy.components.predictors.baselines import (
        AdaBoostDecisionTree,
        ElasticNetModel,
        GradientBoosting,
        KNNRegressor,
        LassoModel,
        MultiViewRandomForest,
        MultiViewXGBoost,
        NaiveCellLineMeanPredictor,
        NaiveDrugMeanPredictor,
        NaiveMeanEffectsPredictor,
        NaivePredictor,
        NaiveTissueDrugMeanPredictor,
        NaiveTissueMeanPredictor,
        RandomForest,
        SingleDrugElasticNet,
        SingleDrugRandomForest,
        SVMRegressor,
    )
    from drevalpy.components.predictors.baselines.zoo_preset import MultiViewLightGBM
    from .DIPK.dipk import DIPKModel
    from .DrugGNN import DrugGNN
    from .MOLIR.molir import MOLIR
    from .PharmaFormer.pharmaformer import PharmaFormerModel
    from .Precily import PrecilyModel
    from .SimpleNeuralNetwork.multi_view_neural_network import MultiViewNeuralNetwork
    from .SimpleNeuralNetwork.simple_neural_network import SimpleNeuralNetwork
    from .SparseGO.sparsego import SparseGOModel as SparseGO
    from .SRMF.srmf import SRMF
    from .SuperFELTR.superfeltr import SuperFELTR

    SINGLE_DRUG_MODEL_FACTORY: dict[str, type[DRPModel]]
    MULTI_DRUG_MODEL_FACTORY: dict[str, type[DRPModel]]
    MODEL_FACTORY: dict[str, type[DRPModel]]


def _lazy_load_public_models() -> None:
    global _LAZY_LOADED
    if _LAZY_LOADED:
        return

    from drevalpy.components.predictors.baselines import (
        AdaBoostDecisionTree,
        ElasticNetModel,
        GradientBoosting,
        KNNRegressor,
        LassoModel,
        MultiViewRandomForest,
        MultiViewXGBoost,
        NaiveCellLineMeanPredictor,
        NaiveDrugMeanPredictor,
        NaiveMeanEffectsPredictor,
        NaivePredictor,
        NaiveTissueDrugMeanPredictor,
        NaiveTissueMeanPredictor,
        RandomForest,
        SingleDrugElasticNet,
        SingleDrugRandomForest,
        SVMRegressor,
    )
    from drevalpy.components.predictors.baselines.zoo_preset import MultiViewLightGBM
    from .DIPK.dipk import DIPKModel
    from .DrugGNN import DrugGNN
    from .MOLIR.molir import MOLIR
    from .PharmaFormer.pharmaformer import PharmaFormerModel
    from .Precily import PrecilyModel
    from .SimpleNeuralNetwork.multi_view_neural_network import MultiViewNeuralNetwork
    from .SimpleNeuralNetwork.simple_neural_network import SimpleNeuralNetwork
    from .SparseGO.sparsego import SparseGOModel as SparseGO
    from .SRMF.srmf import SRMF
    from .SuperFELTR.superfeltr import SuperFELTR

    single: dict[str, type[DRPModel]] = {
        "SingleDrugElasticNet": SingleDrugElasticNet,
        "SingleDrugRandomForest": SingleDrugRandomForest,
        "MOLIR": MOLIR,
        "SuperFELTR": SuperFELTR,
    }
    multi: dict[str, type[DRPModel]] = {
        "NaivePredictor": NaivePredictor,
        "NaiveCellLineMeanPredictor": NaiveCellLineMeanPredictor,
        "NaiveDrugMeanPredictor": NaiveDrugMeanPredictor,
        "NaiveMeanEffectsPredictor": NaiveMeanEffectsPredictor,
        "NaiveTissueMeanPredictor": NaiveTissueMeanPredictor,
        "NaiveTissueDrugMeanPredictor": NaiveTissueDrugMeanPredictor,
        "AdaBoostDecisionTree": AdaBoostDecisionTree,
        "ElasticNet": ElasticNetModel,
        "Lasso": LassoModel,
        "GradientBoosting": GradientBoosting,
        "KNNRegressor": KNNRegressor,
        "RandomForest": RandomForest,
        "MultiViewRandomForest": MultiViewRandomForest,
        "SVR": SVMRegressor,
        "DrugGNN": DrugGNN,
        "SimpleNeuralNetwork": SimpleNeuralNetwork,
        "MultiViewNeuralNetwork": MultiViewNeuralNetwork,
        "MultiViewXGBoost": MultiViewXGBoost,
        "MultiViewLightGBM": MultiViewLightGBM,
        "DIPK": DIPKModel,
        "PharmaFormer": PharmaFormerModel,
        "SRMF": SRMF,
        "Precily": PrecilyModel,
        "SparseGO": SparseGO,
    }
    factory = multi.copy()
    factory.update(single)

    g = globals()
    g.update(
        {
            "AdaBoostDecisionTree": AdaBoostDecisionTree,
            "ElasticNetModel": ElasticNetModel,
            "GradientBoosting": GradientBoosting,
            "KNNRegressor": KNNRegressor,
            "LassoModel": LassoModel,
            "MultiViewRandomForest": MultiViewRandomForest,
            "MultiViewXGBoost": MultiViewXGBoost,
            "MultiViewLightGBM": MultiViewLightGBM,
            "NaiveCellLineMeanPredictor": NaiveCellLineMeanPredictor,
            "NaiveDrugMeanPredictor": NaiveDrugMeanPredictor,
            "NaiveMeanEffectsPredictor": NaiveMeanEffectsPredictor,
            "NaivePredictor": NaivePredictor,
            "NaiveTissueDrugMeanPredictor": NaiveTissueDrugMeanPredictor,
            "NaiveTissueMeanPredictor": NaiveTissueMeanPredictor,
            "RandomForest": RandomForest,
            "SingleDrugElasticNet": SingleDrugElasticNet,
            "SingleDrugRandomForest": SingleDrugRandomForest,
            "SVMRegressor": SVMRegressor,
            "DIPKModel": DIPKModel,
            "DrugGNN": DrugGNN,
            "MOLIR": MOLIR,
            "MultiViewNeuralNetwork": MultiViewNeuralNetwork,
            "PharmaFormerModel": PharmaFormerModel,
            "PrecilyModel": PrecilyModel,
            "SRMF": SRMF,
            "SimpleNeuralNetwork": SimpleNeuralNetwork,
            "SparseGOModel": SparseGO,
            "SuperFELTR": SuperFELTR,
            "SparseGO": SparseGO,
            "SINGLE_DRUG_MODEL_FACTORY": single,
            "MULTI_DRUG_MODEL_FACTORY": multi,
            "MODEL_FACTORY": factory,
        }
    )
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
