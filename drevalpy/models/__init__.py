"""Public drug response prediction models and legacy experiment adapters.

This package exposes `~drevalpy.models.drp_model.DRPModel` subclasses,
``MODEL_FACTORY``, model orchestration (factory, config IO/spec, zoo,
`~drevalpy.models.composed_model.ComposedModel`), and public model
compatibility classes. Baseline and literature predictor implementations live
under `drevalpy.components.predictors` and are exposed here for backward
compatibility via `drevalpy.models._component_bridge`.

Factory tables and concrete model classes are loaded lazily so importing
``drevalpy.models.drp_model`` from component implementations does not pull in
the full public model graph during package initialization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .construct_model import construct_model
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
    from drevalpy.components.predictors.literature.public_models import (
        MOLIR,
        SRMF,
        DIPKModel,
        DrugGNN,
        MultiViewNeuralNetwork,
        PharmaFormerModel,
        PrecilyModel,
        SimpleNeuralNetwork,
        SuperFELTR,
    )
    from .baselines.multi_view_lightgbm import MultiViewLightGBM
    from .SparseGO.sparsego import SparseGOModel

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
    from drevalpy.components.predictors.literature.public_models import (
        MOLIR,
        SRMF,
        DIPKModel,
        DrugGNN,
        MultiViewNeuralNetwork,
        PharmaFormerModel,
        PrecilyModel,
        SimpleNeuralNetwork,
        SuperFELTR,
    )
    from .baselines.multi_view_lightgbm import MultiViewLightGBM
    from .SparseGO.sparsego import SparseGOModel

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
        "SparseGO": SparseGOModel,
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
            "SuperFELTR": SuperFELTR,
            "SparseGO": SparseGOModel,
            "SINGLE_DRUG_MODEL_FACTORY": single,
            "MULTI_DRUG_MODEL_FACTORY": multi,
            "MODEL_FACTORY": factory,
        }
    )
    _LAZY_LOADED = True


def __getattr__(name: str) -> Any:
    if name in __all__ and name != "DRPModel":
        _lazy_load_public_models()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
