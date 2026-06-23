"""Public drug response prediction models and legacy experiment adapters.

This package exposes :class:`~drevalpy.models.drp_model.DRPModel` subclasses,
``MODEL_FACTORY``, and literature model compatibility classes. Baseline classes
delegate to the modular stack in :mod:`drevalpy.components` through
:mod:`drevalpy.models._component_bridge`. Literature models are implemented under
:mod:`drevalpy.components.predictors.literature` and exposed here for backward
compatibility.
"""

__all__ = [
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

from .baselines.multi_view_lightgbm import MultiViewLightGBM
from .baselines.multi_view_random_forest import MultiViewRandomForest
from .baselines.multi_view_xgboost import MultiViewXGBoost
from .baselines.naive_pred import (
    NaiveCellLineMeanPredictor,
    NaiveDrugMeanPredictor,
    NaiveMeanEffectsPredictor,
    NaivePredictor,
    NaiveTissueDrugMeanPredictor,
    NaiveTissueMeanPredictor,
)
from .baselines.singledrug_baselines import SingleDrugElasticNet, SingleDrugRandomForest
from .baselines.sklearn_models import (
    AdaBoostDecisionTree,
    ElasticNetModel,
    GradientBoosting,
    KNNRegressor,
    LassoModel,
    RandomForest,
    SVMRegressor,
)
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
from .drp_model import DRPModel
from .SparseGO.sparsego import SparseGOModel

# SINGLE_DRUG_MODEL_FACTORY is used in the pipeline!
SINGLE_DRUG_MODEL_FACTORY: dict[str, type[DRPModel]] = {
    "SingleDrugElasticNet": SingleDrugElasticNet,
    "SingleDrugRandomForest": SingleDrugRandomForest,
    "MOLIR": MOLIR,
    "SuperFELTR": SuperFELTR,
}

# MULTI_DRUG_MODEL_FACTORY is used in the pipeline!
MULTI_DRUG_MODEL_FACTORY: dict[str, type[DRPModel]] = {
    # Naive predictors
    "NaivePredictor": NaivePredictor,
    "NaiveCellLineMeanPredictor": NaiveCellLineMeanPredictor,
    "NaiveDrugMeanPredictor": NaiveDrugMeanPredictor,
    "NaiveMeanEffectsPredictor": NaiveMeanEffectsPredictor,
    "NaiveTissueMeanPredictor": NaiveTissueMeanPredictor,
    "NaiveTissueDrugMeanPredictor": NaiveTissueDrugMeanPredictor,
    # Sklearn Baselines
    "AdaBoostDecisionTree": AdaBoostDecisionTree,
    "ElasticNet": ElasticNetModel,
    "Lasso": LassoModel,
    "GradientBoosting": GradientBoosting,
    "KNNRegressor": KNNRegressor,
    "RandomForest": RandomForest,
    "MultiViewRandomForest": MultiViewRandomForest,
    "SVR": SVMRegressor,
    # Other Baselines
    "DrugGNN": DrugGNN,
    "SimpleNeuralNetwork": SimpleNeuralNetwork,
    "MultiViewNeuralNetwork": MultiViewNeuralNetwork,
    "MultiViewXGBoost": MultiViewXGBoost,
    "MultiViewLightGBM": MultiViewLightGBM,
    # Published models
    "DIPK": DIPKModel,
    "PharmaFormer": PharmaFormerModel,
    "SRMF": SRMF,
    "Precily": PrecilyModel,
    "SparseGO": SparseGOModel,
}

# MODEL_FACTORY is used in the pipeline!
MODEL_FACTORY = MULTI_DRUG_MODEL_FACTORY.copy()
MODEL_FACTORY.update(SINGLE_DRUG_MODEL_FACTORY)
