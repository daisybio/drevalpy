"""Public drug response prediction models and legacy experiment adapters.

This package exposes :class:`~drevalpy.models.drp_model.DRPModel` subclasses,
``MODEL_FACTORY``, and public model compatibility classes. Baseline and literature
implementations live under :mod:`drevalpy.components.predictors` and are exposed
here for backward compatibility via :mod:`drevalpy.models._component_bridge`.
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
from .baselines.multi_view_lightgbm import MultiViewLightGBM
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
