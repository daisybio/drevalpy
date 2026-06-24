"""Public DRPModel baseline adapters backed by the component stack."""

from drevalpy.components.predictors.baselines.multi_view_random_forest import MultiViewRandomForest
from drevalpy.components.predictors.baselines.multi_view_xgboost import MultiViewXGBoost
from drevalpy.components.predictors.baselines.naive_pred import (
    NaiveCellLineMeanPredictor,
    NaiveDrugMeanPredictor,
    NaiveMeanEffectsPredictor,
    NaiveModel,
    NaivePredictor,
    NaiveTissueDrugMeanPredictor,
    NaiveTissueMeanPredictor,
)
from drevalpy.components.predictors.baselines.singledrug_baselines import (
    SingleDrugElasticNet,
    SingleDrugRandomForest,
)
from drevalpy.components.predictors.baselines.sklearn_models import (
    AdaBoostDecisionTree,
    ElasticNetModel,
    GradientBoosting,
    KNNRegressor,
    LassoModel,
    RandomForest,
    SklearnModel,
    SVMRegressor,
)

__all__ = [
    "AdaBoostDecisionTree",
    "ElasticNetModel",
    "GradientBoosting",
    "KNNRegressor",
    "LassoModel",
    "MultiViewRandomForest",
    "MultiViewXGBoost",
    "NaiveCellLineMeanPredictor",
    "NaiveDrugMeanPredictor",
    "NaiveMeanEffectsPredictor",
    "NaiveModel",
    "NaivePredictor",
    "NaiveTissueDrugMeanPredictor",
    "NaiveTissueMeanPredictor",
    "RandomForest",
    "SingleDrugElasticNet",
    "SingleDrugRandomForest",
    "SklearnModel",
    "SVMRegressor",
]
