"""Module containing all drug response prediction models."""

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
    "XGDP",
    "KNNRegressor",
    "AdaBoostDecisionTree",
    "Lasso",
    "MultiViewXGBoost",
]

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
from .DIPK.dipk import DIPKModel
from .drp_model import DRPModel
from .DrugGNN import DrugGNN
from .MOLIR.molir import MOLIR
from .PharmaFormer.pharmaformer import PharmaFormerModel
from .SimpleNeuralNetwork.multi_view_neural_network import MultiViewNeuralNetwork
from .SimpleNeuralNetwork.simple_neural_network import SimpleNeuralNetwork
from .SRMF.srmf import SRMF
from .SuperFELTR.superfeltr import SuperFELTR
from .XGDP.xgdp import XGDP

# SINGLE_DRUG_MODEL_FACTORY is used in the pipeline!
SINGLE_DRUG_MODEL_FACTORY: dict[str, type[DRPModel]] = {
    "SingleDrugElasticNet": SingleDrugElasticNet,
    "SingleDrugRandomForest": SingleDrugRandomForest,
    "MOLIR": MOLIR,
    "SuperFELTR": SuperFELTR,
}

# MULTI_DRUG_MODEL_FACTORY is used in the pipeline!
MULTI_DRUG_MODEL_FACTORY: dict[str, type[DRPModel]] = {
    "NaivePredictor": NaivePredictor,
    "NaiveDrugMeanPredictor": NaiveDrugMeanPredictor,
    "NaiveCellLineMeanPredictor": NaiveCellLineMeanPredictor,
    "NaiveMeanEffectsPredictor": NaiveMeanEffectsPredictor,
    "NaiveTissueMeanPredictor": NaiveTissueMeanPredictor,
    "NaiveTissueDrugMeanPredictor": NaiveTissueDrugMeanPredictor,
    "ElasticNet": ElasticNetModel,
    "RandomForest": RandomForest,
    "SVR": SVMRegressor,
    "SimpleNeuralNetwork": SimpleNeuralNetwork,
    "MultiViewNeuralNetwork": MultiViewNeuralNetwork,
    "MultiViewRandomForest": MultiViewRandomForest,
    "GradientBoosting": GradientBoosting,
    "SRMF": SRMF,
    "DIPK": DIPKModel,
    "DrugGNN": DrugGNN,
    "PharmaFormer": PharmaFormerModel,
    "XGDP": XGDP,
    "KNNRegressor": KNNRegressor,
    "AdaBoostDecisionTree": AdaBoostDecisionTree,
    "Lasso": LassoModel,
    "MultiViewXGBoost": MultiViewXGBoost,
}

# MODEL_FACTORY is used in the pipeline!
MODEL_FACTORY = MULTI_DRUG_MODEL_FACTORY.copy()
MODEL_FACTORY.update(SINGLE_DRUG_MODEL_FACTORY)
