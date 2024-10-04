"""
Module containing all drug response prediction models.
"""

__all__ = [
    "NaivePredictor",
    "NaiveDrugMeanPredictor",
    "NaiveCellLineMeanPredictor",
    "ElasticNetModel",
    "RandomForest",
    "SVMRegressor",
    "SimpleNeuralNetwork",
    "MultiOmicsNeuralNetwork",
    "MultiOmicsRandomForest",
    "SingleDrugRandomForest",
    "SRMF" "MULTI_DRUG_MODEL_FACTORY",
    "SINGLE_DRUG_MODEL_FACTORY",
    "MODEL_FACTORY",
]

from .baselines.naive_pred import (
    NaivePredictor,
    NaiveDrugMeanPredictor,
    NaiveCellLineMeanPredictor,
)
from .baselines.sklearn_models import (
    ElasticNetModel,
    RandomForest,
    SVMRegressor,
    GradientBoosting,
)
from .baselines.multi_omics_random_forest import MultiOmicsRandomForest
from .simple_neural_network.simple_neural_network import SimpleNeuralNetwork
from .simple_neural_network.multiomics_neural_network import MultiOmicsNeuralNetwork
from .baselines.singledrug_random_forest import SingleDrugRandomForest
from .SRMF.srmf import SRMF

SINGLE_DRUG_MODEL_FACTORY = {
    "SingleDrugRandomForest": SingleDrugRandomForest,
}

MULTI_DRUG_MODEL_FACTORY = {
    "NaivePredictor": NaivePredictor,
    "NaiveDrugMeanPredictor": NaiveDrugMeanPredictor,
    "NaiveCellLineMeanPredictor": NaiveCellLineMeanPredictor,
    "ElasticNet": ElasticNetModel,
    "RandomForest": RandomForest,
    "SVR": SVMRegressor,
    "SimpleNeuralNetwork": SimpleNeuralNetwork,
    "MultiOmicsNeuralNetwork": MultiOmicsNeuralNetwork,
    "MultiOmicsRandomForest": MultiOmicsRandomForest,
    "GradientBoosting": GradientBoosting,
    "SRMF": SRMF,
}

MODEL_FACTORY = MULTI_DRUG_MODEL_FACTORY.copy()
MODEL_FACTORY.update(SINGLE_DRUG_MODEL_FACTORY)
