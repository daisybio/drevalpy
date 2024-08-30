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
    "MODEL_FACTORY",
    "SINGLE_DRUG_MODEL_FACTORY",
]

from .baselines.naive_pred import (
    NaivePredictor,
    NaiveDrugMeanPredictor,
    NaiveCellLineMeanPredictor,
)
from .baselines.sklearn_models import ElasticNetModel, RandomForest, SVMRegressor
from .baselines.multi_omics_random_forest import MultiOmicsRandomForest
from .simple_neural_network.simple_neural_network import SimpleNeuralNetwork
from .simple_neural_network.multiomics_neural_network import MultiOmicsNeuralNetwork
from .baselines.singledrug_random_forest import SingleDrugRandomForest

SINGLE_DRUG_MODEL_FACTORY = {
    "SingleDrugRandomForest": SingleDrugRandomForest,
}

MODEL_FACTORY = {
    "NaivePredictor": NaivePredictor,
    "NaiveDrugMeanPredictor": NaiveDrugMeanPredictor,
    "NaiveCellLineMeanPredictor": NaiveCellLineMeanPredictor,
    "ElasticNet": ElasticNetModel,
    "RandomForest": RandomForest,
    "SVR": SVMRegressor,
    "SimpleNeuralNetwork": SimpleNeuralNetwork,
    "MultiOmicsNeuralNetwork": MultiOmicsNeuralNetwork,
    "MultiOmicsRandomForest": MultiOmicsRandomForest,
}

MODEL_FACTORY.update(SINGLE_DRUG_MODEL_FACTORY)
