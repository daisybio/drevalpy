__all__ = [
    "SimpleNeuralNetwork",
    "ElasticNetModel",
    "RandomForest",
    "SVMRegressor",
    "NaivePredictor",
    "NaiveDrugMeanPredictor",
    "NaiveCellLineMeanPredictor",
]

from .Baselines.naive_pred import (
    NaivePredictor,
    NaiveDrugMeanPredictor,
    NaiveCellLineMeanPredictor,
)
from .Baselines.elastic_net_model import ElasticNetModel
from .Baselines.random_forest import RandomForest
from .Baselines.svm import SVMRegressor
from .SimpleNeuralNetwork.simple_neural_network import SimpleNeuralNetwork
from .Baselines.singledrug_random_forest import SingleDrugRandomForest

MODEL_FACTORY = {
    "SimpleNeuralNetwork": SimpleNeuralNetwork,
    "ElasticNet": ElasticNetModel,
    "RandomForest": RandomForest,
    "SVR": SVMRegressor,
    "NaivePredictor": NaivePredictor,
    "NaiveDrugMeanPredictor": NaiveDrugMeanPredictor,
    "NaiveCellLineMeanPredictor": NaiveCellLineMeanPredictor,
    "SingleDrugRandomForest": SingleDrugRandomForest,
}
