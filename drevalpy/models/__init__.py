__all__ = [
    "SimpleNeuralNetwork",
    "MultiOmicsNeuralNetwork",
    "ElasticNetModel",
    "RandomForest",
    "MultiOmicsRandomForest",
    "SingleDrugRandomForest",
    "SVMRegressor",
    "NaivePredictor",
    "NaiveDrugMeanPredictor",
    "NaiveCellLineMeanPredictor",
    "MODEL_FACTORY",
]

from .Baselines.naive_pred import (
    NaivePredictor,
    NaiveDrugMeanPredictor,
    NaiveCellLineMeanPredictor,
)
from .Baselines.elastic_net_model import ElasticNetModel
from .Baselines.random_forest import RandomForest, MultiOmicsRandomForest
from .Baselines.svm import SVMRegressor
from .SimpleNeuralNetwork.simple_neural_network import SimpleNeuralNetwork
from .SimpleNeuralNetwork.multiomics_neural_network import MultiOmicsNeuralNetwork
from .Baselines.singledrug_random_forest import SingleDrugRandomForest

MODEL_FACTORY = {
    "SimpleNeuralNetwork": SimpleNeuralNetwork,
    "MultiOmicsNeuralNetwork": MultiOmicsNeuralNetwork,
    "ElasticNet": ElasticNetModel,
    "RandomForest": RandomForest,
    "MultiOmicsRandomForest": MultiOmicsRandomForest,
    "SVR": SVMRegressor,
    "NaivePredictor": NaivePredictor,
    "NaiveDrugMeanPredictor": NaiveDrugMeanPredictor,
    "NaiveCellLineMeanPredictor": NaiveCellLineMeanPredictor,
    "SingleDrugRandomForest": SingleDrugRandomForest,
}
