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
    "NaivePredictor": NaivePredictor,
    "NaiveDrugMeanPredictor": NaiveDrugMeanPredictor,
    "NaiveCellLineMeanPredictor": NaiveCellLineMeanPredictor,
    "ElasticNet": ElasticNetModel,
    "RandomForest": RandomForest,
    "SVR": SVMRegressor,
    "SimpleNeuralNetwork": SimpleNeuralNetwork,
    "MultiOmicsNeuralNetwork": MultiOmicsNeuralNetwork,
    "MultiOmicsRandomForest": MultiOmicsRandomForest,
    "SingleDrugRandomForest": SingleDrugRandomForest,
}
