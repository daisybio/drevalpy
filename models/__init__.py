__all__ = [
    "SimpleNeuralNetwork",
    "ElasticNetModel",
]

from .Baselines.linear_model import ElasticNetModel
from .SimpleNeuralNetwork.simple_neural_network import SimpleNeuralNetwork

MODEL_FACTORY = {"SimpleNeuralNetwork": SimpleNeuralNetwork,
                 "ElasticNet": ElasticNetModel}
