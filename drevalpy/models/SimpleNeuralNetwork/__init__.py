"""Compatibility package for moved SimpleNeuralNetwork implementation."""

from drevalpy.components.predictors.literature.impl.simple_neural_network.simple_neural_network import SimpleNeuralNetwork
from drevalpy.components.predictors.literature.impl.simple_neural_network.multi_view_neural_network import MultiViewNeuralNetwork

__all__ = ["SimpleNeuralNetwork", "MultiViewNeuralNetwork"]
