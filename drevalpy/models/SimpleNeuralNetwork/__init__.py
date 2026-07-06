"""Compatibility package for moved SimpleNeuralNetwork implementation."""

from drevalpy.components.predictors.literature.impl.simple_neural_network.multi_view_neural_network import (
    MultiViewNeuralNetwork,
)
from drevalpy.components.predictors.literature.impl.simple_neural_network.simple_neural_network import (
    SimpleNeuralNetwork,
)

__all__ = ["SimpleNeuralNetwork", "MultiViewNeuralNetwork"]
