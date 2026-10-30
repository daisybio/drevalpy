"""Abstract predictor base classes."""

from .base import Predictor
from .feature_free import FeatureFreePredictor
from .matrix import MatrixPredictor

__all__ = ["FeatureFreePredictor", "MatrixPredictor", "Predictor"]
