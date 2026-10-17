"""Predictor registry: register, discover, and retrieve predictor classes."""

from drevalpy.registry.predictor._registration import get, list, metadata, register, table
from drevalpy.registry.predictor._registry import PredictorRegistry, predictor_registry

__all__ = [
    "PredictorRegistry",
    "get",
    "list",
    "metadata",
    "predictor_registry",
    "register",
    "table",
]
