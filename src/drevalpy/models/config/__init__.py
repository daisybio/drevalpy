"""Declarative configuration for modular featurizer/predictor pairing."""

from __future__ import annotations

from .featurizer import FeaturizerConfig
from .io import from_dict, from_recipe, from_yaml
from .model import ModelConfig
from .predictor import PredictorConfig
from .validation import validate

__all__ = [
    "FeaturizerConfig",
    "ModelConfig",
    "PredictorConfig",
    "from_dict",
    "from_recipe",
    "from_yaml",
    "validate",
]
