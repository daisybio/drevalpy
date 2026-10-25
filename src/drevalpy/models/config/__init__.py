"""Declarative configuration for modular featurizer/predictor pairing."""

from __future__ import annotations

from .featurizer import (
    DrugFeaturizerConfig,
    FeaturizerConfig,
    SampleFeaturizerConfig,
)
from .io import from_dict, from_spec, from_yaml
from .model import ModelConfig
from .predictor import PredictorConfig
from .resolved import ResolvedModelConfig
from .validation import validate

__all__ = [
    "DrugFeaturizerConfig",
    "FeaturizerConfig",
    "ModelConfig",
    "PredictorConfig",
    "ResolvedModelConfig",
    "SampleFeaturizerConfig",
    "from_dict",
    "from_spec",
    "from_yaml",
    "validate",
]
