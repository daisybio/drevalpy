"""Declarative configuration for modular featurizer/predictor pairing."""

from __future__ import annotations

from drevalpy.types.enums.model_scope import ModelScope
from drevalpy.types.enums.prediction_mode import PredictionMode

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
    "ModelScope",
    "PredictionMode",
    "PredictorConfig",
    "ResolvedModelConfig",
    "SampleFeaturizerConfig",
    "from_dict",
    "from_spec",
    "from_yaml",
    "validate",
]
