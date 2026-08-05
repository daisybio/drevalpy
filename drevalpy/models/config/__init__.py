"""Declarative configuration for modular featurizer/predictor pairing."""

from __future__ import annotations

from typing import Any

from drevalpy.models.config.featurizer import (
    CellLineFeaturizerConfig,
    DrugFeaturizerConfig,
    FeaturizerConfig,
)
from drevalpy.models.config.model import ModelConfig
from drevalpy.models.config.predictor import PredictorConfig
from drevalpy.types.model_scope import ModelScope
from drevalpy.types.prediction_mode import PredictionMode

__all__ = [
    "CellLineFeaturizerConfig",
    "DrugFeaturizerConfig",
    "FeaturizerConfig",
    "ModelConfig",
    "ModelScope",
    "PredictionMode",
    "PredictorConfig",
    "build_model_config_from_spec",
    "model_config_from_dict",
    "model_config_from_spec",
    "model_config_from_yaml",
    "validate_model_config",
]

_LAZY_EXPORTS = {
    "build_model_config_from_spec": ("drevalpy.models.config.spec", "build_model_config_from_spec"),
    "model_config_from_dict": ("drevalpy.models.config.io", "model_config_from_dict"),
    "model_config_from_spec": ("drevalpy.models.config.io", "model_config_from_spec"),
    "model_config_from_yaml": ("drevalpy.models.config.io", "model_config_from_yaml"),
    "validate_model_config": ("drevalpy.models.config.validation", "validate_model_config"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        import importlib

        module_name, attr = _LAZY_EXPORTS[name]
        value = getattr(importlib.import_module(module_name), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
