"""Resolve zoo/spec names to `~drevalpy.models.config.ModelConfig` objects."""

from __future__ import annotations

from typing import Any

from drevalpy.models.config import ModelConfig, ResolvedModelConfig
from drevalpy.types.enums.prediction_mode import PredictionMode


def model_config_for_name(
    model_name: str,
    hyperparameters: dict[str, Any] | None = None,
    *,
    prediction_mode: PredictionMode | None = None,
) -> ModelConfig | ResolvedModelConfig:
    """Resolve a factory/zoo name to a modular config with public flat HP applied.

    :param model_name: Built-in or external zoo preset name.
    :param hyperparameters: Optional flat public hyperparameter overrides.
    :param prediction_mode: Optional prediction mode overriding the preset's own value.
    :returns: Template ``ModelConfig``, or ``ResolvedModelConfig`` when overrides are given.
    :raises KeyError: If ``model_name`` is not a known zoo entry.
    """
    from drevalpy.models.zoo import list_zoo_names, zoo_model_config

    if model_name not in list_zoo_names(include_external=True):
        msg = f"Unknown model name: {model_name}"
        raise KeyError(msg)
    return zoo_model_config(model_name, hyperparameters, prediction_mode=prediction_mode)
