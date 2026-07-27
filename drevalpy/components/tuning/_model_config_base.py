"""Resolve base ModelConfig instances for public DRPModel classes."""

from __future__ import annotations

from typing import Any

from drevalpy.models.config import ModelConfig


def base_model_config_for_drp_model(model_class: type[Any]) -> ModelConfig | None:
    """Resolve the base modular config for a public DRPModel class without hyperparameters."""
    base = getattr(model_class, "_base_model_config", None)
    if isinstance(base, ModelConfig):
        return base.model_copy(deep=True)

    model_config = getattr(model_class, "model_config", None)
    if callable(model_config):
        try:
            config = model_config()
        except RuntimeError:
            config = None
        if isinstance(config, ModelConfig):
            return config

    get_model_name = getattr(model_class, "get_model_name", None)
    if not callable(get_model_name):
        return None
    model_name = get_model_name()
    from drevalpy.models.factory import model_config_for_name

    try:
        return model_config_for_name(model_name, {})
    except KeyError:
        return None
