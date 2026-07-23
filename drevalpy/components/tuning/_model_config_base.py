"""Resolve base ModelConfig instances for public DRPModel classes."""

from __future__ import annotations

from typing import Any

from drevalpy.models.config import ModelConfig


def base_model_config_for_drp_model(model_class: type[Any]) -> ModelConfig | None:
    """Resolve the base modular config for a public DRPModel class without hyperparameters."""
    spec = getattr(model_class, "_model_spec", None)
    if isinstance(spec, str):
        return ModelConfig.from_spec(spec)

    model_name = model_class.get_model_name()
    from drevalpy.models.factory import model_config_for_name

    try:
        return model_config_for_name(model_name, {})
    except KeyError:
        return None
