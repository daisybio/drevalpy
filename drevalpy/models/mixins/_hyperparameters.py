"""Hyperparameter resolution mixin for DRPModel subclasses."""

from __future__ import annotations

from typing import Any

from drevalpy.models.config import ModelConfig
from drevalpy.models.factory import model_config_for_name
from drevalpy.models.tuning.public_flat import public_hyperparameters_from_config
from drevalpy.models.tuning.search_space import merge_model_config_spaces, resolve_model_config


class DRPHyperparametersMixin:
    """Mixin providing hyperparameter resolution for DRPModel subclasses.

    Resolves the base ModelConfig for the model class and uses it to derive
    structured search spaces and default hyperparameter mappings.
    """

    @classmethod
    def _resolve_base_config(cls) -> ModelConfig | None:
        """Resolve the base modular config for this model class.

        Resolution order:
        1. ``_base_model_config`` class attribute (deep-copied).
        2. ``model_config()`` class method.
        3. Zoo lookup via ``get_model_name()``.

        :returns: A fresh ModelConfig template, or None if unresolvable.
        """
        base = getattr(cls, "_base_model_config", None)
        if isinstance(base, ModelConfig):
            return ModelConfig.model_validate(base.model_dump(mode="python"))

        model_config_fn = getattr(cls, "model_config", None)
        if callable(model_config_fn):
            try:
                config = model_config_fn()
            except RuntimeError:
                config = None
            if isinstance(config, ModelConfig):
                return config

        get_model_name = getattr(cls, "get_model_name", None)
        if not callable(get_model_name):
            return None
        model_name = get_model_name()

        try:
            config = model_config_for_name(model_name, None)
        except KeyError:
            return None
        return config if isinstance(config, ModelConfig) else None

    @classmethod
    def get_structured_hyperparameter_space(cls) -> dict[str, Any]:
        """Return the merged structured hyperparameter space for this model.

        :returns: Structured hyperparameter search space for tuning.
        """
        config = cls._resolve_base_config()
        if config is None:
            return {}
        return merge_model_config_spaces(config)

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, Any]:
        """Return default hyperparameters used by ``cls()``.

        :returns: Default flat public hyperparameters for a new instance.
        """
        config = cls._resolve_base_config()
        if config is None:
            return {}
        resolved = resolve_model_config(config)
        return public_hyperparameters_from_config(resolved)

    @classmethod
    def get_hyperparameter_set(cls) -> list[dict[str, Any]]:
        """Return the default hyperparameter configuration for this model.

        :returns: Single-element list containing default hyperparameters.
        """
        return [cls.get_default_hyperparameters()]
