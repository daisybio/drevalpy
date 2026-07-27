"""Resolve structured defaults and search spaces for DRPModel classes."""

from __future__ import annotations

from typing import Any

from drevalpy.models.config import ModelConfig

from ._featurizer_walk import iter_featurizer_configs
from ._model_config_base import base_model_config_for_drp_model
from .search_space import (
    apply_merged_to_model_config,
    defaults_from_merged_space,
    merge_model_config_spaces,
)


def default_config_for_drp_model(model_class: type[Any]) -> ModelConfig | None:
    """Return a ``ModelConfig`` with structured defaults applied."""
    config = base_model_config_for_drp_model(model_class)
    if config is None:
        return None
    space = merge_model_config_spaces(config)
    merged_defaults = defaults_from_merged_space(space)
    return apply_merged_to_model_config(config, merged_defaults)


def tuned_config_for_drp_model(
    model_class: type[Any],
    merged_sample: dict[str, Any],
) -> ModelConfig | None:
    """Apply a structured Ray/Optuna sample onto the base model config."""
    config = base_model_config_for_drp_model(model_class)
    if config is None:
        return None
    return apply_merged_to_model_config(config, merged_sample)


def construct_drp_model_from_config(model_class: type[Any], config: ModelConfig) -> Any:
    """Construct a public DRPModel instance from a resolved ``ModelConfig``."""
    from_model_config = getattr(model_class, "from_model_config", None)
    if callable(from_model_config):
        return from_model_config(config)
    from .public_flat import public_hyperparameters_from_config

    return model_class(public_hyperparameters_from_config(config))


def structured_space_for_drp_model(model_class: type[Any]) -> dict[str, Any]:
    """Return the merged structured search space for a DRPModel class."""
    config = base_model_config_for_drp_model(model_class)
    if config is None:
        return {}
    return merge_model_config_spaces(config)


def default_hyperparameters_for_drp_model(model_class: type[Any]) -> dict[str, Any]:
    """Return default hyperparameters used by ``model_class()``."""
    from .public_flat import public_hyperparameters_from_config

    config = default_config_for_drp_model(model_class)
    if config is None:
        return {}
    return public_hyperparameters_from_config(config)


def has_tunable_hyperparameters(model_class: type[Any]) -> bool:
    """Return whether the model exposes a non-empty structured search space."""
    return bool(structured_space_for_drp_model(model_class))


def assert_component_local_hyperparameters(config: ModelConfig) -> None:
    """Raise if namespaced keys leaked into component-local hyperparameter dicts."""
    for featurizer in iter_featurizer_configs(config):
        for key in featurizer.hyperparameters:
            if key == "featurizers":
                continue
            if "." in key or key.startswith(("featurizer.", "predictor.")):
                msg = f"namespaced key {key!r} found in {featurizer.name} hyperparameters"
                raise AssertionError(msg)
    for key in config.predictor.hyperparameters:
        if key.startswith(("featurizer.", "predictor.")) or key.count(".") >= 2:
            msg = f"namespaced key {key!r} found in predictor hyperparameters"
            raise AssertionError(msg)
