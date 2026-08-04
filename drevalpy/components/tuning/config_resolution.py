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
    """Return a ``ModelConfig`` with structured defaults applied.

    Args:
        model_class: Public ``DRPModel`` subclass.

    Returns:
        Resolved config with component defaults, or ``None`` when the model has
        no modular config.
    """
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
    """Apply a structured Ray/Optuna sample onto the base model config.

    Args:
        model_class: Public ``DRPModel`` subclass.
        merged_sample: Flat structured hyperparameter sample from Ray Tune.

    Returns:
        Updated ``ModelConfig``, or ``None`` when the model has no modular config.
    """
    config = base_model_config_for_drp_model(model_class)
    if config is None:
        return None
    return apply_merged_to_model_config(config, merged_sample)


def construct_drp_model_from_config(model_class: type[Any], config: ModelConfig) -> Any:
    """Construct a public DRPModel instance from a resolved ``ModelConfig``.

    Args:
        model_class: Public ``DRPModel`` subclass.
        config: Fully resolved modular configuration.

    Returns:
        Instantiated model object.
    """
    from_resolved = getattr(model_class, "_from_resolved_config", None)
    if callable(from_resolved):
        return from_resolved(config)
    from .public_flat import public_hyperparameters_from_config

    return model_class(public_hyperparameters_from_config(config))


def structured_space_for_drp_model(model_class: type[Any]) -> dict[str, Any]:
    """Return the merged structured search space for a DRPModel class.

    Args:
        model_class: Public ``DRPModel`` subclass.

    Returns:
        Flat search-space dict with prefixed component keys.
    """
    config = base_model_config_for_drp_model(model_class)
    if config is None:
        return {}
    return merge_model_config_spaces(config)


def default_hyperparameters_for_drp_model(model_class: type[Any]) -> dict[str, Any]:
    """Return default hyperparameters used by ``model_class()``.

    Args:
        model_class: Public ``DRPModel`` subclass.

    Returns:
        Public flat hyperparameter mapping for the model's default config.
    """
    from .public_flat import public_hyperparameters_from_config

    config = default_config_for_drp_model(model_class)
    if config is None:
        return {}
    return public_hyperparameters_from_config(config)


def has_tunable_hyperparameters(model_class: type[Any]) -> bool:
    """Return whether the model exposes a non-empty structured search space.

    Args:
        model_class: Public ``DRPModel`` subclass.

    Returns:
        ``True`` when at least one tunable parameter is declared.
    """
    return bool(structured_space_for_drp_model(model_class))


def assert_component_local_hyperparameters(config: ModelConfig) -> None:
    """Raise if namespaced keys leaked into component-local hyperparameter dicts.

    Args:
        config: Model configuration to validate.

    Raises:
        AssertionError: If a featurizer or predictor hyperparameter dict contains
            registry-qualified keys.
    """
    for featurizer in iter_featurizer_configs(config):
        for key in featurizer.hyperparameters:
            if key == "featurizers":
                continue
            if "." in key or key.startswith(("cell_line_featurizer.", "drug_featurizer.", "predictor.")):
                msg = f"namespaced key {key!r} found in {featurizer.name} hyperparameters"
                raise AssertionError(msg)
    for key in config.predictor.hyperparameters:
        if key.startswith(("cell_line_featurizer.", "drug_featurizer.", "predictor.")) or key.count(".") >= 2:
            msg = f"namespaced key {key!r} found in predictor hyperparameters"
            raise AssertionError(msg)
