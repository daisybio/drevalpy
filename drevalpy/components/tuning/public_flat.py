"""Translate between public hyperparameter mappings and ModelConfig."""

from __future__ import annotations

from typing import Any

from drevalpy.components.tuning.config_resolution import assert_component_local_hyperparameters
from drevalpy.components.tuning.hyperparameter_keys import (
    build_ownership_index,
    export_public_mapping,
    resolve_to_qualified_mapping,
)
from drevalpy.models.config import ModelConfig
from drevalpy.models.flat_hyperparameters import (
    PUBLIC_VIEW_KEYS,
    _apply_view_overrides,
    _warn_legacy_view_keys,
)

from ._model_config_base import base_model_config_for_drp_model
from .search_space import apply_merged_to_model_config


def apply_public_hyperparameters_to_config(
    config: ModelConfig,
    mapping: dict[str, Any],
    *,
    warn_legacy_view_keys: bool = True,
) -> ModelConfig:
    """Apply a collision-aware public hyperparameter mapping onto a ``ModelConfig``.

    :param config: config.
    :param mapping: mapping.
    :param warn_legacy_view_keys: warn legacy view keys.
    :returns: Result.
    """
    if not mapping:
        return config.model_copy(deep=True)
    if warn_legacy_view_keys:
        _warn_legacy_view_keys(mapping)

    normalized = dict(mapping)
    if "methylation_n_components" not in normalized and "methylation_pca_components" in normalized:
        normalized["methylation_n_components"] = normalized.pop("methylation_pca_components")

    result = _apply_view_overrides(config.model_copy(deep=True), normalized)
    index = build_ownership_index(result)
    qualified = resolve_to_qualified_mapping(
        result,
        normalized,
        index,
        reserved_keys=PUBLIC_VIEW_KEYS,
    )
    if qualified:
        result = apply_merged_to_model_config(result, qualified)
    result.validate()
    assert_component_local_hyperparameters(result)
    return result


def public_hyperparameters_from_config(
    config: ModelConfig,
    *,
    include_view_keys: bool = False,
) -> dict[str, Any]:
    """Export a model config into a collision-aware public hyperparameter mapping.

    :param config: config.
    :param include_view_keys: include view keys.
    :returns: Result.
    """
    return export_public_mapping(config, include_view_keys=include_view_keys)


def config_from_public_hyperparameters(
    model_class: type[Any],
    hyperparameters: dict[str, Any] | None,
) -> ModelConfig | None:
    """Convert a public hyperparameter mapping into a ``ModelConfig``.

    :param model_class: model class.
    :param hyperparameters: hyperparameters.
    :returns: Result.
    """
    config = base_model_config_for_drp_model(model_class)
    if config is None:
        return None
    if not hyperparameters:
        return config
    return apply_public_hyperparameters_to_config(config, hyperparameters)


def model_config_for_drp_model(
    model_class: type[Any],
    hyperparameters: dict[str, Any] | None = None,
) -> ModelConfig | None:
    """Resolve a modular config for a public DRPModel class.

    :param model_class: model class.
    :param hyperparameters: hyperparameters.
    :returns: Result.
    """
    if hyperparameters:
        return config_from_public_hyperparameters(model_class, hyperparameters)
    return base_model_config_for_drp_model(model_class)


def flat_hyperparameters_from_model_config(config: ModelConfig) -> dict[str, Any]:
    """Backward-compatible alias for ``public_hyperparameters_from_config``.

    :param config: config.
    :returns: Result.
    """
    return public_hyperparameters_from_config(config)


def config_from_build_hyperparameters(
    model_class: type[Any],
    hyperparameters: dict[str, Any] | None,
) -> ModelConfig | None:
    """Backward-compatible alias for ``config_from_public_hyperparameters``.

    :param model_class: model class.
    :param hyperparameters: hyperparameters.
    :returns: Result.
    """
    return config_from_public_hyperparameters(model_class, hyperparameters)


def tuned_flat_hyperparameters(
    model_class: type[Any],
    merged_sample: dict[str, Any],
) -> dict[str, Any]:
    """Convert a merged Ray/Optuna sample into a public hyperparameter mapping.

    :param model_class: model class.
    :param merged_sample: merged sample.
    :returns: Result.
    """
    from .config_resolution import tuned_config_for_drp_model

    config = tuned_config_for_drp_model(model_class, merged_sample)
    if config is None:
        return dict(merged_sample)
    return public_hyperparameters_from_config(config)


def apply_public_flat_hyperparameters_alias(config: ModelConfig, flat: dict[str, Any]) -> ModelConfig:
    """Backward-compatible alias for ``apply_public_hyperparameters_to_config``.

    :param config: config.
    :param flat: flat.
    :returns: Result.
    """
    return apply_public_hyperparameters_to_config(config, flat)
