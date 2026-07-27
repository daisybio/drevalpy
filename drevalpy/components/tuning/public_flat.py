"""Translate between public flat hyperparameters and ModelConfig."""

from __future__ import annotations

from typing import Any

from drevalpy.components.registry import get_predictor
from drevalpy.models.config import ModelConfig
from drevalpy.models.featurizer_mapping import cell_line_views_from_model_config, drug_views_from_model_config
from drevalpy.models.flat_hyperparameters import apply_public_flat_hyperparameters

from ._model_config_base import base_model_config_for_drp_model
from .compatibility_keys import append_featurizer_flat_keys
from .search_space import apply_merged_to_model_config


def public_hyperparameters_from_config(
    config: ModelConfig,
    *,
    include_view_keys: bool = False,
) -> dict[str, Any]:
    """Flatten a model config into public constructor hyperparameters.

    By default, deprecated ``cell_line_views`` / ``drug_views`` keys are omitted;
    featurizer composition already encodes the inputs. Pass
    ``include_view_keys=True`` only for explicit legacy serialization.
    """
    flat: dict[str, Any] = {}
    if include_view_keys:
        cell_line_views = cell_line_views_from_model_config(config)
        drug_views = drug_views_from_model_config(config)
        if cell_line_views:
            flat["cell_line_views"] = cell_line_views
        if drug_views:
            flat["drug_views"] = drug_views
    if config.predictor is not None:
        predictor_cls = get_predictor(config.predictor.name)
        engine_cls = getattr(predictor_cls, "_engine_cls", None)
        if engine_cls is not None:
            flat.update(engine_cls.get_default_hyperparameters())
        else:
            flat.update(predictor_cls.get_default_hyperparameters())
        flat.update(config.predictor.hyperparameters)
    append_featurizer_flat_keys(flat, config.cell_line_featurizer, "cell_line")
    append_featurizer_flat_keys(flat, config.drug_featurizer, "drug")
    return flat


def config_from_public_hyperparameters(
    model_class: type[Any],
    hyperparameters: dict[str, Any] | None,
) -> ModelConfig | None:
    """Convert public flat or structured hyperparameters into a ``ModelConfig``."""
    config = base_model_config_for_drp_model(model_class)
    if config is None:
        return None
    if not hyperparameters:
        return config
    if any("." in key for key in hyperparameters):
        return apply_merged_to_model_config(config.model_copy(deep=True), hyperparameters)
    return apply_public_flat_hyperparameters(config, hyperparameters)


def model_config_for_drp_model(
    model_class: type[Any],
    hyperparameters: dict[str, Any] | None = None,
) -> ModelConfig | None:
    """Resolve a modular config for a public DRPModel class."""
    if hyperparameters:
        return config_from_public_hyperparameters(model_class, hyperparameters)
    return base_model_config_for_drp_model(model_class)


def flat_hyperparameters_from_model_config(config: ModelConfig) -> dict[str, Any]:
    """Backward-compatible alias for ``public_hyperparameters_from_config``."""
    return public_hyperparameters_from_config(config)


def config_from_build_hyperparameters(
    model_class: type[Any],
    hyperparameters: dict[str, Any] | None,
) -> ModelConfig | None:
    """Backward-compatible alias for ``config_from_public_hyperparameters``."""
    return config_from_public_hyperparameters(model_class, hyperparameters)


def tuned_flat_hyperparameters(
    model_class: type[Any],
    merged_sample: dict[str, Any],
) -> dict[str, Any]:
    """Convert a merged Ray/Optuna sample into public constructor hyperparameters."""
    from .config_resolution import tuned_config_for_drp_model

    config = tuned_config_for_drp_model(model_class, merged_sample)
    if config is None:
        return dict(merged_sample)
    return public_hyperparameters_from_config(config)


def apply_public_flat_hyperparameters_alias(config: ModelConfig, flat: dict[str, Any]) -> ModelConfig:
    """Backward-compatible alias for ``apply_public_flat_hyperparameters``."""
    return apply_public_flat_hyperparameters(config, flat)
