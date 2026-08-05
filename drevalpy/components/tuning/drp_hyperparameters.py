"""Resolve structured hyperparameter spaces and defaults for DRPModel classes."""

from __future__ import annotations

from drevalpy.components.tuning.compatibility_keys import LEGACY_FEATURIZER_FLAT_KEYS, PUBLIC_VIEW_KEYS

from ._model_config_base import base_model_config_for_drp_model
from .compatibility_keys import append_featurizer_flat_keys
from .config_resolution import (
    assert_component_local_hyperparameters,
    construct_drp_model_from_config,
    default_config_for_drp_model,
    default_hyperparameters_for_drp_model,
    has_tunable_hyperparameters,
    structured_space_for_drp_model,
    tuned_config_for_drp_model,
)
from .public_flat import (
    apply_public_flat_hyperparameters_alias,
    config_from_build_hyperparameters,
    config_from_public_hyperparameters,
    flat_hyperparameters_from_model_config,
    model_config_for_drp_model,
    public_hyperparameters_from_config,
    tuned_flat_hyperparameters,
)

# Backward-compatible aliases for callers/tests that import private names.
_LEGACY_FEATURIZER_FLAT_KEYS = LEGACY_FEATURIZER_FLAT_KEYS
_PUBLIC_VIEW_KEYS = PUBLIC_VIEW_KEYS

# Backward-compatible private aliases used by tests and migration docs.
_append_featurizer_flat_keys = append_featurizer_flat_keys
_apply_public_flat_hyperparameters = apply_public_flat_hyperparameters_alias

__all__ = [
    "assert_component_local_hyperparameters",
    "base_model_config_for_drp_model",
    "construct_drp_model_from_config",
    "config_from_build_hyperparameters",
    "config_from_public_hyperparameters",
    "default_config_for_drp_model",
    "default_hyperparameters_for_drp_model",
    "flat_hyperparameters_from_model_config",
    "has_tunable_hyperparameters",
    "model_config_for_drp_model",
    "public_hyperparameters_from_config",
    "structured_space_for_drp_model",
    "tuned_config_for_drp_model",
    "tuned_flat_hyperparameters",
]
