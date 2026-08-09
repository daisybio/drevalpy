"""Resolve structured hyperparameter spaces and defaults for DRPModel classes."""

from __future__ import annotations

from drevalpy.components.core.tuning.compatibility_keys import LEGACY_FEATURIZER_FLAT_KEYS, PUBLIC_VIEW_KEYS

from ._model_config_base import base_model_config_for_drp_model
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
    config_from_public_hyperparameters,
    public_hyperparameters_from_config,
)

# Backward-compatible aliases for callers/tests that import private names.
_LEGACY_FEATURIZER_FLAT_KEYS = LEGACY_FEATURIZER_FLAT_KEYS
_PUBLIC_VIEW_KEYS = PUBLIC_VIEW_KEYS

__all__ = [
    "assert_component_local_hyperparameters",
    "base_model_config_for_drp_model",
    "config_from_public_hyperparameters",
    "construct_drp_model_from_config",
    "default_config_for_drp_model",
    "default_hyperparameters_for_drp_model",
    "has_tunable_hyperparameters",
    "public_hyperparameters_from_config",
    "structured_space_for_drp_model",
    "tuned_config_for_drp_model",
]
