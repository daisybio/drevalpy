"""Compatibility re-export — implementation lives in :mod:`drevalpy.models.factory`."""

from drevalpy.models.factory import (
    LEGACY_PREDICTOR_BY_MODEL_NAME,
    NAIVE_PREDICTOR_BY_MODEL_NAME,
    SKLEARN_PREDICTOR_BY_MODEL_NAME,
    featurizer_configs_from_view_hyperparameters,
    legacy_model_config,
    model_config_for_name,
    naive_model_config,
    sklearn_model_config,
    sklearn_model_config_from_zoo,
)

__all__ = [
    "LEGACY_PREDICTOR_BY_MODEL_NAME",
    "NAIVE_PREDICTOR_BY_MODEL_NAME",
    "SKLEARN_PREDICTOR_BY_MODEL_NAME",
    "featurizer_configs_from_view_hyperparameters",
    "legacy_model_config",
    "model_config_for_name",
    "naive_model_config",
    "sklearn_model_config",
    "sklearn_model_config_from_zoo",
]
