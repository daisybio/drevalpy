"""Tests for public flat hyperparameter translation."""

from __future__ import annotations

import drevalpy.components.register_builtins as register_builtins
from drevalpy.components.tuning.public_flat import (
    config_from_public_hyperparameters,
    model_config_for_drp_model,
    public_hyperparameters_from_config,
)
from drevalpy.models import construct_model


def test_public_round_trip_for_factory_model() -> None:
    register_builtins.register_builtin_components()
    model_cls = construct_model("ElasticNet")
    config = model_config_for_drp_model(model_cls)
    assert config is not None
    public = public_hyperparameters_from_config(config)
    rebuilt = config_from_public_hyperparameters(model_cls, public)
    assert rebuilt is not None
    assert rebuilt.predictor.name == "elasticNet"


def test_construct_model_spec_resolves_without_hyperparameters() -> None:
    register_builtins.register_builtin_components()
    model_cls = construct_model("PcaIdentityRF", "pca[expression]:identity:randomForest")
    config = model_config_for_drp_model(model_cls)
    assert config is not None
    assert config.predictor.name == "randomForest"
