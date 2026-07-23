"""Tests for default and tuned ModelConfig resolution."""

from __future__ import annotations

import drevalpy.components.register_builtins as register_builtins
from drevalpy.components.tuning.config_resolution import (
    assert_component_local_hyperparameters,
    default_config_for_drp_model,
    has_tunable_hyperparameters,
    structured_space_for_drp_model,
)
from drevalpy.models import MODEL_FACTORY


def test_default_config_for_elastic_net_is_component_local() -> None:
    register_builtins.register_builtin_components()
    config = default_config_for_drp_model(MODEL_FACTORY["ElasticNet"])
    assert config is not None
    assert config.predictor.name == "elasticNet"
    assert_component_local_hyperparameters(config)


def test_structured_space_for_naive_is_empty() -> None:
    register_builtins.register_builtin_components()
    assert structured_space_for_drp_model(MODEL_FACTORY["NaivePredictor"]) == {}
    assert has_tunable_hyperparameters(MODEL_FACTORY["NaivePredictor"]) is False
