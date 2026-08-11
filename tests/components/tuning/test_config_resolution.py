"""Tests for default and tuned ModelConfig resolution."""

from __future__ import annotations

from drevalpy.components.registry import register_builtins
from drevalpy.models import construct_model
from drevalpy.models.tuning.config_resolution import (
    assert_component_local_hyperparameters,
    default_config_for_drp_model,
    has_tunable_hyperparameters,
    structured_space_for_drp_model,
)


def test_default_config_for_elastic_net_is_component_local() -> None:
    register_builtins.register_builtin_components()
    config = default_config_for_drp_model(construct_model("ElasticNet"))
    assert config is not None
    assert config.template.predictor.name == "elasticNet"
    assert_component_local_hyperparameters(config)


def test_structured_space_for_naive_is_empty() -> None:
    register_builtins.register_builtin_components()
    assert structured_space_for_drp_model(construct_model("NaivePredictor")) == {}
    assert has_tunable_hyperparameters(construct_model("NaivePredictor")) is False
