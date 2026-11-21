"""Tests for the :mod:`drevalpy.components` package, which re-exports nothing."""

from __future__ import annotations

import pytest

import drevalpy.components as components
import drevalpy.models.config as model_config

MODEL_CONFIG_NAMES = (
    "ModelConfig",
    "FeaturizerConfig",
    "PredictorConfig",
)
FORMER_EXPORTS = (
    "register_builtin_components",
    "register_sample_featurizer",
    "register_drug_featurizer",
    "register_predictor",
    "load_extensions",
    "list_predictor_metadata",
    "ComposedModel",
    "from_spec",
    "get_zoo_config",
    "list_zoo_names",
)


@pytest.mark.parametrize("name", MODEL_CONFIG_NAMES)
def test_model_config_lives_under_models(name: str) -> None:
    assert hasattr(model_config, name)
    assert not hasattr(components, name)


@pytest.mark.parametrize("name", FORMER_EXPORTS)
def test_components_reexports_nothing(name: str) -> None:
    assert not hasattr(components, name)
