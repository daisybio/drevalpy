"""Tests for stable drevalpy.components public exports."""

from __future__ import annotations

import pytest

import drevalpy.components as components
import drevalpy.models.config as model_config

MODEL_CONFIG_NAMES = (
    "ModelConfig",
    "FeaturizerConfig",
    "SampleFeaturizerConfig",
    "DrugFeaturizerConfig",
    "PredictorConfig",
    "PredictionMode",
)
ORCHESTRATION_NAMES = (
    "ComposedModel",
    "zoo_config",
    "from_spec",
    "model_config_for_name",
    "get_zoo_config",
    "list_zoo_names",
    "ComponentDRPBridge",
    "format_model_id",
    "parse_model_id",
)


@pytest.mark.parametrize(
    "name",
    [
        "register_builtin_components",
        "register_sample_featurizer",
        "register_drug_featurizer",
        "register_predictor",
        "load_extensions",
        "list_predictor_metadata",
    ],
)
def test_public_exports_are_importable(name: str) -> None:
    assert hasattr(components, name)


@pytest.mark.parametrize("name", MODEL_CONFIG_NAMES)
def test_model_config_lives_under_models(name: str) -> None:
    assert hasattr(model_config, name)
    assert not hasattr(components, name)


@pytest.mark.parametrize("name", ORCHESTRATION_NAMES)
def test_components_do_not_reexport_orchestration(name: str) -> None:
    assert not hasattr(components, name)
