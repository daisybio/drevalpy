"""Tests for stable drevalpy.components public exports."""

from __future__ import annotations

import drevalpy.components as components
import drevalpy.models.config as model_config


def test_public_exports_are_importable() -> None:
    expected = {
        "register_builtin_components",
        "register_cell_line_featurizer",
        "register_drug_featurizer",
        "register_predictor",
        "load_extensions",
        "list_predictor_metadata",
    }
    for name in expected:
        assert hasattr(components, name), name


def test_model_config_lives_under_models() -> None:
    expected = {
        "ModelConfig",
        "FeaturizerConfig",
        "CellLineFeaturizerConfig",
        "DrugFeaturizerConfig",
        "PredictorConfig",
        "PredictionMode",
    }
    for name in expected:
        assert hasattr(model_config, name), name
    for name in expected:
        assert not hasattr(components, name), name


def test_components_do_not_reexport_orchestration() -> None:
    orchestration_exports = {
        "ComposedModel",
        "_build_from_spec",
        "from_spec",
        "model_config_for_name",
        "get_zoo_config",
        "list_zoo_names",
        "ComponentDRPBridge",
        "format_model_id",
        "parse_model_id",
    }
    for name in orchestration_exports:
        assert not hasattr(components, name), name
