"""Tests for stable drevalpy.components public exports."""

from __future__ import annotations

import drevalpy.components as components


def test_public_exports_are_importable() -> None:
    expected = {
        "ModelConfig",
        "FeaturizerConfig",
        "PredictorConfig",
        "PredictionMode",
        "ensure_components_registered",
        "register_builtin_components",
        "register_cell_line_featurizer",
        "register_drug_featurizer",
        "register_predictor",
        "format_model_id",
        "parse_model_id",
        "load_extensions",
        "list_predictor_metadata",
    }
    for name in expected:
        assert hasattr(components, name), name


def test_components_do_not_reexport_orchestration() -> None:
    orchestration_exports = {
        "ComposedModel",
        "build_model_config_from_spec",
        "model_config_from_spec",
        "model_config_for_name",
        "get_zoo_config",
        "list_zoo_names",
        "ComponentDRPBridge",
    }
    for name in orchestration_exports:
        assert not hasattr(components, name), name
