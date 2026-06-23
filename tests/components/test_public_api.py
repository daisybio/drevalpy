"""Smoke tests for stable drevalpy.components public exports."""

from __future__ import annotations

import drevalpy.components as components


def test_public_exports_are_importable() -> None:
    expected = {
        "ModelConfig",
        "FeaturizerConfig",
        "PredictorConfig",
        "PredictionMode",
        "ComposedModel",
        "build_model_config_from_spec",
        "ensure_components_registered",
        "register_builtin_components",
        "register_cell_line_featurizer",
        "register_drug_featurizer",
        "register_predictor",
        "format_model_id",
        "parse_model_id",
        "model_config_from_spec",
        "load_extensions",
        "list_zoo_names",
        "get_zoo_config",
        "list_predictor_metadata",
    }
    for name in expected:
        assert hasattr(components, name), name
