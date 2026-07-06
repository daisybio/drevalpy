"""Tests for featurizer-to-view mapping and identity-only loading."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from drevalpy.components.featurizer_config_parse import normalize_featurizer_config
from drevalpy.components.register_builtins import ensure_components_registered
from drevalpy.models.config import FeaturizerConfig, ModelConfig, PredictorConfig
from drevalpy.models.featurizer_mapping import (
    cell_line_entity_id_only_from_model_config,
    cell_line_views_from_model_config,
    drug_entity_id_only_from_model_config,
    drug_views_from_model_config,
)


@pytest.fixture(autouse=True)
def _register_components() -> None:
    ensure_components_registered()


def _model_config(**kwargs: object) -> ModelConfig:
    defaults: dict[str, object] = {
        "predictor": PredictorConfig(name="randomForest"),
    }
    defaults.update(kwargs)
    return ModelConfig.model_validate(defaults)


def test_identity_featurizers_resolve_to_empty_views() -> None:
    config = _model_config(
        cell_line_featurizer=FeaturizerConfig.model_validate(
            normalize_featurizer_config("identity", default_registry="cell_line"),
        ),
        drug_featurizer=FeaturizerConfig.model_validate(
            normalize_featurizer_config("identity", default_registry="drug"),
        ),
    )
    assert cell_line_entity_id_only_from_model_config(config)
    assert drug_entity_id_only_from_model_config(config)
    assert cell_line_views_from_model_config(config) == []
    assert drug_views_from_model_config(config) == []


def test_fingerprint_featurizer_still_resolves_fingerprints_view() -> None:
    config = _model_config(
        drug_featurizer=FeaturizerConfig.model_validate(
            normalize_featurizer_config("fingerprints", default_registry="drug"),
        ),
    )
    assert not drug_entity_id_only_from_model_config(config)
    assert drug_views_from_model_config(config) == ["fingerprints"]


def test_identity_drug_loading_uses_drug_ids_not_fingerprints() -> None:
    from drevalpy.components.data_loading import load_drug_features_for_model_config

    config = _model_config(
        drug_featurizer=FeaturizerConfig.model_validate(
            normalize_featurizer_config("identity", default_registry="drug"),
        ),
    )
    with patch("drevalpy.components.data_loading.load_drug_ids_from_csv") as load_ids:
        with patch("drevalpy.components.data_loading.load_drug_feature_views") as load_views:
            load_drug_features_for_model_config(config, "/data", "GDSC1")
    load_ids.assert_called_once_with("/data", "GDSC1")
    load_views.assert_not_called()


def test_no_drug_featurizer_skips_drug_loading() -> None:
    from drevalpy.components.data_loading import load_drug_features_for_model_config

    config = _model_config()
    with patch("drevalpy.components.data_loading.load_drug_ids_from_csv") as load_ids:
        with patch("drevalpy.components.data_loading.load_drug_feature_views") as load_views:
            result = load_drug_features_for_model_config(config, "/data", "GDSC1")
    assert result is None
    load_ids.assert_not_called()
    load_views.assert_not_called()
