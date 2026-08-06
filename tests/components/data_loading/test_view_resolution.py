"""Tests for featurizer-to-view mapping and identity-only loading."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from drevalpy.components.data_loading.view_resolution import (
    cell_line_entity_id_only_from_model_config,
    cell_line_views_from_model_config,
    drug_entity_id_only_from_model_config,
    drug_views_from_model_config,
)
from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.models.config import CellLineFeaturizerConfig, DrugFeaturizerConfig, ModelConfig, PredictorConfig


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


def _model_config(**kwargs: object) -> ModelConfig:
    defaults: dict[str, object] = {
        "predictor": PredictorConfig(name="randomForest"),
        "cell_line_featurizer": CellLineFeaturizerConfig(name="scaledGeneExpression"),
        "drug_featurizer": DrugFeaturizerConfig(name="fingerprints"),
    }
    defaults.update(kwargs)
    return ModelConfig.model_validate(defaults)


def test_identity_featurizers_resolve_to_empty_views() -> None:
    config = _model_config(
        cell_line_featurizer=CellLineFeaturizerConfig.model_validate("identity"),
        drug_featurizer=DrugFeaturizerConfig.model_validate("identity"),
    )
    assert cell_line_entity_id_only_from_model_config(config)
    assert drug_entity_id_only_from_model_config(config)
    assert cell_line_views_from_model_config(config) == []
    assert drug_views_from_model_config(config) == []


def test_constant_featurizers_resolve_to_empty_views() -> None:
    config = _model_config(
        cell_line_featurizer=CellLineFeaturizerConfig.model_validate("constant"),
        drug_featurizer=DrugFeaturizerConfig.model_validate("constant"),
    )
    assert cell_line_entity_id_only_from_model_config(config)
    assert drug_entity_id_only_from_model_config(config)
    assert cell_line_views_from_model_config(config) == []
    assert drug_views_from_model_config(config) == []


def test_bracket_featurizers_resolve_canonical_views() -> None:
    config = _model_config(
        cell_line_featurizer=CellLineFeaturizerConfig.model_validate("raw[mutations]+pca[methylation]"),
    )
    assert cell_line_views_from_model_config(config) == [
        "mutations",
        "methylation",
    ]


@pytest.mark.parametrize("name", ["landmarkGenes", "landmarkGenesReduced"])
def test_landmark_featurizers_resolve_gene_expression(name: str) -> None:
    config = _model_config(
        cell_line_featurizer=CellLineFeaturizerConfig.model_validate(name),
    )
    assert cell_line_views_from_model_config(config) == ["gene_expression"]


def test_fingerprint_featurizer_still_resolves_fingerprints_view() -> None:
    config = _model_config(
        drug_featurizer=DrugFeaturizerConfig.model_validate("fingerprints"),
    )
    assert not drug_entity_id_only_from_model_config(config)
    assert drug_views_from_model_config(config) == ["fingerprints"]


def test_view_featurizer_resolves_options_view() -> None:
    config = _model_config(
        drug_featurizer=DrugFeaturizerConfig(
            name="view",
            options={"view": "drug_chemberta_embeddings"},
        ),
    )
    assert drug_views_from_model_config(config) == ["drug_chemberta_embeddings"]


def test_identity_drug_loading_uses_drug_ids_not_fingerprints() -> None:
    from drevalpy.components.data_loading import load_drug_features_for_model_config

    config = _model_config(
        drug_featurizer=DrugFeaturizerConfig.model_validate("identity"),
    )
    with patch("drevalpy.components.data_loading.feature_loaders.load_drug_ids_from_csv") as load_ids:
        with patch("drevalpy.components.data_loading.feature_loaders.load_drug_feature_views") as load_views:
            load_drug_features_for_model_config(config, "/data", "GDSC1")
    load_ids.assert_called_once_with("/data", "GDSC1")
    load_views.assert_not_called()


def test_no_drug_featurizer_skips_drug_loading() -> None:
    from drevalpy.components.data_loading import load_drug_features_for_model_config

    config = _model_config(
        predictor=PredictorConfig(name="naiveCellLineMean"),
        drug_featurizer=None,
    )
    with patch("drevalpy.components.data_loading.feature_loaders.load_drug_ids_from_csv") as load_ids:
        with patch("drevalpy.components.data_loading.feature_loaders.load_drug_feature_views") as load_views:
            result = load_drug_features_for_model_config(config, "/data", "GDSC1")
    assert result is None
    load_ids.assert_not_called()
    load_views.assert_not_called()
