"""Tests for legacy flat-key compatibility helpers."""

from __future__ import annotations

from drevalpy.components.tuning.compatibility_keys import append_featurizer_flat_keys
from drevalpy.models.config import CellLineFeaturizerConfig, DrugFeaturizerConfig, ModelConfig, PredictorConfig
from drevalpy.models.config._featurizer_parse import normalize_featurizer_config


def test_append_featurizer_flat_keys_exports_methylation_alias() -> None:
    config = ModelConfig(
        cell_line_featurizer=CellLineFeaturizerConfig.model_validate(
            normalize_featurizer_config({"pca[methylation]": {"n_components": 42}}, default_registry="cell_line"),
        ),
        drug_featurizer=DrugFeaturizerConfig.model_validate(
            normalize_featurizer_config("fingerprints", default_registry="drug"),
        ),
        predictor=PredictorConfig(name="randomForest"),
    )
    flat: dict = {}
    append_featurizer_flat_keys(flat, config.cell_line_featurizer, "cell_line")
    assert flat["methylation_n_components"] == 42
    assert flat["methylation_pca_components"] == 42


def test_append_featurizer_flat_keys_skips_architecture_only_kwargs() -> None:
    config = ModelConfig(
        cell_line_featurizer=CellLineFeaturizerConfig.model_validate(
            normalize_featurizer_config(
                [{"name": "identity"}, {"name": "tissue", "options": {"allow_missing": True}}],
                default_registry="cell_line",
            ),
        ),
        drug_featurizer=DrugFeaturizerConfig.model_validate(
            normalize_featurizer_config("identity", default_registry="drug"),
        ),
        predictor=PredictorConfig(name="naiveMeanEffects"),
    )
    flat: dict = {}
    append_featurizer_flat_keys(flat, config.cell_line_featurizer, "cell_line")
    assert "allow_missing" not in flat
    assert config.cell_line_featurizer is not None
    children = config.cell_line_featurizer.featurizers
    assert children is not None
    assert children[1].options is not None
    assert children[1].options["allow_missing"] is True
