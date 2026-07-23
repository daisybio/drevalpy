"""Tests for legacy flat-key compatibility helpers."""

from __future__ import annotations

from drevalpy.components.featurizer_config_parse import normalize_featurizer_config
from drevalpy.components.tuning.compatibility_keys import append_featurizer_flat_keys
from drevalpy.models.config import FeaturizerConfig, ModelConfig, PredictorConfig


def test_append_featurizer_flat_keys_exports_methylation_alias() -> None:
    config = ModelConfig(
        cell_line_featurizer=FeaturizerConfig.model_validate(
            normalize_featurizer_config({"pca[methylation]": {"n_components": 42}}, default_registry="cell_line"),
        ),
        drug_featurizer=None,
        predictor=PredictorConfig(name="naiveMean"),
    )
    flat: dict = {}
    append_featurizer_flat_keys(flat, config.cell_line_featurizer, "cell_line")
    assert flat["methylation_n_components"] == 42
    assert flat["methylation_pca_components"] == 42
