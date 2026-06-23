"""Tests for internal ModelConfig dataclasses."""

from dataclasses import asdict

from drevalpy.components.config import (
    FeaturizerConfig,
    ModelConfig,
    PredictionMode,
    PredictorConfig,
)


def test_featurizer_config_preserves_view_fields() -> None:
    cell_line = FeaturizerConfig(
        type="pca",
        registry="cell_line",
        view="gene_expression",
        hyperparameters={"n_components": 128},
    )
    drug = FeaturizerConfig(
        type="identity",
        registry="drug",
        view="fingerprints",
    )
    assert cell_line.view == "gene_expression"
    assert drug.view == "fingerprints"


def test_model_id_for_full_triple() -> None:
    config = ModelConfig(
        cell_line_featurizer=FeaturizerConfig(type="identity", registry="cell_line"),
        drug_featurizer=FeaturizerConfig(type="oneHot", registry="drug"),
        predictor=PredictorConfig(type="randomForest"),
    )
    assert config.model_id == "identity:oneHot:randomForest"


def test_model_id_for_predictor_only_baseline() -> None:
    config = ModelConfig(
        cell_line_featurizer=None,
        drug_featurizer=None,
        predictor=PredictorConfig(type="naiveMean"),
    )
    assert config.model_id == "naiveMean"


def test_model_id_none_for_partial_featurizer_config() -> None:
    config = ModelConfig(
        cell_line_featurizer=FeaturizerConfig(type="identity", registry="cell_line"),
        drug_featurizer=None,
        predictor=PredictorConfig(type="randomForest"),
    )
    assert config.model_id is None


def test_config_is_serializable() -> None:
    config = ModelConfig(
        cell_line_featurizer=FeaturizerConfig(type="identity", registry="cell_line", view="gene_expression"),
        drug_featurizer=FeaturizerConfig(type="identity", registry="drug", view="fingerprints"),
        predictor=PredictorConfig(type="elasticNet", hyperparameters={"alpha": 1.0}),
        prediction_mode=PredictionMode.REGRESSION,
    )
    payload = asdict(config)
    assert payload["cell_line_featurizer"]["view"] == "gene_expression"
    assert payload["predictor"]["hyperparameters"]["alpha"] == 1.0


def test_model_config_from_spec_classmethod() -> None:
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()
    config = ModelConfig.from_spec("NaivePredictor")
    assert config.predictor.type == "naiveMean"


def test_model_config_from_dict_classmethod() -> None:
    config = ModelConfig.from_dict({"predictor": {"type": "naiveMean"}})
    assert config.predictor.type == "naiveMean"
