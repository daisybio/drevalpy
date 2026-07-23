"""Tests for internal ModelConfig models."""

from drevalpy.models.config import (
    FeaturizerConfig,
    ModelConfig,
    PredictionMode,
    PredictorConfig,
)


def test_featurizer_config_compact_string_shorthand() -> None:
    config = FeaturizerConfig.model_validate("fingerprints")
    assert config.name == "fingerprints"
    assert config.registry == "cell_line"


def test_featurizer_config_compact_one_key_mapping() -> None:
    config = FeaturizerConfig.model_validate(
        {
            "pca[methylation]": {"n_components": 64},
        }
    )
    assert config.name == "pca"
    assert config.view == "methylation"
    assert config.hyperparameters["n_components"] == 64


def test_featurizer_config_preserves_view_fields() -> None:
    cell_line = FeaturizerConfig(
        name="pca",
        registry="cell_line",
        view="gene_expression",
        hyperparameters={"n_components": 128},
    )
    assert cell_line.view == "gene_expression"


def test_model_id_for_full_triple() -> None:
    config = ModelConfig(
        cell_line_featurizer=FeaturizerConfig(name="scaledGeneExpression", registry="cell_line"),
        drug_featurizer=FeaturizerConfig(name="identity", registry="drug"),
        predictor=PredictorConfig(name="randomForest"),
    )
    assert config.model_id == "scaledGeneExpression:identity:randomForest"


def test_model_id_for_predictor_only_baseline() -> None:
    config = ModelConfig(
        cell_line_featurizer=None,
        drug_featurizer=None,
        predictor=PredictorConfig(name="naiveMean"),
    )
    assert config.model_id == "naiveMean"


def test_model_id_none_for_partial_featurizer_config() -> None:
    config = ModelConfig(
        cell_line_featurizer=FeaturizerConfig(name="scaledGeneExpression", registry="cell_line"),
        drug_featurizer=None,
        predictor=PredictorConfig(name="randomForest"),
    )
    assert config.model_id is None


def test_model_config_parses_compact_featurizer_sections() -> None:
    config = ModelConfig.model_validate(
        {
            "cell_line_featurizer": [
                "scaledGeneExpression",
                {"pca[methylation]": {"n_components": 100}},
                "raw[mutations]",
            ],
            "drug_featurizer": "fingerprints",
            "predictor": "randomForest",
        }
    )
    assert config.cell_line_featurizer is not None
    assert config.cell_line_featurizer.name == "concatFeaturizers"
    assert config.drug_featurizer is not None
    assert config.drug_featurizer.name == "fingerprints"
    assert config.predictor.name == "randomForest"


def test_model_config_parses_predictor_one_key_hyperparameters() -> None:
    config = ModelConfig.model_validate(
        {
            "cell_line_featurizer": "scaledGeneExpression",
            "drug_featurizer": "fingerprints",
            "predictor": {"randomForest": {"n_estimators": 10}},
        }
    )
    assert config.predictor.name == "randomForest"
    assert config.predictor.hyperparameters["n_estimators"] == 10


def test_config_is_serializable() -> None:
    config = ModelConfig(
        cell_line_featurizer=FeaturizerConfig(name="scaledGeneExpression", registry="cell_line"),
        drug_featurizer=FeaturizerConfig(name="fingerprints", registry="drug"),
        predictor=PredictorConfig(name="elasticNet", hyperparameters={"alpha": 1.0}),
        prediction_mode=PredictionMode.REGRESSION,
    )
    payload = config.model_dump(mode="python")
    assert payload["cell_line_featurizer"]["name"] == "scaledGeneExpression"
    assert payload["predictor"]["name"] == "elasticNet"
    assert payload["predictor"]["hyperparameters"]["alpha"] == 1.0


def test_model_config_from_spec_classmethod() -> None:
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()
    config = ModelConfig.from_spec("NaivePredictor")
    assert config.predictor.name == "naiveMean"


def test_model_config_from_dict_classmethod() -> None:
    config = ModelConfig.from_dict({"predictor": "naiveMean"})
    assert config.predictor.name == "naiveMean"
