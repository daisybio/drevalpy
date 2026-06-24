"""Tests for internal ModelConfig models."""

from drevalpy.components.config import (
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
            "methylationPCA": {"n_components": 64},
        }
    )
    assert config.name == "methylationPCA"
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
        drug_featurizer=FeaturizerConfig(name="oneHot", registry="drug"),
        predictor=PredictorConfig(type="randomForest"),
    )
    assert config.model_id == "scaledGeneExpression:oneHot:randomForest"


def test_model_id_for_predictor_only_baseline() -> None:
    config = ModelConfig(
        cell_line_featurizer=None,
        drug_featurizer=None,
        predictor=PredictorConfig(type="naiveMean"),
    )
    assert config.model_id == "naiveMean"


def test_model_id_none_for_partial_featurizer_config() -> None:
    config = ModelConfig(
        cell_line_featurizer=FeaturizerConfig(name="scaledGeneExpression", registry="cell_line"),
        drug_featurizer=None,
        predictor=PredictorConfig(type="randomForest"),
    )
    assert config.model_id is None


def test_model_config_parses_compact_featurizer_sections() -> None:
    config = ModelConfig.model_validate(
        {
            "cell_line_featurizer": {
                "concatFeaturizers": {
                    "featurizers": [
                        "scaledGeneExpression",
                        {"methylationPCA": {"n_components": 100}},
                        "mutations",
                    ],
                },
            },
            "drug_featurizer": "fingerprints",
            "predictor": {"type": "randomForest"},
        }
    )
    assert config.cell_line_featurizer is not None
    assert config.cell_line_featurizer.name == "concatFeaturizers"
    assert config.drug_featurizer is not None
    assert config.drug_featurizer.name == "fingerprints"


def test_config_is_serializable() -> None:
    config = ModelConfig(
        cell_line_featurizer=FeaturizerConfig(name="scaledGeneExpression", registry="cell_line"),
        drug_featurizer=FeaturizerConfig(name="fingerprints", registry="drug"),
        predictor=PredictorConfig(type="elasticNet", hyperparameters={"alpha": 1.0}),
        prediction_mode=PredictionMode.REGRESSION,
    )
    payload = config.model_dump(mode="python")
    assert payload["cell_line_featurizer"]["name"] == "scaledGeneExpression"
    assert payload["predictor"]["hyperparameters"]["alpha"] == 1.0


def test_model_config_from_spec_classmethod() -> None:
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()
    config = ModelConfig.from_spec("NaivePredictor")
    assert config.predictor.type == "naiveMean"


def test_model_config_from_dict_classmethod() -> None:
    config = ModelConfig.from_dict({"predictor": {"type": "naiveMean"}})
    assert config.predictor.type == "naiveMean"
