"""Tests for drevalpy.models.config.model."""

import pytest
from pydantic import ValidationError

from drevalpy.models.config import (
    CellLineFeaturizerConfig,
    DrugFeaturizerConfig,
    FeaturizerConfig,
    ModelConfig,
    ModelScope,
    PredictionMode,
    PredictorConfig,
    from_dict,
    from_spec,
)


def test_featurizer_config_compact_string_shorthand() -> None:
    config = FeaturizerConfig.model_validate("fingerprints")
    assert config.name == "fingerprints"
    assert config.registry == "cell_line"


def test_cell_line_and_drug_featurizer_configs_fix_registry() -> None:
    cell = CellLineFeaturizerConfig(name="scaledGeneExpression")
    drug = DrugFeaturizerConfig(name="fingerprints")
    assert cell.registry == "cell_line"
    assert drug.registry == "drug"
    assert isinstance(cell, FeaturizerConfig)
    assert isinstance(drug, FeaturizerConfig)


def test_slot_subclasses_override_mismatched_registry() -> None:
    cell = CellLineFeaturizerConfig.model_validate(
        {"name": "scaledGeneExpression", "registry": "drug"},
    )
    drug = DrugFeaturizerConfig.model_validate(
        {"name": "fingerprints", "registry": "cell_line"},
    )
    assert cell.registry == "cell_line"
    assert drug.registry == "drug"


def test_featurizer_config_compact_one_key_mapping() -> None:
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()
    config = FeaturizerConfig.model_validate(
        {
            "pca[methylation]": {"n_components": 64},
        }
    )
    assert config.name == "pca"
    assert config.view == "methylation"
    assert config.hyperparameter_space is not None
    assert config.hyperparameter_space["n_components"]["default"] == 64


def test_featurizer_config_preserves_view_fields() -> None:
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()
    cell_line = CellLineFeaturizerConfig(
        name="pca",
        view="gene_expression",
        hyperparameter_space={"n_components": {"type": "int", "low": 8, "high": 512, "default": 128}},
    )
    assert cell_line.view == "gene_expression"


def test_model_id_for_full_triple() -> None:
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()
    config = ModelConfig(
        cell_line_featurizer=CellLineFeaturizerConfig(name="scaledGeneExpression"),
        drug_featurizer=DrugFeaturizerConfig(name="identity"),
        predictor=PredictorConfig(name="randomForest"),
    )
    assert config.model_id == "scaledGeneExpression:identity:randomForest"


def test_model_id_for_predictor_only_baseline() -> None:
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()
    config = ModelConfig(
        cell_line_featurizer=None,
        drug_featurizer=None,
        predictor=PredictorConfig(name="naiveMean"),
    )
    assert config.model_id == "naiveMean"


def test_model_id_none_for_partial_multi_drug_config() -> None:
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()
    with pytest.raises(ValidationError, match="requires.*drug_featurizer|requires featurizers"):
        ModelConfig(
            cell_line_featurizer=CellLineFeaturizerConfig(name="scaledGeneExpression"),
            drug_featurizer=None,
            predictor=PredictorConfig(name="randomForest"),
        )


def test_model_id_for_implicit_identity_single_drug() -> None:
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()
    config = ModelConfig(
        cell_line_featurizer=CellLineFeaturizerConfig(name="scaledGeneExpression"),
        drug_featurizer=None,
        predictor=PredictorConfig(name="singleDrugElasticNet"),
        scope=ModelScope.SINGLE_DRUG,
    )
    assert config.drug_featurizer is not None
    assert config.drug_featurizer.name == "identity"
    assert config.model_id == "scaledGeneExpression:singleDrugElasticNet"


def test_single_drug_does_not_override_explicit_drug_featurizer() -> None:
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()
    with pytest.raises(ValidationError, match="requires drug_featurizer='identity'"):
        ModelConfig(
            cell_line_featurizer=CellLineFeaturizerConfig(name="scaledGeneExpression"),
            drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
            predictor=PredictorConfig(name="singleDrugElasticNet"),
            scope=ModelScope.SINGLE_DRUG,
        )


def test_multi_drug_scope_does_not_inject_identity() -> None:
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()
    with pytest.raises(ValidationError, match="requires.*drug_featurizer|requires featurizers"):
        ModelConfig(
            cell_line_featurizer=CellLineFeaturizerConfig(name="scaledGeneExpression"),
            drug_featurizer=None,
            predictor=PredictorConfig(name="elasticNet"),
            scope=ModelScope.MULTI_DRUG,
        )


def test_string_scope_from_yaml_coerces_to_model_scope() -> None:
    """YAML leaves scope as str; normalization must still yield ModelScope."""
    from drevalpy.components.register_builtins import register_builtin_components
    from drevalpy.models.config import ModelScope

    register_builtin_components()
    config = ModelConfig.model_validate(
        {
            "cell_line_featurizer": "scaledGeneExpression",
            "predictor": "singleDrugElasticNet",
            "scope": "single_drug",
        }
    )
    assert config.scope == ModelScope.SINGLE_DRUG
    assert isinstance(config.scope, ModelScope)
    assert config.drug_featurizer is not None
    assert config.drug_featurizer.name == "identity"


def test_model_config_parses_compact_featurizer_sections() -> None:
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()
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
    assert isinstance(config.cell_line_featurizer, CellLineFeaturizerConfig)
    assert config.cell_line_featurizer.name == "concatFeaturizers"
    assert config.drug_featurizer is not None
    assert isinstance(config.drug_featurizer, DrugFeaturizerConfig)
    assert config.drug_featurizer.name == "fingerprints"
    assert config.predictor.name == "randomForest"


def test_model_config_parses_predictor_one_key_hyperparameters() -> None:
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()
    config = ModelConfig.model_validate(
        {
            "cell_line_featurizer": "scaledGeneExpression",
            "drug_featurizer": "fingerprints",
            "predictor": {"randomForest": {"n_estimators": 10}},
        }
    )
    assert config.predictor.name == "randomForest"
    assert config.predictor.hyperparameter_space is not None
    assert config.predictor.hyperparameter_space["n_estimators"]["default"] == 10


def test_model_config_rejects_base_featurizer_config_in_slots() -> None:
    with pytest.raises(ValidationError):
        ModelConfig.model_validate(
            {
                "cell_line_featurizer": FeaturizerConfig(name="scaledGeneExpression", registry="drug"),
                "drug_featurizer": FeaturizerConfig(name="fingerprints", registry="cell_line"),
                "predictor": PredictorConfig(name="elasticNet"),
            }
        )


def test_config_is_serializable() -> None:
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()
    config = ModelConfig(
        cell_line_featurizer=CellLineFeaturizerConfig(name="scaledGeneExpression"),
        drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
        predictor=PredictorConfig(
            name="elasticNet",
            hyperparameter_space={"alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True, "default": 1.0}},
        ),
        prediction_mode=PredictionMode.REGRESSION,
    )
    payload = config.model_dump(mode="python")
    assert payload["cell_line_featurizer"]["name"] == "scaledGeneExpression"
    assert payload["predictor"]["name"] == "elasticNet"
    assert payload["predictor"]["hyperparameter_space"]["alpha"]["default"] == 1.0


def test_from_spec_classmethod() -> None:
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()
    config = from_spec("NaivePredictor")
    assert config.predictor.name == "naiveMean"


def test_from_dict_classmethod() -> None:
    config = from_dict({"predictor": "naiveMean"})
    assert config.predictor.name == "naiveMean"
