"""Tests for drevalpy.models.config.model."""

import pytest
from pydantic import ValidationError

from drevalpy.models.config import (
    DrugFeaturizerConfig,
    FeaturizerConfig,
    ModelConfig,
    ModelScope,
    PredictionMode,
    PredictorConfig,
    SampleFeaturizerConfig,
    from_dict,
    from_spec,
)

SAMPLE = SampleFeaturizerConfig(name="scaledGeneExpression")


def test_featurizer_config_compact_forms() -> None:
    config = FeaturizerConfig.model_validate("fingerprints")
    assert (config.name, config.registry) == ("fingerprints", "sample")

    pca = FeaturizerConfig.model_validate({"pca[methylation]": {"n_components": 64}})
    assert (pca.name, pca.view) == ("pca", "methylation")
    assert pca.hyperparameter_space is not None
    assert pca.hyperparameter_space["n_components"]["default"] == 64


def test_slot_subclasses_fix_the_registry() -> None:
    sample = SampleFeaturizerConfig.model_validate({"name": "scaledGeneExpression", "registry": "drug"})
    drug = DrugFeaturizerConfig.model_validate({"name": "fingerprints", "registry": "sample"})

    assert (sample.registry, drug.registry) == ("sample", "drug")
    assert isinstance(sample, FeaturizerConfig)
    assert isinstance(drug, FeaturizerConfig)


@pytest.mark.parametrize(
    ("drug", "predictor", "model_id"),
    [
        (DrugFeaturizerConfig(name="identity"), "randomForest", "scaledGeneExpression:identity:randomForest"),
        (None, "singleDrugElasticNet", "scaledGeneExpression:singleDrugElasticNet"),
    ],
    ids=["full-triple", "implicit-identity-single-drug"],
)
def test_model_id(drug, predictor, model_id) -> None:
    config = ModelConfig(sample_featurizer=SAMPLE, drug_featurizer=drug, predictor=PredictorConfig(name=predictor))

    assert config.model_id == model_id
    if drug is None:
        assert config.drug_featurizer is not None
        assert config.drug_featurizer.name == "identity"


def test_model_id_for_predictor_only_baseline() -> None:
    config = ModelConfig(sample_featurizer=None, drug_featurizer=None, predictor=PredictorConfig(name="naiveMean"))

    assert config.model_id == "naiveMean"


@pytest.mark.parametrize(
    ("drug", "predictor", "match"),
    [
        (None, "randomForest", "requires.*drug_featurizer|requires featurizers"),
        (None, "elasticNet", "requires.*drug_featurizer|requires featurizers"),
        (DrugFeaturizerConfig(name="fingerprints"), "singleDrugElasticNet", "requires drug_featurizer='identity'"),
    ],
    ids=["partial-multi-drug", "multi-drug-gets-no-identity", "single-drug-keeps-explicit-drug"],
)
def test_inconsistent_featurizer_slots_are_rejected(drug, predictor, match) -> None:
    with pytest.raises(ValidationError, match=match):
        ModelConfig(sample_featurizer=SAMPLE, drug_featurizer=drug, predictor=PredictorConfig(name=predictor))


def test_scope_is_derived_from_the_predictor_and_never_stated() -> None:
    payload = {"sample_featurizer": "scaledGeneExpression", "predictor": "singleDrugElasticNet"}

    config = ModelConfig.model_validate(payload)

    assert config.scope is ModelScope.SINGLE_DRUG
    assert "scope" not in config.model_dump()
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ModelConfig.model_validate({**payload, "scope": "single_drug"})


def test_model_config_parses_compact_sections() -> None:
    config = ModelConfig.model_validate(
        {
            "sample_featurizer": [
                "scaledGeneExpression",
                {"pca[methylation]": {"n_components": 100}},
                "raw[mutations]",
            ],
            "drug_featurizer": "fingerprints",
            "predictor": {"randomForest": {"n_estimators": 10}},
        }
    )

    assert isinstance(config.sample_featurizer, SampleFeaturizerConfig)
    assert config.sample_featurizer.name == "concatFeaturizers"
    assert isinstance(config.drug_featurizer, DrugFeaturizerConfig)
    assert config.drug_featurizer.name == "fingerprints"
    assert config.predictor.name == "randomForest"
    assert config.predictor.hyperparameter_space is not None
    assert config.predictor.hyperparameter_space["n_estimators"]["default"] == 10


def test_model_config_rejects_base_featurizer_config_in_slots() -> None:
    with pytest.raises(ValidationError):
        ModelConfig.model_validate(
            {
                "sample_featurizer": FeaturizerConfig(name="scaledGeneExpression", registry="drug"),
                "drug_featurizer": FeaturizerConfig(name="fingerprints", registry="sample"),
                "predictor": PredictorConfig(name="elasticNet"),
            }
        )


def test_config_is_serializable() -> None:
    config = ModelConfig(
        sample_featurizer=SAMPLE,
        drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
        predictor=PredictorConfig(
            name="elasticNet",
            hyperparameter_space={"alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True, "default": 1.0}},
        ),
        prediction_mode=PredictionMode.REGRESSION,
    )
    payload = config.model_dump(mode="python")
    assert payload["sample_featurizer"]["name"] == "scaledGeneExpression"
    assert payload["predictor"]["hyperparameter_space"]["alpha"]["default"] == 1.0


def test_from_spec_and_from_dict() -> None:
    assert from_spec("NaivePredictor").predictor.name == "naiveMean"
    assert from_dict({"predictor": "naiveMean"}).predictor.name == "naiveMean"
