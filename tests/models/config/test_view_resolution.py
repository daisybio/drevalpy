"""Tests for featurizer-to-view mapping and identity-only loading."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from drevalpy.models.config import DrugFeaturizerConfig, ModelConfig, PredictorConfig, SampleFeaturizerConfig


def _model_config(**kwargs: object) -> ModelConfig:
    defaults: dict[str, object] = {
        "predictor": PredictorConfig(name="randomForest"),
        "sample_featurizer": SampleFeaturizerConfig(name="scaledGeneExpression"),
        "drug_featurizer": DrugFeaturizerConfig(name="fingerprints"),
    }
    defaults.update(kwargs)
    return ModelConfig.model_validate(defaults)


@pytest.mark.parametrize("name", ["identity", "constant"])
def test_id_only_featurizers_resolve_to_empty_views(name: str) -> None:
    config = _model_config(
        sample_featurizer=SampleFeaturizerConfig.model_validate(name),
        drug_featurizer=DrugFeaturizerConfig.model_validate(name),
    )
    assert config.sample_entity_id_only()
    assert config.drug_entity_id_only()
    assert config.sample_views() == []
    assert config.drug_views() == []


@pytest.mark.parametrize(
    ("featurizer", "views"),
    [
        (SampleFeaturizerConfig.model_validate("raw[mutations]+pca[methylation]"), ["mutations", "methylation"]),
        (SampleFeaturizerConfig.model_validate("landmarkGenes"), ["gene_expression"]),
        (SampleFeaturizerConfig.model_validate("landmarkGenesReduced"), ["gene_expression"]),
        (
            SampleFeaturizerConfig.model_validate("molirOmics"),
            ["gene_expression", "mutations", "copy_number_variation_gistic"],
        ),
        (
            SampleFeaturizerConfig.model_validate("superfeltrOmics"),
            ["gene_expression", "mutations", "copy_number_variation_gistic"],
        ),
        (SampleFeaturizerConfig(name="sparsegoOntology", options={"input_type": "expression"}), ["gene_expression"]),
        (SampleFeaturizerConfig(name="sparsegoOntology", options={"input_type": "mutations"}), ["mutations"]),
        (SampleFeaturizerConfig.model_validate("tissue"), []),
        (SampleFeaturizerConfig(name="scaledGeneExpression", options={"view": "proteomics"}), ["proteomics"]),
    ],
    ids=[
        "brackets",
        "landmark",
        "landmark-reduced",
        "molir",
        "superfeltr",
        "sparsego-expression",
        "sparsego-mutations",
        "tissue",
        "view-override",
    ],
)
def test_sample_views(featurizer, views) -> None:
    assert _model_config(sample_featurizer=featurizer).sample_views() == views


def test_drug_views() -> None:
    config = _model_config()
    assert not config.drug_entity_id_only()
    assert config.drug_views() == ["morgan_fingerprint"]

    view = _model_config(
        drug_featurizer=DrugFeaturizerConfig(name="view", options={"view": "drug_chemberta_embeddings"})
    )
    assert view.drug_views() == ["drug_chemberta_embeddings"]


def test_feature_based_predictor_requires_a_drug_featurizer() -> None:
    with pytest.raises(ValidationError, match="requires a drug_featurizer"):
        _model_config(predictor=PredictorConfig(name="naiveSampleMean"), drug_featurizer=None)
