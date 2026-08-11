"""Tests for featurizer-to-view mapping and identity-only loading."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from drevalpy.models.config import CellLineFeaturizerConfig, DrugFeaturizerConfig, ModelConfig, PredictorConfig
from drevalpy.registry._builtins import register_builtin_components


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
    assert config.cell_line_entity_id_only()
    assert config.drug_entity_id_only()
    assert config.cell_line_views() == []
    assert config.drug_views() == []


def test_constant_featurizers_resolve_to_empty_views() -> None:
    config = _model_config(
        cell_line_featurizer=CellLineFeaturizerConfig.model_validate("constant"),
        drug_featurizer=DrugFeaturizerConfig.model_validate("constant"),
    )
    assert config.cell_line_entity_id_only()
    assert config.drug_entity_id_only()
    assert config.cell_line_views() == []
    assert config.drug_views() == []


def test_bracket_featurizers_resolve_canonical_views() -> None:
    config = _model_config(
        cell_line_featurizer=CellLineFeaturizerConfig.model_validate("raw[mutations]+pca[methylation]"),
    )
    assert config.cell_line_views() == [
        "mutations",
        "methylation",
    ]


@pytest.mark.parametrize("name", ["landmarkGenes", "landmarkGenesReduced"])
def test_landmark_featurizers_resolve_gene_expression(name: str) -> None:
    config = _model_config(
        cell_line_featurizer=CellLineFeaturizerConfig.model_validate(name),
    )
    assert config.cell_line_views() == ["gene_expression"]


@pytest.mark.parametrize("name", ["molirOmics", "superfeltrOmics"])
def test_multi_omics_featurizers_resolve_all_three_views(name: str) -> None:
    config = _model_config(
        cell_line_featurizer=CellLineFeaturizerConfig.model_validate(name),
    )
    assert config.cell_line_views() == [
        "gene_expression",
        "mutations",
        "copy_number_variation_gistic",
    ]


@pytest.mark.parametrize(
    ("input_type", "expected"),
    [("expression", "gene_expression"), ("mutations", "mutations")],
)
def test_sparsego_resolves_view_from_input_type(input_type: str, expected: str) -> None:
    config = _model_config(
        cell_line_featurizer=CellLineFeaturizerConfig(
            name="sparsegoOntology",
            options={"input_type": input_type},
        ),
    )
    assert config.cell_line_views() == [expected]


def test_tissue_featurizer_resolves_no_omics_views() -> None:
    config = _model_config(
        cell_line_featurizer=CellLineFeaturizerConfig.model_validate("tissue"),
    )
    assert config.cell_line_views() == []


def test_view_override_is_honoured_over_declared_input_views() -> None:
    config = _model_config(
        cell_line_featurizer=CellLineFeaturizerConfig(
            name="scaledGeneExpression",
            options={"view": "proteomics"},
        ),
    )
    assert config.cell_line_views() == ["proteomics"]


def test_fingerprint_featurizer_still_resolves_fingerprints_view() -> None:
    config = _model_config(
        drug_featurizer=DrugFeaturizerConfig.model_validate("fingerprints"),
    )
    assert not config.drug_entity_id_only()
    assert config.drug_views() == ["morgan_fingerprint"]


def test_view_featurizer_resolves_options_view() -> None:
    config = _model_config(
        drug_featurizer=DrugFeaturizerConfig(
            name="view",
            options={"view": "drug_chemberta_embeddings"},
        ),
    )
    assert config.drug_views() == ["drug_chemberta_embeddings"]


def test_feature_based_predictor_requires_a_drug_featurizer() -> None:
    with pytest.raises(ValidationError, match="requires a drug_featurizer"):
        _model_config(
            predictor=PredictorConfig(name="naiveCellLineMean"),
            drug_featurizer=None,
        )
