"""Tests for built-in model zoo under drevalpy.models."""

from __future__ import annotations

import pytest

from drevalpy.models.config import validate
from drevalpy.models.factory import model_config_for_name
from drevalpy.models.zoo import get_zoo_config, list_zoo_names, zoo_model_config


def test_builtin_zoo_lists_passing_models() -> None:
    names = list_zoo_names(include_external=False)
    assert "ElasticNet" in names
    assert "NaivePredictor" in names
    assert "DIPK" in names
    assert "PharmaFormer" in names
    assert "SingleDrugElasticNet" in names


def test_list_zoo_names_filters_by_scope() -> None:
    from drevalpy.types.model_scope import ModelScope

    single = list_zoo_names(include_external=False, scope=ModelScope.SINGLE_DRUG)
    multi = list_zoo_names(include_external=False, scope="multi_drug")
    assert "SingleDrugElasticNet" in single
    assert "ElasticNet" in multi
    assert set(single).isdisjoint(multi)


def test_zoo_elastic_net_defaults() -> None:
    zoo_config = get_zoo_config("ElasticNet")
    assert zoo_config.cell_line_featurizer is not None
    assert zoo_config.cell_line_featurizer.name == "scaledGeneExpression"
    assert zoo_config.drug_featurizer is not None
    assert zoo_config.drug_featurizer.name == "fingerprints"
    assert zoo_config.predictor.name == "elasticNet"


def test_zoo_naive_presets_use_information_accurate_featurizers() -> None:
    naive = get_zoo_config("NaivePredictor")
    assert naive.predictor.name == "naiveMean"
    assert naive.cell_line_featurizer is None
    assert naive.drug_featurizer is None

    cell_mean = get_zoo_config("NaiveCellLineMeanPredictor")
    assert cell_mean.cell_line_featurizer is not None
    assert cell_mean.cell_line_featurizer.name == "identity"
    assert cell_mean.drug_featurizer is not None
    assert cell_mean.drug_featurizer.name == "constant"

    drug_mean = get_zoo_config("NaiveDrugMeanPredictor")
    assert drug_mean.cell_line_featurizer is not None
    assert drug_mean.cell_line_featurizer.name == "constant"
    assert drug_mean.drug_featurizer is not None
    assert drug_mean.drug_featurizer.name == "identity"

    tissue_mean = get_zoo_config("NaiveTissueMeanPredictor")
    assert tissue_mean.cell_line_featurizer is not None
    assert tissue_mean.cell_line_featurizer.name == "tissue"
    assert tissue_mean.drug_featurizer is not None
    assert tissue_mean.drug_featurizer.name == "constant"

    tissue_drug = get_zoo_config("NaiveTissueDrugMeanPredictor")
    assert tissue_drug.cell_line_featurizer is not None
    assert tissue_drug.cell_line_featurizer.name == "tissue"
    assert tissue_drug.drug_featurizer is not None
    assert tissue_drug.drug_featurizer.name == "identity"

    mean_effects = get_zoo_config("NaiveMeanEffectsPredictor")
    assert mean_effects.cell_line_featurizer is not None
    assert mean_effects.cell_line_featurizer.name == "concatFeaturizers"
    children = mean_effects.cell_line_featurizer.featurizers
    assert children is not None
    assert [child.name for child in children] == ["identity", "tissue"]
    assert children[1].options is not None
    assert children[1].options["allow_missing"] is True
    assert mean_effects.drug_featurizer is not None
    assert mean_effects.drug_featurizer.name == "identity"


def test_zoo_model_config_merges_hyperparameters() -> None:
    from drevalpy.models.config import ResolvedModelConfig

    config = zoo_model_config("ElasticNet", {"alpha": 0.25})
    assert isinstance(config, ResolvedModelConfig)
    assert config.predictor_values()["alpha"] == 0.25


def test_zoo_model_config_rejects_view_keys() -> None:
    with pytest.raises(ValueError, match=r"Legacy view keys|no longer supported"):
        zoo_model_config(
            "ElasticNet",
            {"cell_line_views": ["gene_expression"], "alpha": 0.1},
        )


def test_zoo_model_config_routes_methylation_flat_key_to_pca_child() -> None:
    from drevalpy.models.config import ResolvedModelConfig

    resolved = zoo_model_config("MultiViewRandomForest", {"methylation_n_components": 11})
    assert isinstance(resolved, ResolvedModelConfig)
    assert resolved.predictor_values().get("methylation_n_components") is None
    assert resolved.featurizer_values("cell_line", "pca[methylation]")["n_components"] == 11


def test_external_zoo_rejects_builtin_collision_and_is_atomic(tmp_path) -> None:
    from drevalpy.models.zoo import clear_external_zoo, load_external_zoo_file

    clear_external_zoo()
    bad = tmp_path / "zoo.yaml"
    bad.write_text(
        """
goodEntry:
  cell_line_featurizer: scaledGeneExpression
  drug_featurizer: fingerprints
  predictor: elasticNet
ElasticNet:
  cell_line_featurizer: scaledGeneExpression
  drug_featurizer: fingerprints
  predictor: elasticNet
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="collides with a built-in"):
        load_external_zoo_file(bad)
    assert "goodEntry" not in list_zoo_names(include_external=True)
    clear_external_zoo()


def test_model_config_for_name_uses_zoo_entry() -> None:
    from drevalpy.models.config import ResolvedModelConfig

    config = model_config_for_name("ElasticNet", {"alpha": 0.5})
    assert isinstance(config, ResolvedModelConfig)
    assert config.template.predictor.name == "elasticNet"
    assert config.predictor_values()["alpha"] == 0.5


def test_single_drug_sklearn_zoo_entries_use_identity_for_routing() -> None:
    elastic_net = get_zoo_config("SingleDrugElasticNet")
    random_forest = get_zoo_config("SingleDrugRandomForest")

    assert elastic_net.predictor.name == "singleDrugElasticNet"
    assert random_forest.predictor.name == "singleDrugRandomForest"
    assert elastic_net.drug_featurizer is not None
    assert random_forest.drug_featurizer is not None
    assert elastic_net.drug_featurizer.name == "identity"
    assert random_forest.drug_featurizer.name == "identity"
    assert elastic_net.model_id == "scaledGeneExpression:singleDrugElasticNet"
    assert elastic_net.scope.value == "single_drug"
    validate(elastic_net)
    validate(random_forest)


def test_multi_drug_sklearn_predictor_without_drug_featurizer_fails() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="requires a drug_featurizer"):
        get_zoo_config("ElasticNet").replace(drug_featurizer=None)
