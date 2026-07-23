"""Tests for built-in model zoo under drevalpy.models."""

from __future__ import annotations

import pytest

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


def test_zoo_naive_has_identity_featurizers() -> None:
    zoo_config = get_zoo_config("NaivePredictor")
    assert zoo_config.predictor.name == "naiveMean"
    assert zoo_config.cell_line_featurizer is not None
    assert zoo_config.cell_line_featurizer.name == "identity"
    assert zoo_config.drug_featurizer is not None
    assert zoo_config.drug_featurizer.name == "identity"


def test_zoo_model_config_merges_hyperparameters() -> None:
    config = zoo_model_config("ElasticNet", {"alpha": 0.25})
    assert config.predictor.hyperparameters["alpha"] == 0.25


def test_zoo_model_config_does_not_leak_view_keys_into_predictor() -> None:
    config = zoo_model_config(
        "ElasticNet",
        {"cell_line_views": ["gene_expression"], "alpha": 0.1},
    )
    assert config.cell_line_featurizer is not None
    assert config.cell_line_featurizer.name == "scaledGeneExpression"
    assert "cell_line_views" not in config.predictor.hyperparameters
    assert config.predictor.hyperparameters["alpha"] == 0.1


def test_zoo_model_config_routes_methylation_flat_key_to_pca_child() -> None:
    from drevalpy.components.featurizer_config_parse import normalize_featurizer_config
    from drevalpy.models.config import FeaturizerConfig

    config = zoo_model_config("MultiViewRandomForest", {"methylation_n_components": 11})
    assert "methylation_n_components" not in config.predictor.hyperparameters
    assert config.cell_line_featurizer is not None
    for child in config.cell_line_featurizer.hyperparameters.get("featurizers", []):
        child_cfg = FeaturizerConfig.model_validate(
            normalize_featurizer_config(child, default_registry="cell_line"),
        )
        if child_cfg.name == "pca" and child_cfg.view == "methylation":
            assert child_cfg.hyperparameters["n_components"] == 11
            return
    raise AssertionError("methylation PCA child not found")


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
    config = model_config_for_name("ElasticNet", {"alpha": 0.5})
    assert config.predictor.name == "elasticNet"
    assert config.predictor.hyperparameters["alpha"] == 0.5


def test_single_drug_sklearn_zoo_entries_use_optional_drug_predictors() -> None:
    elastic_net = get_zoo_config("SingleDrugElasticNet")
    random_forest = get_zoo_config("SingleDrugRandomForest")

    assert elastic_net.predictor.name == "singleDrugElasticNet"
    assert random_forest.predictor.name == "singleDrugRandomForest"
    assert elastic_net.drug_featurizer is None
    assert random_forest.drug_featurizer is None
    assert elastic_net.scope.value == "single_drug"
    elastic_net.validate()
    random_forest.validate()


def test_multi_drug_sklearn_predictor_without_drug_featurizer_fails() -> None:
    config = get_zoo_config("ElasticNet").model_copy(update={"drug_featurizer": None})

    with pytest.raises(ValueError, match="requires a drug_featurizer"):
        config.validate()
