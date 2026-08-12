"""Tests for the built-in model zoo under :mod:`drevalpy.models.zoo`."""

from __future__ import annotations

import pytest

from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.models import construct_model
from drevalpy.models.config import ModelConfig, ModelScope, from_spec, validate
from drevalpy.models.factory import model_config_for_name
from drevalpy.models.zoo import get_zoo_config, list_zoo_names, zoo_model_config
from drevalpy.registry._builtins import register_builtin_components
from drevalpy.registry.predictor import get as get_predictor

LITERATURE_ZOO_NAMES = [
    "DrugGNN",
    "PharmaFormer",
    "DIPK",
    "MOLIR",
    "SuperFELTR",
    "Precily",
    "SRMF",
    "SimpleNeuralNetwork",
    "MultiViewNeuralNetwork",
    "SparseGO",
]

BLOCK_ZOO_NAMES = {"DrugGNN", "PharmaFormer", "DIPK", "MOLIR", "SuperFELTR", "Precily", "SRMF", "SparseGO"}


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


def test_builtin_zoo_lists_passing_models() -> None:
    names = list_zoo_names(include_external=False)
    assert "ElasticNet" in names
    assert "NaivePredictor" in names
    assert "DIPK" in names
    assert "PharmaFormer" in names
    assert "SingleDrugElasticNet" in names


def test_list_zoo_names_filters_by_scope() -> None:
    from drevalpy.types.enums.model_scope import ModelScope

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
    with pytest.raises(ValueError, match=r"Unknown hyperparameter"):
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


def test_get_zoo_config_applies_prediction_mode_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from drevalpy.types.enums.prediction_mode import PredictionMode

    monkeypatch.setattr(get_predictor("elasticNet"), "supported_modes", frozenset(PredictionMode))
    assert get_zoo_config("ElasticNet").prediction_mode == PredictionMode.REGRESSION
    overridden = get_zoo_config("ElasticNet", prediction_mode=PredictionMode.CLASSIFICATION)
    assert overridden.prediction_mode == PredictionMode.CLASSIFICATION
    via_zoo = zoo_model_config("ElasticNet", prediction_mode=PredictionMode.CLASSIFICATION)
    assert isinstance(via_zoo, ModelConfig)
    assert via_zoo.prediction_mode == PredictionMode.CLASSIFICATION


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

    preset = get_zoo_config("ElasticNet")
    with pytest.raises(ValidationError, match="requires a drug_featurizer"):
        ModelConfig(
            cell_line_featurizer=preset.cell_line_featurizer,
            drug_featurizer=None,
            predictor=preset.predictor,
            prediction_mode=preset.prediction_mode,
        )


@pytest.mark.parametrize("name", LITERATURE_ZOO_NAMES)
def test_literature_zoo_entries_validate(name: str) -> None:
    assert name in list_zoo_names(include_external=False)
    config = get_zoo_config(name)
    validate(config)
    assert config.cell_line_featurizer is not None
    if name in BLOCK_ZOO_NAMES:
        assert config.drug_featurizer is not None
        assert issubclass(get_predictor(config.predictor.name), BlockPredictor)


@pytest.mark.parametrize("name", LITERATURE_ZOO_NAMES)
def test_literature_zoo_entries_create_model(name: str) -> None:
    config = get_zoo_config(name)
    model = construct_model(name, config)()
    assert model is not None


@pytest.mark.parametrize("name", ["MOLIR", "SuperFELTR"])
def test_single_drug_zoo_entries_route_with_identity(name: str) -> None:
    config = get_zoo_config(name)
    assert config.drug_featurizer is not None
    assert config.drug_featurizer.name == "identity"
    validate(config)


def test_from_spec_resolves_literature_zoo() -> None:
    config = from_spec("DrugGNN")
    assert isinstance(config, ModelConfig)
    assert config.predictor.name == "drugGNN"
    assert config.cell_line_featurizer is not None
    assert config.drug_featurizer is not None


def test_single_drug_zoo_membership() -> None:
    single_names = list_zoo_names(include_external=False, scope=ModelScope.SINGLE_DRUG)
    assert set(single_names) == {
        "SingleDrugElasticNet",
        "SingleDrugRandomForest",
        "MOLIR",
        "SuperFELTR",
    }
    for name in single_names:
        model_class = construct_model(name)
        assert model_class.is_single_drug() is True
        assert get_zoo_config(name).scope == ModelScope.SINGLE_DRUG


def test_multi_drug_zoo_excludes_single_drug_scope() -> None:
    multi_names = list_zoo_names(include_external=False, scope=ModelScope.MULTI_DRUG)
    for name in multi_names:
        model_class = construct_model(name)
        assert model_class.is_single_drug() is False
        assert get_zoo_config(name).scope == ModelScope.MULTI_DRUG
