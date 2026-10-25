"""The built-in model zoo and external zoo YAML loading under :mod:`drevalpy.models.zoo`."""

from __future__ import annotations

import shutil

import pytest
from pydantic import ValidationError

from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.errors import UnknownNameError
from drevalpy.models import DRPModel, ModelSpec, zoo
from drevalpy.models.config import ModelConfig, ResolvedModelConfig, validate
from drevalpy.models.factory import model_config_for_name
from drevalpy.models.zoo import (
    clear_external_zoo,
    get_zoo_config,
    list_zoo_names,
    load_external_zoo_file,
    zoo_model_config,
)
from drevalpy.models.zoo._external_load import _collect_zoo_entries_from_yaml, _load_zoo_yaml_mapping
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.types.enums.model_scope import ModelScope
from drevalpy.types.enums.prediction_mode import PredictionMode

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
    "EnsembleMF",
    "PaccMann",
]

BLOCK_ZOO_NAMES = set(LITERATURE_ZOO_NAMES) - {"SimpleNeuralNetwork", "MultiViewNeuralNetwork"}


def test_every_builtin_zoo_file_is_listed() -> None:
    assert list_zoo_names(include_external=False) == sorted(path.stem for path in zoo._BUILTIN_ZOO_DIR.glob("*.yaml"))


def test_a_builtin_preset_that_fails_validation_is_skipped_with_a_warning(
    tmp_path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    shutil.copy(zoo._BUILTIN_ZOO_DIR / "Ridge.yaml", tmp_path / "Ridge.yaml")
    (tmp_path / "Broken.yaml").write_text("name: Broken\npredictor:\n  name: noSuchPredictor\n")
    monkeypatch.setattr(zoo, "_BUILTIN_ZOO_DIR", tmp_path)

    with caplog.at_level("WARNING", logger="drevalpy"):
        entries = zoo._load_builtin_entries()

    assert list(entries) == ["Ridge"]
    assert "Broken.yaml" in caplog.text


def test_scope_listing_partitions_the_zoo() -> None:
    single = list_zoo_names(include_external=False, scope=ModelScope.SINGLE_DRUG)
    multi = list_zoo_names(include_external=False, scope="multi_drug")

    assert set(single) == {"SingleDrugElasticNet", "SingleDrugKNN", "SingleDrugRandomForest", "MOLIR", "SuperFELTR"}
    assert set(single).isdisjoint(multi)
    assert set(single) | set(multi) == set(list_zoo_names(include_external=False))
    assert all(get_zoo_config(name).scope == ModelScope.SINGLE_DRUG for name in single)
    assert all(ModelSpec.from_name(name).config.scope is ModelScope.MULTI_DRUG for name in multi)


@pytest.mark.parametrize(
    ("name", "sample", "drug", "predictor"),
    [
        ("ElasticNet", "scaledGeneExpression", "fingerprints", "elasticNet"),
        ("NaivePredictor", None, None, "naiveMean"),
        ("NaiveSampleMeanPredictor", "identity", "constant", ...),
        ("NaiveDrugMeanPredictor", "constant", "identity", ...),
        ("NaiveTissueMeanPredictor", "tissue", "constant", ...),
        ("NaiveTissueDrugMeanPredictor", "tissue", "identity", ...),
        ("NaiveMeanEffectsPredictor", "concatFeaturizers", "identity", ...),
        ("SingleDrugElasticNet", "scaledGeneExpression", "identity", "singleDrugElasticNet"),
        ("SingleDrugRandomForest", "scaledGeneExpression", "identity", "singleDrugRandomForest"),
        ("MOLIR", ..., "identity", ...),
        ("SuperFELTR", ..., "identity", ...),
    ],
)
def test_preset_featurizer_wiring(name: str, sample, drug, predictor) -> None:
    config = get_zoo_config(name)
    actual = (
        config.sample_featurizer and config.sample_featurizer.name,
        config.drug_featurizer and config.drug_featurizer.name,
        config.predictor.name,
    )

    for got, want in zip(actual, (sample, drug, predictor), strict=True):
        assert want is ... or got == want
    validate(config)


def test_naive_mean_effects_concatenates_identity_and_a_tolerant_tissue() -> None:
    sample = get_zoo_config("NaiveMeanEffectsPredictor").sample_featurizer
    assert sample is not None
    children = sample.featurizers
    assert children is not None
    assert [child.name for child in children] == ["identity", "tissue"]
    assert children[1].options is not None
    assert children[1].options["allow_missing"] is True


def test_zoo_model_config_merges_hyperparameters() -> None:
    config = zoo_model_config("ElasticNet", {"alpha": 0.25})

    assert isinstance(config, ResolvedModelConfig)
    assert config.predictor_values()["alpha"] == 0.25
    assert model_config_for_name("ElasticNet", {"alpha": 0.5}).predictor_values()["alpha"] == 0.5  # type: ignore[union-attr]


def test_zoo_model_config_rejects_view_keys() -> None:
    with pytest.raises(ValueError, match=r"Unknown hyperparameter"):
        zoo_model_config("ElasticNet", {"sample_views": ["gene_expression"], "alpha": 0.1})


def test_zoo_model_config_routes_methylation_flat_key_to_pca_child(tmp_path) -> None:
    preset = tmp_path / "zoo.yaml"
    preset.write_text(
        "methylationForest:\n  sample_featurizer:\n    - scaledGeneExpression\n    - pca[methylation]\n"
        "  drug_featurizer: fingerprints\n  predictor: randomForest\n",
        encoding="utf-8",
    )
    clear_external_zoo()
    load_external_zoo_file(preset)
    try:
        resolved = zoo_model_config("methylationForest", {"methylation_n_components": 11})
    finally:
        clear_external_zoo()
    assert isinstance(resolved, ResolvedModelConfig)
    assert resolved.predictor_values().get("methylation_n_components") is None
    assert resolved.featurizer_values("sample", "pca[methylation]")["n_components"] == 11


@pytest.mark.parametrize("name", ["MultiViewRandomForest", "MultiViewXGBoost", "MultiViewLightGBM"])
def test_multi_view_zoo_presets_use_the_1_6_1_default_views(name: str) -> None:
    sample = get_zoo_config(name).sample_featurizer
    assert sample is not None
    assert sample.name == "concatFeaturizers"
    assert sample.featurizers is not None
    gene_expression, mutations = sample.featurizers
    assert (gene_expression.name, dict(gene_expression.options)) == (
        "scaledGeneExpression",
        {"gene_list": "landmark_genes"},
    )
    assert (mutations.name, mutations.view, dict(mutations.options)) == (
        "raw",
        "mutations",
        {"gene_list": "drug_target_genes_all_drugs"},
    )


def test_get_zoo_config_rejects_an_unknown_name_with_unknown_name_error() -> None:
    with pytest.raises(UnknownNameError, match="Unknown zoo entry: NoSuchPreset") as caught:
        get_zoo_config("NoSuchPreset")
    assert isinstance(caught.value, KeyError)


def test_get_zoo_config_applies_prediction_mode_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_predictor("elasticNet"), "supported_modes", frozenset(PredictionMode))
    assert get_zoo_config("ElasticNet").prediction_mode == PredictionMode.REGRESSION
    assert (
        get_zoo_config("ElasticNet", prediction_mode=PredictionMode.CLASSIFICATION).prediction_mode
        == PredictionMode.CLASSIFICATION
    )
    via_zoo = zoo_model_config("ElasticNet", prediction_mode=PredictionMode.CLASSIFICATION)
    assert isinstance(via_zoo, ModelConfig)
    assert via_zoo.prediction_mode == PredictionMode.CLASSIFICATION


def test_external_zoo_rejects_builtin_collision_and_is_atomic(tmp_path) -> None:
    clear_external_zoo()
    entry = "  sample_featurizer: scaledGeneExpression\n  drug_featurizer: fingerprints\n  predictor: elasticNet\n"
    bad = tmp_path / "zoo.yaml"
    bad.write_text(f"goodEntry:\n{entry}ElasticNet:\n{entry}", encoding="utf-8")

    with pytest.raises(ValueError, match="collides with a built-in"):
        load_external_zoo_file(bad)

    assert "goodEntry" not in list_zoo_names(include_external=True)
    clear_external_zoo()


def test_single_drug_knn_zoo_entry_standardizes_per_drug() -> None:
    config = get_zoo_config("SingleDrugKNN")

    assert config.predictor.name == "singleDrugKnn"
    assert config.sample_featurizer is not None
    assert (config.sample_featurizer.name, config.sample_featurizer.options) == (
        "landmarkGenes",
        {"gene_list_stem": "landmark_genes"},
    )
    resolved = ModelSpec.from_name("SingleDrugKNN").resolve()
    assert resolved.featurizer_values("sample", "landmarkGenes")["standardize"] is False
    assert resolved.predictor_values() == {"n_neighbors": 3, "weights": "distance"}
    validate(config)


def test_random_forest_ae_molgnet_zoo_entry_has_the_1_6_1_defaults() -> None:
    config = get_zoo_config("RandomForestAEMolGNet")

    assert config.predictor.name == "randomForest"
    assert config.sample_featurizer is not None
    assert config.sample_featurizer.name == "aeGeneExpression"
    assert config.sample_featurizer.options == {"gene_list": "tcga_autoencoder_genes_available"}
    assert config.drug_featurizer is not None
    assert config.drug_featurizer.name == "molgnetPooled"
    spec = ModelSpec.from_name("RandomForestAEMolGNet")
    assert spec.requires_early_stopping
    assert spec.default_hyperparameters == {
        "epochs_autoencoder": 2000,
        "lr_autoencoder": 1e-4,
        "patience_autoencoder": 50,
        "batch_size_autoencoder": 1024,
        "pooling": "mean_max",
        "n_estimators": 100,
        "max_samples": 0.2,
        "max_depth": 5,
    }
    assert spec.hyperparameter_space["predictor.randomForest.max_depth"]["choices"] == (5, 10, 30)
    validate(config)


def test_multi_drug_sklearn_predictor_without_drug_featurizer_fails() -> None:
    preset = get_zoo_config("ElasticNet")
    with pytest.raises(ValidationError, match="requires a drug_featurizer"):
        ModelConfig(
            sample_featurizer=preset.sample_featurizer,
            drug_featurizer=None,
            predictor=preset.predictor,
            prediction_mode=preset.prediction_mode,
        )


@pytest.mark.parametrize("name", LITERATURE_ZOO_NAMES)
def test_literature_zoo_entries_validate_and_build_a_model(name: str) -> None:
    config = get_zoo_config(name)

    validate(config)

    assert config.sample_featurizer is not None
    if name in BLOCK_ZOO_NAMES:
        assert config.drug_featurizer is not None
        assert issubclass(get_predictor(config.predictor.name), BlockPredictor)
    assert DRPModel(ModelSpec(name, config)) is not None


def test_ensemble_mf_zoo_entry_matches_the_1_6_1_defaults() -> None:
    config = get_zoo_config("EnsembleMF")

    assert config.predictor.name == "ensembleMF"
    assert config.sample_featurizer is not None
    assert config.sample_featurizer.name == "scaledGeneExpression"
    assert config.sample_featurizer.options == {"gene_list": "landmark_genes", "transform": "rank"}
    assert config.drug_featurizer is not None
    assert config.drug_featurizer.hyperparameter_space is not None
    assert config.drug_featurizer.hyperparameter_space["n_bits"]["default"] == 256
    defaults = get_predictor("ensembleMF").get_default_hyperparameters()
    assert (defaults["n_ensemble"], defaults["seed"]) == (20, 0)


def test_paccmann_zoo_entry_matches_the_1_6_1_defaults() -> None:
    config = get_zoo_config("PaccMann")

    assert config.sample_featurizer is not None
    assert config.sample_featurizer.name == "landmarkGenes"
    assert config.sample_featurizer.options == {
        "gene_list_stem": "gene_list_paccmann_network_prop",
        "arcsinh": False,
        "fit_on_pairs": True,
    }
    assert config.drug_featurizer is not None
    assert config.drug_featurizer.name == "smilesTokens"
    defaults = get_predictor("paccMann").get_default_hyperparameters()
    assert (defaults["epochs"], defaults["patience"]) == (10, 10)


def test_load_zoo_yaml_mapping_requires_file(tmp_path) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError, match="not found"):
        _load_zoo_yaml_mapping(missing)


def test_collect_zoo_entries_rejects_non_mapping_entry(tmp_path) -> None:
    data = {"bad": "not-a-dict"}
    with pytest.raises(ValueError, match="must be a mapping"):
        _collect_zoo_entries_from_yaml(data, source=tmp_path / "z.yaml", builtin_names=frozenset())


def test_collect_zoo_entries_single_document_format(tmp_path) -> None:
    payload = {
        "predictor": "elasticNet",
        "sample_featurizer": "scaledGeneExpression",
        "drug_featurizer": "fingerprints",
        "name": "customEntry",
    }
    parsed = _collect_zoo_entries_from_yaml(payload, source=tmp_path / "z.yaml", builtin_names=frozenset())
    assert len(parsed) == 1
    assert parsed[0][0] == "customEntry"
    assert parsed[0][1].predictor.name == "elasticNet"
