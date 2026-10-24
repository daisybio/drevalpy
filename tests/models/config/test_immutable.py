"""Tests for immutable ModelConfig and featurizer/predictor templates."""

from __future__ import annotations

from types import MappingProxyType

import pytest
from pydantic import ValidationError

from drevalpy.models.config import (
    DrugFeaturizerConfig,
    FeaturizerConfig,
    ModelConfig,
    PredictorConfig,
    ResolvedModelConfig,
    SampleFeaturizerConfig,
)
from drevalpy.models.zoo import get_zoo_config


def _template() -> ModelConfig:
    return ModelConfig(
        sample_featurizer=SampleFeaturizerConfig(name="scaledGeneExpression"),
        drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
        predictor=PredictorConfig(name="elasticNet"),
    )


def test_model_config_is_frozen_and_scope_is_read_only() -> None:
    with pytest.raises(ValidationError, match="frozen"):
        _template().drug_featurizer = None
    assert isinstance(ModelConfig.scope, property)
    assert ModelConfig.scope.fset is None


def test_concat_children_are_tuple_not_list() -> None:
    config = SampleFeaturizerConfig.model_validate(["scaledGeneExpression", {"pca[methylation]": {"n_components": 32}}])

    assert config.name == "concatFeaturizers"
    assert isinstance(config.featurizers, tuple)
    assert config.featurizers[1].name == "pca"
    with pytest.raises(TypeError):
        config.featurizers[0] = SampleFeaturizerConfig(name="raw", view="expression")  # type: ignore[index]


def test_predictor_shorthand_writes_hyperparameter_space_defaults() -> None:
    config = ModelConfig.model_validate(
        {
            "sample_featurizer": "scaledGeneExpression",
            "drug_featurizer": "fingerprints",
            "predictor": {"randomForest": {"n_estimators": 10}},
        }
    )
    space = config.predictor.hyperparameter_space
    assert space is not None
    assert space["n_estimators"]["default"] == 10
    with pytest.raises(AttributeError):
        _ = config.predictor.hyperparameters  # type: ignore[attr-defined]


def test_zoo_config_copy_isolation() -> None:
    first = get_zoo_config("MultiViewLightGBM")
    second = get_zoo_config("MultiViewLightGBM")
    assert first == second
    assert first is not second
    assert first.sample_featurizer is not None
    assert first.sample_featurizer.featurizers is not None
    child = first.sample_featurizer.featurizers[1]
    assert child.options is not None
    with pytest.raises(TypeError):
        child.options["gene_list"] = "landmark_genes"  # type: ignore[index]


def test_mapping_fields_are_deeply_frozen_views() -> None:
    featurizer = FeaturizerConfig.model_validate(
        {"name": "tissue", "options": {"allow_missing": True, "nested": {"inner": [1, {"deep": 2}]}}}
    )
    predictor = PredictorConfig.model_validate(
        {"name": "randomForest", "hyperparameter_space": {"n_estimators": {"default": 5, "options": [1, 2]}}}
    )
    resolved = ResolvedModelConfig(template=_template(), values={"predictor.elasticNet.alpha": 0.5})

    assert featurizer.options is not None
    assert featurizer.hyperparameter_space is None
    assert predictor.hyperparameter_space is not None
    for mapping in (
        featurizer.options,
        featurizer.options["nested"],
        predictor.hyperparameter_space,
        predictor.hyperparameter_space["n_estimators"],
        resolved.values,
        ResolvedModelConfig(template=_template()).values,
    ):
        assert isinstance(mapping, MappingProxyType)
        with pytest.raises(TypeError):
            mapping["injected"] = 1  # type: ignore[index]
    assert featurizer.options["nested"]["inner"] == (1, MappingProxyType({"deep": 2}))
    assert predictor.hyperparameter_space["n_estimators"]["options"] == (1, 2)


def test_frozen_mappings_dump_plain_containers() -> None:
    featurizer = FeaturizerConfig.model_validate({"name": "tissue", "options": {"nested": {"inner": [1, {"deep": 2}]}}})

    dumped = featurizer.model_dump(mode="python")

    assert dumped["options"] == {"nested": {"inner": [1, {"deep": 2}]}}
    assert isinstance(dumped["options"], dict)
    assert isinstance(dumped["options"]["nested"]["inner"], list)
    assert isinstance(dumped["options"]["nested"]["inner"][1], dict)
    assert featurizer.model_dump(mode="json")["options"] == dumped["options"]
    assert PredictorConfig(name="elasticNet").model_dump(mode="python")["hyperparameter_space"] is None
