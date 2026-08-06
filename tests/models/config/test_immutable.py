"""Tests for immutable ModelConfig and featurizer/predictor templates."""

from __future__ import annotations

from types import MappingProxyType

import pytest
from pydantic import ValidationError

from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.models.config import (
    CellLineFeaturizerConfig,
    DrugFeaturizerConfig,
    FeaturizerConfig,
    ModelConfig,
    PredictorConfig,
    ResolvedModelConfig,
)
from drevalpy.models.zoo import get_zoo_config


def test_model_config_rejects_field_assignment() -> None:
    register_builtin_components()
    config = ModelConfig(
        cell_line_featurizer=CellLineFeaturizerConfig(name="scaledGeneExpression"),
        drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
        predictor=PredictorConfig(name="elasticNet"),
    )
    with pytest.raises(ValidationError, match="frozen"):
        config.drug_featurizer = None


def test_model_config_scope_is_a_read_only_property() -> None:
    register_builtin_components()
    assert isinstance(ModelConfig.scope, property)
    assert ModelConfig.scope.fset is None


def test_featurizer_config_rejects_deep_options_mutation() -> None:
    register_builtin_components()
    config = FeaturizerConfig.model_validate(
        {"name": "tissue", "options": {"allow_missing": True}},
    )
    assert config.options is not None
    with pytest.raises(TypeError):
        config.options["allow_missing"] = False  # type: ignore[index]


def test_concat_children_are_tuple_not_list() -> None:
    register_builtin_components()
    config = CellLineFeaturizerConfig.model_validate(
        ["scaledGeneExpression", {"pca[methylation]": {"n_components": 32}}],
    )
    assert config.name == "concatFeaturizers"
    assert isinstance(config.featurizers, tuple)
    assert config.featurizers is not None
    assert config.featurizers[1].name == "pca"
    with pytest.raises(TypeError):
        config.featurizers[0] = CellLineFeaturizerConfig(name="raw", view="expression")  # type: ignore[index]


def test_predictor_shorthand_writes_hyperparameter_space_defaults() -> None:
    register_builtin_components()
    config = ModelConfig.model_validate(
        {
            "cell_line_featurizer": "scaledGeneExpression",
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
    register_builtin_components()
    first = get_zoo_config("MultiViewLightGBM")
    second = get_zoo_config("MultiViewLightGBM")
    assert first == second
    assert first is not second
    assert first.cell_line_featurizer is not None
    child = first.cell_line_featurizer.featurizers[1]
    assert child.hyperparameter_space is not None
    with pytest.raises(TypeError):
        child.hyperparameter_space["n_components"] = {"default": 8}  # type: ignore[index]


def test_frozen_mapping_fields_are_deeply_frozen_views() -> None:
    register_builtin_components()
    featurizer = FeaturizerConfig.model_validate(
        {
            "name": "tissue",
            "options": {"allow_missing": True, "nested": {"inner": [1, {"deep": 2}]}},
        },
    )
    predictor = PredictorConfig.model_validate(
        {"name": "randomForest", "hyperparameter_space": {"n_estimators": {"default": 5, "options": [1, 2]}}},
    )
    resolved = ResolvedModelConfig(
        template=ModelConfig(
            cell_line_featurizer=CellLineFeaturizerConfig(name="scaledGeneExpression"),
            drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
            predictor=PredictorConfig(name="elasticNet"),
        ),
        values={"predictor.elasticNet.alpha": 0.5},
    )

    assert featurizer.options is not None
    assert featurizer.hyperparameter_space is None
    assert predictor.hyperparameter_space is not None
    frozen_mappings = [
        featurizer.options,
        featurizer.options["nested"],
        predictor.hyperparameter_space,
        predictor.hyperparameter_space["n_estimators"],
        resolved.values,
    ]
    for mapping in frozen_mappings:
        assert isinstance(mapping, MappingProxyType)
        with pytest.raises(TypeError):
            mapping["injected"] = 1  # type: ignore[index]
    assert featurizer.options["nested"]["inner"] == (1, MappingProxyType({"deep": 2}))
    assert predictor.hyperparameter_space["n_estimators"]["options"] == (1, 2)


def test_frozen_mapping_default_is_frozen() -> None:
    register_builtin_components()
    resolved = ResolvedModelConfig(
        template=ModelConfig(
            cell_line_featurizer=CellLineFeaturizerConfig(name="scaledGeneExpression"),
            drug_featurizer=DrugFeaturizerConfig(name="fingerprints"),
            predictor=PredictorConfig(name="elasticNet"),
        ),
    )
    assert isinstance(resolved.values, MappingProxyType)
    with pytest.raises(TypeError):
        resolved.values["predictor.elasticNet.alpha"] = 1.0  # type: ignore[index]


def test_frozen_mapping_dumps_plain_containers() -> None:
    register_builtin_components()
    featurizer = FeaturizerConfig.model_validate(
        {
            "name": "tissue",
            "options": {"nested": {"inner": [1, {"deep": 2}]}},
        },
    )
    dumped = featurizer.model_dump(mode="python")
    assert dumped["options"] == {"nested": {"inner": [1, {"deep": 2}]}}
    assert isinstance(dumped["options"], dict)
    assert isinstance(dumped["options"]["nested"]["inner"], list)
    assert isinstance(dumped["options"]["nested"]["inner"][1], dict)
    assert featurizer.model_dump(mode="json")["options"] == dumped["options"]
    assert PredictorConfig(name="elasticNet").model_dump(mode="python")["hyperparameter_space"] is None
