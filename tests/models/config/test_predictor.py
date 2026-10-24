"""Tests for the ``PredictorConfig`` pydantic model (field validation, immutability, instantiation)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from drevalpy.models.config.predictor import PredictorConfig
from drevalpy.registry.predictor import get as get_predictor


class TestFields:
    def test_defaults_and_coercion(self) -> None:
        assert PredictorConfig(name="elasticNet").hyperparameter_space is None
        assert PredictorConfig.model_validate({"name": 5, "hyperparameter_space": None}).name == "5"

    def test_name_is_required_and_extra_fields_are_forbidden(self) -> None:
        with pytest.raises(ValidationError, match="name"):
            PredictorConfig()  # type: ignore[call-arg]
        with pytest.raises(ValidationError, match="[Ee]xtra"):
            PredictorConfig(name="elasticNet", not_a_field=1)  # type: ignore[call-arg]


class TestRecipeNormalization:
    def test_accepted_notations(self) -> None:
        assert PredictorConfig.model_validate("elasticNet").name == "elasticNet"

        short = PredictorConfig.model_validate({"randomForest": {"n_estimators": 10}})
        assert short.name == "randomForest"
        assert short.hyperparameter_space is not None
        assert short.hyperparameter_space["n_estimators"]["default"] == 10

        space = {"n_estimators": {"type": "int", "low": 2, "high": 20, "default": 4}}
        canonical = PredictorConfig.model_validate({"name": "randomForest", "hyperparameter_space": space})
        assert canonical.hyperparameter_space is not None
        assert canonical.hyperparameter_space["n_estimators"]["default"] == 4

    def test_rejects_a_non_tunable_value(self) -> None:
        with pytest.raises(ValidationError, match="non-tunable options"):
            PredictorConfig.model_validate({"randomForest": {"not_a_knob": 1}})


class TestHyperparameterSpaceValidation:
    @pytest.mark.parametrize(
        ("space", "match"),
        [
            ({"n_estimators": {"low": 1, "high": 5}}, "default"),
            ({"n_estimators": 5}, "n_estimators"),
            ({"n_estimators": {"low": 1}}, r"PredictorConfig\('randomForest'\).hyperparameter_space"),
        ],
        ids=["no-default", "non-mapping-entry", "error-names-the-config"],
    )
    def test_rejects_invalid_entries(self, space, match) -> None:
        with pytest.raises(ValidationError, match=match):
            PredictorConfig.model_validate({"name": "randomForest", "hyperparameter_space": space})

    def test_accepts_default_entries_and_an_empty_space(self) -> None:
        config = PredictorConfig.model_validate(
            {"name": "randomForest", "hyperparameter_space": {"n_estimators": {"default": 5}}}
        )
        assert config.hyperparameter_space is not None
        assert (
            PredictorConfig.model_validate({"name": "elasticNet", "hyperparameter_space": {}}).hyperparameter_space
            == {}
        )


class TestImmutability:
    def test_frozen_hashable_and_deeply_read_only(self) -> None:
        config = PredictorConfig.model_validate({"name": "randomForest", "hyperparameter_space": {"a": {"default": 1}}})

        with pytest.raises(ValidationError, match="frozen"):
            config.name = "other"
        assert hash(PredictorConfig(name="elasticNet")) == hash(PredictorConfig(name="elasticNet"))
        assert config.hyperparameter_space is not None
        with pytest.raises(TypeError):
            config.hyperparameter_space["b"] = {"default": 2}  # type: ignore[index]
        with pytest.raises(TypeError):
            config.hyperparameter_space["a"]["default"] = 2  # type: ignore[index]

        dumped = config.model_dump(mode="json")
        assert dumped == {"name": "randomForest", "hyperparameter_space": {"a": {"default": 1}}}
        assert isinstance(dumped["hyperparameter_space"], dict)


class TestCreateInstance:
    def test_builds_the_registered_class(self) -> None:
        instance = PredictorConfig(name="elasticNet").create_instance({"alpha": 0.1, "l1_ratio": 0.5})

        assert isinstance(instance, get_predictor("elasticNet"))
        assert PredictorConfig(name="naiveMeanEffects").create_instance() is not None

    def test_unknown_name_raises_at_instantiation_time(self) -> None:
        config = PredictorConfig.model_construct(name="notAPredictor", hyperparameter_space=None)

        with pytest.raises(ValueError, match="notAPredictor"):
            config.create_instance()
