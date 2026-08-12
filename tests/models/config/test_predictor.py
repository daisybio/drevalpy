"""Tests for the ``PredictorConfig`` pydantic model.

Asserts the model's own field validation and immutability directly, rather than
through ``ModelConfig``. Recipe-string normalization lives in
``test_predictor_parse.py``; this file only checks that ``PredictorConfig``
routes through it.
"""

from __future__ import annotations

from types import MappingProxyType

import pytest
from pydantic import ValidationError

from drevalpy.models.config.predictor import PredictorConfig
from drevalpy.registry._builtins import register_builtin_components
from drevalpy.registry.predictor import get as get_predictor


@pytest.fixture(autouse=True)
def _registry() -> None:
    """Register the built-ins that the name lookups resolve against."""
    register_builtin_components()


class TestFields:
    """Declared fields and their defaults."""

    def test_name_is_required(self) -> None:
        with pytest.raises(ValidationError, match="name"):
            PredictorConfig()  # type: ignore[call-arg]

    def test_hyperparameter_space_defaults_to_none(self) -> None:
        assert PredictorConfig(name="elasticNet").hyperparameter_space is None

    def test_name_is_coerced_to_str(self) -> None:
        assert PredictorConfig.model_validate({"name": 5, "hyperparameter_space": None}).name == "5"

    def test_extra_fields_are_forbidden(self) -> None:
        with pytest.raises(ValidationError, match="[Ee]xtra"):
            PredictorConfig(name="elasticNet", not_a_field=1)  # type: ignore[call-arg]


class TestRecipeNormalization:
    """The ``before`` validator accepts the compact recipe notations."""

    def test_bare_recipe_string(self) -> None:
        assert PredictorConfig.model_validate("elasticNet").name == "elasticNet"

    def test_one_key_mapping_moves_values_into_the_space(self) -> None:
        config = PredictorConfig.model_validate({"randomForest": {"n_estimators": 10}})

        assert config.name == "randomForest"
        assert config.hyperparameter_space is not None
        assert config.hyperparameter_space["n_estimators"]["default"] == 10

    def test_canonical_mapping_passes_through(self) -> None:
        space = {"n_estimators": {"type": "int", "low": 2, "high": 20, "default": 4}}

        config = PredictorConfig.model_validate({"name": "randomForest", "hyperparameter_space": space})

        assert config.hyperparameter_space is not None
        assert config.hyperparameter_space["n_estimators"]["default"] == 4

    def test_rejects_a_non_tunable_value(self) -> None:
        with pytest.raises(ValidationError, match="non-tunable options"):
            PredictorConfig.model_validate({"randomForest": {"not_a_knob": 1}})


class TestHyperparameterSpaceValidation:
    """The ``after`` validator enforces the search-space contract."""

    def test_accepts_entries_declaring_a_default(self) -> None:
        config = PredictorConfig.model_validate(
            {"name": "randomForest", "hyperparameter_space": {"n_estimators": {"default": 5}}},
        )

        assert config.hyperparameter_space is not None

    def test_rejects_an_entry_without_a_default(self) -> None:
        with pytest.raises(ValidationError, match="default"):
            PredictorConfig.model_validate(
                {"name": "randomForest", "hyperparameter_space": {"n_estimators": {"low": 1, "high": 5}}},
            )

    def test_rejects_a_non_mapping_entry(self) -> None:
        with pytest.raises(ValidationError, match="n_estimators"):
            PredictorConfig.model_validate(
                {"name": "randomForest", "hyperparameter_space": {"n_estimators": 5}},
            )

    def test_error_names_the_offending_config(self) -> None:
        with pytest.raises(ValidationError, match=r"PredictorConfig\('randomForest'\).hyperparameter_space"):
            PredictorConfig.model_validate(
                {"name": "randomForest", "hyperparameter_space": {"n_estimators": {"low": 1}}},
            )

    def test_an_empty_space_is_accepted(self) -> None:
        config = PredictorConfig.model_validate({"name": "elasticNet", "hyperparameter_space": {}})

        assert config.hyperparameter_space == {}


class TestImmutability:
    """The model is frozen and its mapping field is deeply frozen."""

    def test_rejects_field_assignment(self) -> None:
        config = PredictorConfig(name="elasticNet")

        with pytest.raises(ValidationError, match="frozen"):
            config.name = "randomForest"

    def test_hyperparameter_space_is_a_read_only_view(self) -> None:
        config = PredictorConfig.model_validate({"name": "randomForest", "hyperparameter_space": {"a": {"default": 1}}})

        assert isinstance(config.hyperparameter_space, MappingProxyType)
        with pytest.raises(TypeError):
            config.hyperparameter_space["b"] = {"default": 2}  # type: ignore[index]

    def test_nested_space_entries_are_frozen(self) -> None:
        config = PredictorConfig.model_validate({"name": "randomForest", "hyperparameter_space": {"a": {"default": 1}}})

        assert config.hyperparameter_space is not None
        with pytest.raises(TypeError):
            config.hyperparameter_space["a"]["default"] = 2  # type: ignore[index]

    def test_is_hashable(self) -> None:
        assert hash(PredictorConfig(name="elasticNet")) == hash(PredictorConfig(name="elasticNet"))

    def test_dumps_a_plain_mapping(self) -> None:
        config = PredictorConfig.model_validate({"name": "randomForest", "hyperparameter_space": {"a": {"default": 1}}})

        dumped = config.model_dump(mode="json")

        assert dumped == {"name": "randomForest", "hyperparameter_space": {"a": {"default": 1}}}
        assert isinstance(dumped["hyperparameter_space"], dict)


class TestCreateInstance:
    """Instantiation through the predictor registry."""

    def test_builds_the_registered_class(self) -> None:
        instance = PredictorConfig(name="elasticNet").create_instance({"alpha": 0.1, "l1_ratio": 0.5})

        assert isinstance(instance, get_predictor("elasticNet"))

    def test_hyperparameters_default_to_empty(self) -> None:
        instance = PredictorConfig(name="naiveMeanEffects").create_instance()

        assert instance is not None

    def test_unknown_name_raises_at_instantiation_time(self) -> None:
        config = PredictorConfig.model_construct(name="notAPredictor", hyperparameter_space=None)

        with pytest.raises(ValueError, match="notAPredictor"):
            config.create_instance()
