"""Tests for hyperparameter-space validation, ``default_values`` and ``TunableComponentMixin``."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pydantic import ValidationError

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.contracts.hyperparameter_space import (
    TunableComponentMixin,
    default_values,
    validate_component_hyperparameter_space,
    validate_hyperparameter_space,
)
from drevalpy.components.featurizers.base import Featurizer
from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.abstract.matrix import MatrixPredictor
from drevalpy.models.config import FeaturizerConfig, PredictorConfig
from drevalpy.registry.predictor import register as register_predictor
from drevalpy.registry.sample_featurizer import register as register_sample_featurizer
from tests.registry._helpers import isolated_component_registries


@pytest.fixture(autouse=True)
def _isolated_registries() -> Iterator[None]:
    yield from isolated_component_registries()


def test_validate_hyperparameter_space_rejects_incomplete_specs_and_accepts_complete_ones() -> None:
    with pytest.raises(ValueError, match="missing 'default' for 'alpha'"):
        validate_hyperparameter_space({"alpha": {"type": "float", "low": 0.1, "high": 1.0}}, context="unit-test")
    with pytest.raises(ValueError, match="non-mapping specs for 'alpha'"):
        validate_hyperparameter_space({"alpha": 1.0}, context="unit-test")
    validate_hyperparameter_space({"alpha": {"type": "float", "default": 0.5}}, context="unit-test")

    class Ok:
        @classmethod
        def get_hyperparameter_space(cls) -> dict[str, dict[str, object]]:
            return {"alpha": {"type": "float", "default": 1.0}}

    validate_component_hyperparameter_space("ok", Ok)


def test_default_values_reads_the_default_of_every_entry() -> None:
    space = {"alpha": {"type": "float", "default": 0.25}, "depth": {"type": "int", "default": 3}}

    assert default_values(space, context="unit-test") == {"alpha": 0.25, "depth": 3}


@pytest.mark.parametrize("space", [None, {}])
def test_default_values_of_an_empty_space_is_empty(space) -> None:
    assert default_values(space, context="unit-test") == {}


def test_default_values_of_an_invalid_space_raises_with_its_context() -> None:
    with pytest.raises(ValueError, match="Invalid hyperparameter space in my-context: missing 'default' for 'alpha'"):
        default_values({"alpha": {"type": "float"}}, context="my-context")


def test_predictor_registration_rejects_space_without_default() -> None:
    with pytest.raises(ValueError, match="missing 'default'"):

        @register_predictor(
            "badSpacePred",
            description="missing default",
            sample_input_format=FeatureFormat.NUMERIC_MATRIX,
            drug_input_format=FeatureFormat.NUMERIC_MATRIX,
        )
        class BadSpacePred(MatrixPredictor):
            @classmethod
            def get_hyperparameter_space(cls) -> dict[str, dict[str, object]]:
                return {"alpha": {"type": "float", "low": 0.1, "high": 1.0}}

            def _fit_matrix(self, x, y) -> None:
                return None

            def _predict_matrix(self, x):
                return x[:, 0]


def test_featurizer_registration_rejects_space_without_default() -> None:
    with pytest.raises(ValueError, match="missing 'default'"):

        @register_sample_featurizer(
            "badSpaceFeat",
            description="missing default",
            output_format=FeatureFormat.NUMERIC_MATRIX,
        )
        class BadSpaceFeat:
            @classmethod
            def get_hyperparameter_space(cls) -> dict[str, dict[str, object]]:
                return {"n_components": {"type": "int", "low": 8, "high": 64}}


def test_configs_reject_a_space_without_default() -> None:
    with pytest.raises(ValidationError, match="missing 'default'"):
        PredictorConfig(name="elasticNet", hyperparameter_space={"alpha": {"type": "float", "low": 0.1, "high": 1.0}})
    with pytest.raises(ValidationError, match="missing 'default'"):
        FeaturizerConfig(
            name="pca",
            view="expression",
            hyperparameter_space={"n_components": {"type": "int", "low": 8, "high": 64}},
        )


def test_both_component_bases_inherit_the_mixin_without_re_declaring_its_hooks() -> None:
    for component in (Featurizer, Predictor):
        assert issubclass(component, TunableComponentMixin)
        for hook in ("get_hyperparameter_space", "get_default_hyperparameters"):
            assert hook not in component.__dict__


def test_mixin_defaults_are_read_off_the_declared_space_and_validated() -> None:
    class Untunable(TunableComponentMixin):
        pass

    class Tunable(TunableComponentMixin):
        @classmethod
        def get_hyperparameter_space(cls) -> dict[str, dict[str, object]]:
            return {"alpha": {"type": "float", "low": 0.1, "high": 1.0, "default": 0.25}}

    class Incomplete(TunableComponentMixin):
        @classmethod
        def get_hyperparameter_space(cls) -> dict[str, dict[str, object]]:
            return {"alpha": {"type": "float", "low": 0.1, "high": 1.0}}

    assert Untunable.get_hyperparameter_space() == {}
    assert Untunable.get_default_hyperparameters() == {}
    assert Tunable.get_default_hyperparameters() == {"alpha": 0.25}
    with pytest.raises(ValueError, match=r"Incomplete.get_hyperparameter_space\(\)"):
        Incomplete.get_default_hyperparameters()
