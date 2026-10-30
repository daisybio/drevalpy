"""Tests for hyperparameter-space validation, ``default_values`` and ``TunableComponent``."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from drevalpy.components.contracts.hyperparameter_space import (
    TunableComponent,
    default_values,
    fixed_entries,
    is_fixed,
    tunable_entries,
    validate_component_hyperparameter_space,
    validate_hyperparameter_space,
)
from drevalpy.components.featurizers.featurizer import Featurizer
from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.abstract.matrix import MatrixPredictor
from drevalpy.models.config import FeaturizerConfig, PredictorConfig
from drevalpy.registry.featurizer import register as register_featurizer
from drevalpy.registry.predictor import register as register_predictor
from tests.registry._helpers import empty_component_registries


@pytest.fixture(autouse=True)
def _isolated_registries() -> None:
    empty_component_registries()


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

        @register_featurizer("badSpaceFeat", description="missing default")
        class BadSpaceFeat(Featurizer):
            def fit(self, source, train_ids, *, train_pair_ids=None, es_pair_ids=None):
                return self

            def transform(self, source, ids):
                raise NotImplementedError

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


def test_both_component_bases_inherit_tunable_component_without_re_declaring_its_hooks() -> None:
    for component in (Featurizer, Predictor):
        assert issubclass(component, TunableComponent)
        for hook in ("get_hyperparameter_space", "default_hyperparameters"):
            assert hook not in component.__dict__


def test_defaults_are_read_off_the_declared_space_and_validated() -> None:
    class Untunable(TunableComponent):
        pass

    class Tunable(TunableComponent):
        @classmethod
        def get_hyperparameter_space(cls) -> dict[str, dict[str, object]]:
            return {"alpha": {"type": "float", "low": 0.1, "high": 1.0, "default": 0.25}}

    class Incomplete(TunableComponent):
        @classmethod
        def get_hyperparameter_space(cls) -> dict[str, dict[str, object]]:
            return {"alpha": {"type": "float", "low": 0.1, "high": 1.0}}

    assert Untunable.get_hyperparameter_space() == {}
    assert Untunable.default_hyperparameters() == {}
    assert Tunable.default_hyperparameters() == {"alpha": 0.25}
    with pytest.raises(ValueError, match=r"Incomplete.get_hyperparameter_space\(\)"):
        Incomplete.default_hyperparameters()


def test_fixed_entries_need_a_default_and_carry_no_search_keys() -> None:
    validate_hyperparameter_space({"seed": {"type": "fixed", "default": 0}}, context="unit-test")
    with pytest.raises(ValueError, match="missing 'default' for 'seed'"):
        validate_hyperparameter_space({"seed": {"type": "fixed"}}, context="unit-test")
    with pytest.raises(ValueError, match="'fixed' specs must not declare low/high/choices/log for 'seed'"):
        validate_hyperparameter_space({"seed": {"type": "fixed", "default": 0, "low": 0}}, context="unit-test")


def test_fixed_entries_have_defaults_but_are_not_tunable() -> None:
    space = {
        "alpha": {"type": "float", "low": 0.1, "high": 1.0, "default": 0.5},
        **fixed_entries({"seed": 0, "solver": None}),
    }

    assert space["seed"] == {"type": "fixed", "default": 0}
    assert [name for name, spec in space.items() if is_fixed(spec)] == ["seed", "solver"]
    assert tunable_entries(space) == {"alpha": space["alpha"]}
    assert default_values(space, context="unit-test") == {"alpha": 0.5, "seed": 0, "solver": None}
