"""Tests for hyperparameter-space validation, ``default_values`` and ``TunableComponent``."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from drevalpy.components.contracts.hyperparameter_space import (
    TunableComponent,
    admits,
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

        @register_featurizer("badSpaceFeat", side="sample", description="missing default")
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
            side="sample",
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


_INT = {"type": "int", "low": 2, "high": 5, "default": 3}
_FLOAT = {"type": "float", "low": 0.1, "high": 0.5, "default": 0.2}
_POW2 = {"type": "pow2", "low": 8, "high": 11, "default": 256}
_LAYERS = {"type": "categorical", "choices": [[4, 2], (8, 4)], "default": [4, 2]}


@pytest.mark.parametrize(
    ("spec", "value", "expected"),
    [
        pytest.param(_INT, 3, True, id="int-inside"),
        pytest.param(_INT, 2, True, id="int-low-edge"),
        pytest.param(_INT, 5, True, id="int-high-edge"),
        pytest.param(_INT, 1, False, id="int-below"),
        pytest.param(_INT, 6, False, id="int-above"),
        pytest.param(_INT, 3.0, False, id="int-given-a-float"),
        pytest.param(_INT, True, False, id="int-given-a-bool"),
        pytest.param(_FLOAT, 0.3, True, id="float-inside"),
        pytest.param(_FLOAT, 0.1, True, id="float-low-edge"),
        pytest.param(_FLOAT, 0.5, True, id="float-high-edge"),
        pytest.param(_FLOAT, 0.05, False, id="float-below"),
        pytest.param(_FLOAT, 0.6, False, id="float-above"),
        pytest.param({**_FLOAT, "low": 0, "high": 2}, 1, True, id="float-given-an-int"),
        pytest.param({**_FLOAT, "low": 0, "high": 2}, True, False, id="float-given-a-bool"),
        pytest.param(_FLOAT, "0.3", False, id="float-given-a-string"),
        pytest.param(_POW2, 256, True, id="pow2-low-edge"),
        pytest.param(_POW2, 2048, True, id="pow2-high-edge"),
        pytest.param(_POW2, 512, True, id="pow2-inside"),
        pytest.param(_POW2, 128, False, id="pow2-below"),
        pytest.param(_POW2, 4096, False, id="pow2-above"),
        pytest.param(_POW2, 300, False, id="pow2-not-a-power"),
        pytest.param(_POW2, 0, False, id="pow2-zero"),
        pytest.param(_POW2, -256, False, id="pow2-negative"),
        pytest.param(_POW2, 8, False, id="pow2-given-the-exponent"),
        pytest.param({"type": "pow2", "default": 1}, True, False, id="pow2-given-a-bool"),
        pytest.param(_LAYERS, [4, 2], True, id="categorical-list"),
        pytest.param(_LAYERS, (4, 2), True, id="categorical-tuple-matches-list"),
        pytest.param(_LAYERS, [8, 4], True, id="categorical-list-matches-tuple"),
        pytest.param(_LAYERS, [2, 2], False, id="categorical-outside"),
        pytest.param({"type": "categorical", "choices": ["a", "b"], "default": "a"}, "c", False, id="categorical-str"),
        pytest.param({"choices": [1, 2], "default": 1}, 3, False, id="categorical-is-the-default-type"),
        pytest.param({"type": "fixed", "default": 5}, 10_000, True, id="fixed-admits-anything"),
        pytest.param(5, "anything", True, id="non-mapping-admits-anything"),
        pytest.param({"type": "int", "default": 3}, -100, True, id="int-without-bounds"),
        pytest.param({"type": "int", "low": 2, "default": 3}, 10**9, True, id="int-without-high"),
        pytest.param({"type": "int", "low": 2, "default": 3}, 1, False, id="int-with-only-low"),
        pytest.param({"type": "float", "high": 1.0, "default": 0.5}, -5.0, True, id="float-without-low"),
        pytest.param({"type": "pow2", "default": 4}, 2**20, True, id="pow2-without-bounds"),
        pytest.param({"type": "categorical", "default": "x"}, "y", True, id="categorical-without-choices"),
    ],
)
def test_admits(spec, value, expected: bool) -> None:
    assert admits(spec, value) is expected


@pytest.mark.parametrize(
    "spec",
    [
        pytest.param({"type": "int", "low": 50, "high": 200, "default": 10}, id="int"),
        pytest.param({"type": "float", "low": 0.1, "high": 0.5, "default": 1.0}, id="float"),
        pytest.param({"type": "pow2", "low": 9, "high": 11, "default": 256}, id="pow2"),
        pytest.param({"type": "categorical", "choices": [[4, 2]], "default": [2, 2]}, id="categorical"),
    ],
)
def test_validate_hyperparameter_space_rejects_a_default_outside_its_entry(spec) -> None:
    with pytest.raises(ValueError, match="'default' outside the range or choices for 'knob'"):
        validate_hyperparameter_space({"knob": spec}, context="unit-test")


def test_configs_reject_a_default_outside_the_space() -> None:
    with pytest.raises(ValidationError, match="'default' outside the range or choices for 'alpha'"):
        PredictorConfig(
            name="elasticNet",
            hyperparameter_space={"alpha": {"type": "float", "low": 0.1, "high": 1.0, "default": 5.0}},
        )
