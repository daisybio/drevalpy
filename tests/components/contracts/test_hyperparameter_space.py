"""Tests for hyperparameter-space default enforcement."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pydantic import ValidationError

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.contracts.hyperparameter_space import (
    TunableComponentMixin,
    validate_component_hyperparameter_space,
    validate_hyperparameter_space,
)
from drevalpy.components.featurizers.base import Featurizer
from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.abstract.matrix import MatrixPredictor
from drevalpy.models.config import FeaturizerConfig, PredictorConfig
from drevalpy.registry.drug_featurizer import drug_featurizer_registry
from drevalpy.registry.predictor import predictor_registry
from drevalpy.registry.predictor import register as register_predictor
from drevalpy.registry.sample_featurizer import register as register_sample_featurizer
from drevalpy.registry.sample_featurizer import sample_featurizer_registry


@pytest.fixture(autouse=True)
def _clear_registries() -> Iterator[None]:
    sample_featurizer_registry.clear()
    drug_featurizer_registry.clear()
    predictor_registry.clear()
    yield
    from drevalpy.registry._builtins import register_builtin_components

    register_builtin_components()


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


def test_predictor_registration_rejects_space_without_default() -> None:
    with pytest.raises(ValueError, match="missing 'default'"):

        @register_predictor(
            "badSpacePred",
            description="missing default",
            sample_contract=FeatureFormat.NUMERIC_MATRIX,
            drug_contract=FeatureFormat.NUMERIC_MATRIX,
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
            contract=FeatureFormat.NUMERIC_MATRIX,
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
        for hook in ("get_hyperparameter_space", "get_default_hyperparameters", "get_state", "set_state"):
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


def test_the_default_state_is_empty_and_is_fitted_reads_the_state_hook() -> None:
    class Stateless(TunableComponentMixin):
        pass

    class Fitted(Predictor):
        def get_state(self) -> dict[str, object]:
            return {"weights": [1.0]}

        def _fit(self, batch) -> None:
            return None

        def _predict(self, batch):
            return None

    component = Stateless()
    assert component.get_state() == {}
    assert component.set_state({"ignored": 1}) is None
    assert component.get_state() == {}
    assert Fitted().is_fitted() is True
