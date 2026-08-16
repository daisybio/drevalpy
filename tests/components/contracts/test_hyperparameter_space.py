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
from drevalpy.registry.cell_line_featurizer import cell_line_featurizer_registry
from drevalpy.registry.cell_line_featurizer import register as register_cell_line_featurizer
from drevalpy.registry.drug_featurizer import drug_featurizer_registry
from drevalpy.registry.predictor import predictor_registry
from drevalpy.registry.predictor import register as register_predictor


@pytest.fixture(autouse=True)
def _clear_registries() -> Iterator[None]:
    cell_line_featurizer_registry.clear()
    drug_featurizer_registry.clear()
    predictor_registry.clear()
    yield
    from drevalpy.registry._builtins import register_builtin_components

    register_builtin_components()


def test_validate_hyperparameter_space_rejects_missing_default() -> None:
    with pytest.raises(ValueError, match="missing 'default' for 'alpha'"):
        validate_hyperparameter_space(
            {"alpha": {"type": "float", "low": 0.1, "high": 1.0}},
            context="unit-test",
        )


def test_validate_hyperparameter_space_rejects_non_mapping_spec() -> None:
    with pytest.raises(ValueError, match="non-mapping specs for 'alpha'"):
        validate_hyperparameter_space({"alpha": 1.0}, context="unit-test")


def test_validate_hyperparameter_space_accepts_complete_specs() -> None:
    validate_hyperparameter_space(
        {"alpha": {"type": "float", "default": 0.5}},
        context="unit-test",
    )


def test_predictor_registration_rejects_space_without_default() -> None:
    with pytest.raises(ValueError, match="missing 'default'"):

        @register_predictor(
            "badSpacePred",
            description="missing default",
            cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
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

        @register_cell_line_featurizer(
            "badSpaceFeat",
            description="missing default",
            contract=FeatureFormat.NUMERIC_MATRIX,
        )
        class BadSpaceFeat:
            @classmethod
            def get_hyperparameter_space(cls) -> dict[str, dict[str, object]]:
                return {"n_components": {"type": "int", "low": 8, "high": 64}}


def test_predictor_config_rejects_space_without_default() -> None:
    with pytest.raises(ValidationError, match="missing 'default'"):
        PredictorConfig(
            name="elasticNet",
            hyperparameter_space={"alpha": {"type": "float", "low": 0.1, "high": 1.0}},
        )


def test_featurizer_config_rejects_space_without_default() -> None:
    with pytest.raises(ValidationError, match="missing 'default'"):
        FeaturizerConfig(
            name="pca",
            view="expression",
            hyperparameter_space={"n_components": {"type": "int", "low": 8, "high": 64}},
        )


def test_validate_component_hyperparameter_space_uses_class_getter() -> None:
    class Ok:
        @classmethod
        def get_hyperparameter_space(cls) -> dict[str, dict[str, object]]:
            return {"alpha": {"type": "float", "default": 1.0}}

    validate_component_hyperparameter_space("ok", Ok)


class TestTunableComponentMixin:
    """The four hooks both component kinds inherit rather than each declaring.

    ``Featurizer`` and ``Predictor`` are siblings under ``components/``, so this
    module - the leaf both already imported the validator from - is the only home
    for the shared copy that does not invert a dependency between them.
    """

    @pytest.mark.parametrize("component", [Featurizer, Predictor], ids=["featurizer", "predictor"])
    def test_both_component_bases_inherit_the_mixin(self, component: type) -> None:
        assert issubclass(component, TunableComponentMixin)

    @pytest.mark.parametrize(
        "hook",
        ["get_hyperparameter_space", "get_default_hyperparameters", "get_state", "set_state"],
    )
    def test_neither_base_keeps_its_own_copy_of_a_shared_hook(self, hook: str) -> None:
        """A re-declaration would be a second implementation to keep in sync."""
        assert hook not in Featurizer.__dict__
        assert hook not in Predictor.__dict__

    def test_the_default_space_is_empty_for_a_component_with_nothing_to_tune(self) -> None:
        class Untunable(TunableComponentMixin):
            pass

        assert Untunable.get_hyperparameter_space() == {}
        assert Untunable.get_default_hyperparameters() == {}

    def test_defaults_are_read_off_the_declared_space(self) -> None:
        class Tunable(TunableComponentMixin):
            @classmethod
            def get_hyperparameter_space(cls) -> dict[str, dict[str, object]]:
                return {"alpha": {"type": "float", "low": 0.1, "high": 1.0, "default": 0.25}}

        assert Tunable.get_default_hyperparameters() == {"alpha": 0.25}

    def test_defaults_reject_a_space_entry_without_one(self) -> None:
        """The validator runs on the read path, not only at registration."""

        class Incomplete(TunableComponentMixin):
            @classmethod
            def get_hyperparameter_space(cls) -> dict[str, dict[str, object]]:
                return {"alpha": {"type": "float", "low": 0.1, "high": 1.0}}

        with pytest.raises(ValueError, match=r"Incomplete.get_hyperparameter_space\(\)"):
            Incomplete.get_default_hyperparameters()

    def test_the_default_state_is_empty_and_restoring_it_is_a_no_op(self) -> None:
        """An unfitted component has nothing to persist; ``set_state`` must tolerate it."""

        class Stateless(TunableComponentMixin):
            pass

        component = Stateless()

        assert component.get_state() == {}
        assert component.set_state({"ignored": 1}) is None
        assert component.get_state() == {}

    def test_is_fitted_reads_the_inherited_state_hook(self) -> None:
        """``Predictor.is_fitted`` stays predictor-only but is defined over ``get_state``."""

        class Fitted(Predictor):
            def get_state(self) -> dict[str, object]:
                return {"weights": [1.0]}

            def _fit(self, batch) -> None:
                return None

            def _predict(self, batch):
                return None

        assert Fitted().is_fitted() is True
