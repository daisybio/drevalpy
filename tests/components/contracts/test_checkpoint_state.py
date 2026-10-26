"""Tests for ``CheckpointStateMixin``."""

from __future__ import annotations

from drevalpy.components.contracts.checkpoint_state import CheckpointStateMixin
from drevalpy.components.featurizers.base import Featurizer
from drevalpy.components.predictors.abstract.base import Predictor


def test_both_component_bases_inherit_the_mixin_without_re_declaring_its_hooks() -> None:
    for component in (Featurizer, Predictor):
        assert issubclass(component, CheckpointStateMixin)
        for hook in ("get_state", "set_state"):
            assert hook not in component.__dict__


def test_the_default_state_is_empty_and_restoring_it_changes_nothing() -> None:
    class Stateless(CheckpointStateMixin):
        pass

    component = Stateless()

    assert component.get_state() == {}
    assert component.set_state({"ignored": 1}) is None
    assert component.get_state() == {}


def test_is_fitted_reads_the_state_hook() -> None:
    class Fitted(Predictor):
        def get_state(self) -> dict[str, object]:
            return {"weights": [1.0]}

        def _fit(self, batch) -> None:
            return None

        def _predict(self, batch):
            return None

    assert Fitted().is_fitted() is True
