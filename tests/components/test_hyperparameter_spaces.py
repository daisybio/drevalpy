"""Coverage tests for structured hyperparameter spaces on public models."""

from __future__ import annotations

from drevalpy.models import construct_model
from drevalpy.models._model_lookup import known_model_names
from drevalpy.registry._builtins import register_builtin_components as _register_builtins
from drevalpy.registry.predictor import list as list_predictors


def test_model_factory_models_expose_defaults() -> None:
    _register_builtins()
    for model_name in known_model_names(include_external=False):
        model_cls = construct_model(model_name)
        defaults = model_cls.get_default_hyperparameters()
        assert isinstance(defaults, dict), model_name
        assert model_cls.get_hyperparameter_set() == [defaults]


def test_registered_predictors_expose_space_helpers() -> None:
    _register_builtins()
    for name in list_predictors():
        from drevalpy.registry.predictor import get as get_predictor

        cls = get_predictor(name)
        space = cls.get_hyperparameter_space()
        defaults = cls.get_default_hyperparameters()
        assert isinstance(space, dict)
        assert isinstance(defaults, dict)
