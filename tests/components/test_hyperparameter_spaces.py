"""Coverage tests for structured hyperparameter spaces on public models."""

from __future__ import annotations

import drevalpy.components.register_builtins as register_builtins
from drevalpy.components.registry import list_predictors
from drevalpy.models import construct_model
from drevalpy.models._model_lookup import known_model_names


def test_model_factory_models_expose_defaults() -> None:
    register_builtins.register_builtin_components()
    for model_name in known_model_names(include_external=False):
        model_cls = construct_model(model_name)
        defaults = model_cls.get_default_hyperparameters()
        assert isinstance(defaults, dict), model_name
        assert model_cls.get_hyperparameter_set() == [defaults]


def test_registered_predictors_expose_space_helpers() -> None:
    register_builtins.register_builtin_components()
    for name in list_predictors():
        from drevalpy.components.registry import get_predictor

        cls = get_predictor(name)
        space = cls.get_hyperparameter_space()
        defaults = cls.get_default_hyperparameters()
        assert isinstance(space, dict)
        assert isinstance(defaults, dict)
