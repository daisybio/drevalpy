"""Predictor registry: the ``register`` decorator and the registry of predictor classes."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from drevalpy.components.contracts.hyperparameter_space import validate_component_hyperparameter_space
from drevalpy.components.predictors.inputs import predictor_inputs
from drevalpy.registry._component import LiteratureReference, component_decorator
from drevalpy.registry._registry import Registry

__all__ = ["get", "predictor_registry", "register"]

#: Every registered predictor class, in registration order.
predictor_registry: Registry[type[Any]] = Registry("predictor")


def _validate(name: str, cls: type[Any]) -> None:
    """Check that both sides declare their inputs and that the hyperparameter space is valid.

    ``predictor_inputs`` raises ``TypeError`` for a side without a declaration.

    :param name: Registry name under which *cls* is being registered.
    :param cls: Predictor class.
    """
    for side in ("sample", "drug"):
        predictor_inputs(cls, side)
    validate_component_hyperparameter_space(name, cls)


def register(
    name: str,
    *,
    description: str,
    tags: Iterable[str] | None = None,
    reference: LiteratureReference | None = None,
) -> Callable[[type[Any]], type[Any]]:
    """Return a class decorator that registers a predictor under *name*.

    What the predictor reads from each side is its ``sample_inputs`` / ``drug_inputs``
    declaration; registration checks that both are declared.

    :param name: Registry name used in ``ModelConfig`` and recipes.
    :param description: Short human-readable summary for catalog listings.
    :param tags: Optional discovery tags.
    :param reference: Optional literature citation metadata.
    :returns: Class decorator that registers and returns the decorated predictor.
    """
    return component_decorator(
        predictor_registry, name, description=description, tags=tags, reference=reference, validate=_validate
    )


def get(name: str) -> type[Any]:
    """Return the predictor class registered under *name*.

    An unknown *name* raises :class:`~drevalpy.errors.UnknownNameError`.

    :param name: Registry name of the predictor.
    :returns: Predictor class registered under *name*.
    """
    return predictor_registry.get(name)
