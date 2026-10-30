"""Predictor registry: the ``register`` decorator and the registry of predictor classes."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from drevalpy.registry._registry import Registry
from drevalpy.registry.components._metadata import LiteratureReference
from drevalpy.registry.components._register import register_component
from drevalpy.registry.components._registration_metadata import normalize_registration_metadata
from drevalpy.registry.predictor._validate import validate_predictor_registration

__all__ = ["get", "predictor_registry", "register"]

#: Every registered predictor class, in registration order.
predictor_registry: Registry[type[Any]] = Registry("predictor")


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
    metadata = normalize_registration_metadata(description, tags, reference)

    def decorator(cls: type[Any]) -> type[Any]:
        return register_component(predictor_registry, "predictor", name, cls, metadata, validate_predictor_registration)

    return decorator


def get(name: str) -> type[Any]:
    """Return the predictor class registered under *name*.

    An unknown *name* raises :class:`~drevalpy.errors.UnknownNameError`.

    :param name: Registry name of the predictor.
    :returns: Predictor class registered under *name*.
    """
    return predictor_registry.get(name)
