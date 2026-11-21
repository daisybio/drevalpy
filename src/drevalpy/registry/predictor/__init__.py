"""Predictor registry: the ``register`` decorator and the registry of predictor classes."""

from ._registry import get, predictor_registry, register

__all__ = ["get", "predictor_registry", "register"]
