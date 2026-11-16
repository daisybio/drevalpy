"""Splitter registry: register, resolve and retrieve splitter functions."""

from ._groups import Validation
from ._registry import Splitter, canonical, get, register, splitter_registry
from ._validation import SplitValidationError

__all__ = [
    "SplitValidationError",
    "Splitter",
    "Validation",
    "canonical",
    "get",
    "register",
    "splitter_registry",
]
