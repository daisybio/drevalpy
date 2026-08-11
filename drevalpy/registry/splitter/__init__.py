"""Splitter registry: register, discover, and retrieve splitter functions."""

from ._registry import Splitter, SplitterRegistry, splitter_registry
from ._validation import SplitValidationError, Validation

__all__ = [
    "Splitter",
    "SplitValidationError",
    "SplitterRegistry",
    "Validation",
    "get",
    "list",
    "register",
    "splitter_registry",
    "table",
]


def register(mode: str, description: str, validation: Validation):
    """Decorator to register a splitter function under a mode name."""
    return splitter_registry.register(mode, description, validation)


def get(mode: str) -> Splitter:
    """Return the validated splitter for the given mode."""
    return splitter_registry.get(mode)


def list() -> list[str]:  # noqa: A001
    """Return sorted list of registered mode names."""
    return splitter_registry.modes


def table():
    """Return registry contents as a DataFrame."""
    return splitter_registry.to_dataframe()
