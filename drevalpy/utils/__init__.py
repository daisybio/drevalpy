"""Pipeline helpers, argument checks, response transforms, and decorators."""

from __future__ import annotations

from typing import Any

from ._pipeline_function import pipeline_function

__all__ = [
    "check_arguments",
    "get_datasets",
    "get_response_transformation",
    "main",
    "pipeline_function",
]

_LAZY_EXPORTS = {
    "check_arguments": ".validation",
    "get_datasets": ".pipeline",
    "get_response_transformation": ".response_transform",
    "main": ".pipeline",
}


def __getattr__(name: str) -> Any:
    """Lazily import heavier utilities to avoid dataset/utils import cycles.

    Args:
        name: Attribute name requested from this package.

    Returns:
        The resolved attribute value.

    Raises:
        AttributeError: If *name* is not a known lazy export.
    """
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    from importlib import import_module

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
