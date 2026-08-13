"""Splitter registry: register, discover, and retrieve splitter functions."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ._registry import Splitter, SplitterRegistry, splitter_registry
from ._validation import SplitValidationError, Validation

__all__ = [
    "SplitValidationError",
    "Splitter",
    "SplitterRegistry",
    "Validation",
    "get",
    "list",
    "metadata",
    "register",
    "splitter_registry",
    "table",
]


def register(mode: str, description: str, validation: Validation, *, override: bool = False):
    """Decorator to register a splitter function under a mode name.

    :param mode: Mode name (e.g. ``"LPO"``, or a custom name).
    :param description: Human-readable description of the splitting approach.
    :param validation: Which leakage constraint to enforce.
    :param override: Replace an already-registered mode instead of raising.
    :returns: Decorator that registers and returns the wrapped function.
    """
    return splitter_registry.register(mode, description, validation, override=override)


def get(mode: str) -> Splitter:
    """Return the validated splitter for the given mode."""
    return splitter_registry.get(mode)


def list() -> list[str]:  # noqa: A001
    """Return sorted list of registered mode names."""
    return splitter_registry.modes


def table() -> pd.DataFrame:
    """Return registry contents as a DataFrame.

    :returns: DataFrame with Mode, Description and Validation columns.
    """
    return splitter_registry.to_dataframe()


def metadata(mode: str) -> dict[str, Any]:
    """Return metadata for a registered splitter mode.

    :param mode: Registry name of the splitter mode.
    :returns: Metadata dict including description and leakage constraint.
    """
    return splitter_registry.get_metadata(mode)
