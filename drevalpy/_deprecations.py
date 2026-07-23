"""Shared deprecation helpers for user-facing API migrations."""

from __future__ import annotations

import warnings
from typing import Final

_WARNED: set[str] = set()

FACTORY_DICT_NAMES: Final[frozenset[str]] = frozenset(
    {
        "MODEL_FACTORY",
        "MULTI_DRUG_MODEL_FACTORY",
        "SINGLE_DRUG_MODEL_FACTORY",
    }
)


def warn_deprecated(*, what: str, replacement: str, stacklevel: int = 3) -> None:
    """Emit a ``FutureWarning`` for a deprecated public API.

    Warnings are deduplicated by ``what`` so repeated access in the same process
    does not spam. No fixed removal release is promised.

    :param what: Deprecated API description shown in the warning.
    :param replacement: Recommended replacement API description.
    :param stacklevel: ``warnings.warn`` stack level pointing at user code.
    """
    if what in _WARNED:
        return
    _WARNED.add(what)
    warnings.warn(
        f"{what} is deprecated; use {replacement} instead.",
        FutureWarning,
        stacklevel=stacklevel,
    )


def reset_deprecation_warnings() -> None:
    """Clear warn-once state (tests only)."""
    _WARNED.clear()
