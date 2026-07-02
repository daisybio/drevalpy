"""Deprecation helpers for legacy console script entry points."""

from __future__ import annotations

import warnings


def warn_deprecated(*, legacy_script: str, replacement: str) -> None:
    """Emit a deprecation warning for a legacy ``drevalpy-*`` console script.

    :param legacy_script: Former console script name (e.g. ``drevalpy-train-cv``).
    :param replacement: Suggested replacement command (e.g. ``drevalpy train-cv``).
    """
    warnings.warn(
        f"{legacy_script} is deprecated; use `{replacement}` instead.",
        FutureWarning,
        stacklevel=3,
    )
