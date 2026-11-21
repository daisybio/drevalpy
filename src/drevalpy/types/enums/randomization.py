"""How a randomization test replaces a view, and how a randomization is named in output."""

from __future__ import annotations

from enum import StrEnum


class RandomizationType(StrEnum):
    """How a randomization test replaces the values of one view."""

    PERMUTATION = "permutation"
    """Shuffle the rows, so each sample or drug gets another one's values."""

    INVARIANT = "invariant"
    """Draw new values that keep each row's mean and standard deviation (and a graph's degrees)."""


def randomization_label(randomization: tuple[str, str]) -> str:
    """Name a randomization for logs, reports and tables.

    :param randomization: The ``(mode, view)`` pair of a randomized run.
    :returns: ``"<mode> (<view>)"``.
    """
    mode, view = randomization
    return f"{mode} ({view})"
