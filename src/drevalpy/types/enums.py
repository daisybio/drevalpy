"""Closed sets of choices shared across the package: accelerator need, model scope, prediction mode, randomization."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum


class Accelerator(StrEnum):
    """Whether a component runs on a torch accelerator, ordered from weakest to strongest.

    ``NONE`` runs on the CPU only. ``OPTIONAL`` uses an accelerator when torch finds one and
    falls back to the CPU otherwise. ``REQUIRED`` cannot run without one.
    """

    NONE = "none"
    OPTIONAL = "optional"
    REQUIRED = "required"

    @classmethod
    def strongest(cls, levels: Iterable[Accelerator]) -> Accelerator:
        """Return the strongest of *levels*, or ``NONE`` when there are none.

        :param levels: The levels of the components of a model.
        :returns: ``REQUIRED`` if any level is required, else ``OPTIONAL`` if any is optional, else ``NONE``.
        """
        order = list(cls)
        return max(levels, key=order.index, default=cls.NONE)


class ModelScope(StrEnum):
    """Whether one model is fitted globally or independently per drug."""

    MULTI_DRUG = "multi_drug"
    SINGLE_DRUG = "single_drug"


class PredictionMode(StrEnum):
    """Whether the model predicts a continuous response or a discrete class."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class RandomizationType(StrEnum):
    """How a randomization test randomizes what the featurizers of a view output."""

    PERMUTATION = "permutation"
    """Shuffle the rows, so each sample or drug gets another one's values."""

    INVARIANT = "invariant"
    """Draw new values that keep each row's mean and standard deviation (and a graph's degrees)."""


class RandomizationMode(StrEnum):
    """Which view a randomization test randomizes: sample or drug, one view (``R``) or its complement (``C``)."""

    SVRC = "SVRC"
    SVCC = "SVCC"
    SVRD = "SVRD"
    SVCD = "SVCD"


def randomization_label(randomization: tuple[str, str]) -> str:
    """Name a randomization for logs, reports and tables.

    :param randomization: The ``(mode, view)`` pair of a randomized run.
    :returns: ``"<mode> (<view>)"``.
    """
    mode, view = randomization
    return f"{mode} ({view})"
