"""How much a component uses a torch accelerator (a GPU, Apple MPS, or another backend)."""

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
