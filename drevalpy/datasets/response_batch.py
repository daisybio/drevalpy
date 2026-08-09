"""Lightweight immutable container for response triples passed to predictors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ResponseBatch:
    """Lightweight immutable container for response triples passed to predictors.

    Provides ``.response``, ``.cell_line_ids``,
    ``.drug_ids``, and ``__len__``.
    """

    response: np.ndarray
    cell_line_ids: np.ndarray
    drug_ids: np.ndarray

    def __len__(self) -> int:
        """Return the number of response pairs.

        :returns: Length of the response array.
        """
        return len(self.response)
