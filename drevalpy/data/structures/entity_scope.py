"""Index scope for model train/predict operations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class EntityScope:
    """2D pair array defining which (cell_line, drug) pairs to operate on.

    The ``pairs`` array has shape (n_pairs, 2) where column 0 is cell line index
    and column 1 is drug index. Used by _ComponentStack.train() and predict().
    """

    pairs: np.ndarray
