"""Index scope for model train/predict operations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class EntityScope:
    """Index scope for model train/predict operations.

    Unlike SplitMasks (which represents a CV fold partition), EntityScope simply
    says "operate on these entities." Used by _ComponentStack.train() and predict().
    """

    cell_lines: np.ndarray
    drugs: np.ndarray | None = None
