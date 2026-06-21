"""Metadata passed to predictors that need entity identifiers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PairContext:
    """Metadata passed to predictors that need entity identifiers."""

    cell_line_ids: np.ndarray
    drug_ids: np.ndarray
    tissue_ids: np.ndarray | None = None
