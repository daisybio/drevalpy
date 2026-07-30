"""Training-population context for featurizer fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class FeaturizerFitContext:
    """Entity ID populations available during featurizer fit without response values."""

    unique_train_ids: np.ndarray
    pair_expanded_train_ids: np.ndarray
    unique_early_stopping_ids: np.ndarray
    pair_expanded_early_stopping_ids: np.ndarray
    side: Literal["cell_line", "drug"]
