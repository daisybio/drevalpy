"""Base helpers for predictors that consume named feature blocks."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.base import Predictor


class BlockPredictor(Predictor):
    """Predictor that reads side-specific or named featurizer output blocks.

    “Block” includes named matrices (for example ``identity`` / ``tissue``) and
    side-specific design matrices that must not be flattened indiscriminately.
    """

    input_interface: ClassVar[str] = "block"

    def fit(self, batch: ModelInputBatch) -> None:
        """Fit on a featurized predictor input batch.

        :param batch: Featurized pairs with training responses.
        """

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Predict on a featurized predictor input batch.

        :param batch: Featurized pairs to score.

        :returns: One predicted response per pair in *batch*.
        """
        return np.full(batch.n_pairs, np.nan, dtype=np.float64)
