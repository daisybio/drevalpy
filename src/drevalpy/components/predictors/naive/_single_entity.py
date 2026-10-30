"""Shared base for entity-level naive predictors."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.inputs import Inputs
from drevalpy.components.predictors.naive._matrix_means import (
    additive_effects,
    predict_with_effects,
    require_pair_matrix,
)
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch


class SingleEntityNaivePredictor(Predictor):
    """Predict per-entity means from a one-hot design matrix."""

    sample_inputs: ClassVar[Inputs] = Inputs.blocks()
    drug_inputs: ClassVar[Inputs] = Inputs.blocks()
    _feature_side: ClassVar[str] = "sample"

    _dataset_mean: float | None = None
    _effects: np.ndarray | None = None

    def _fit(self, batch: ModelInputBatch) -> None:
        """Fit on training data.

        :param batch: Training batch; its responses give the dataset mean and the per-entity effects.
        :raises RuntimeError: If ``batch.response`` is None.
        """
        y = batch.response
        if y is None:
            raise RuntimeError("batch.response is required for fit")
        design = require_pair_matrix(batch, side=self._feature_side)
        self._dataset_mean = float(np.mean(y))
        self._effects = additive_effects(design, y, baseline=self._dataset_mean)

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Predict responses for each pair in the batch.

        :param batch: Pairs to score.
        :returns: Dataset mean plus the effect of each pair's entity.
        :raises RuntimeError: If the predictor has not been fitted.
        """
        if self._dataset_mean is None or self._effects is None:
            msg = "Call fit before predict"
            raise RuntimeError(msg)
        design = require_pair_matrix(batch, side=self._feature_side)
        return predict_with_effects(design, self._effects, baseline=self._dataset_mean)
