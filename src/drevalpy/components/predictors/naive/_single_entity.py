"""Shared base for entity-level naive predictors."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.predictors.abstract.base import Predictor, get_response
from drevalpy.components.predictors.inputs import Inputs
from drevalpy.components.predictors.naive._matrix_means import (
    additive_effects,
    predict_with_effects,
    require_pair_matrix,
)
from drevalpy.types.data.predictor_input import PredictorInput


class SingleEntityNaivePredictor(Predictor):
    """Predict per-entity means from a one-hot design matrix."""

    sample_inputs: ClassVar[Inputs] = Inputs.blocks()
    drug_inputs: ClassVar[Inputs] = Inputs.blocks()
    _feature_side: ClassVar[str] = "sample"

    _dataset_mean: float | None = None
    _effects: np.ndarray | None = None

    def _fit(self, data: PredictorInput) -> None:
        """Fit on training data.

        :param data: Training batch; its responses give the dataset mean and the per-entity effects.
        """
        y = get_response(data)
        design = self._design(data)
        self._dataset_mean = float(np.mean(y))
        self._effects = additive_effects(design, y, baseline=self._dataset_mean)

    def _predict(self, data: PredictorInput) -> np.ndarray:
        """Predict responses for each pair in the batch.

        :param data: Pairs to score.
        :returns: Dataset mean plus the effect of each pair's entity.
        :raises RuntimeError: If the predictor has not been fitted.
        """
        if self._dataset_mean is None or self._effects is None:
            msg = "Call fit before predict"
            raise RuntimeError(msg)
        design = self._design(data)
        return predict_with_effects(design, self._effects, baseline=self._dataset_mean)

    def _design(self, data: PredictorInput) -> np.ndarray:
        """Return the pair-level one-hot design matrix of the entity side.

        :param data: Batch to read.
        :returns: One row per pair, one column per entity group.
        """
        return require_pair_matrix(data, side=self._feature_side)
