"""Shared base for entity-level naive predictors."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.inputs import Inputs
from drevalpy.components.predictors.naive._category_means import (
    additive_effects,
    effects_of,
    require_pair_codes,
)
from drevalpy.types.data.feature_source import Side
from drevalpy.types.data.predictor_input import PredictorInput


class SingleEntityNaivePredictor(Predictor):
    """Predict per-entity means from the category code of each pair's entity."""

    sample_inputs: ClassVar[Inputs] = Inputs.blocks()
    drug_inputs: ClassVar[Inputs] = Inputs.blocks()
    _feature_side: ClassVar[Side] = Side.SAMPLE

    _dataset_mean: float | None = None
    _effects: np.ndarray | None = None

    def _fit(self, data: PredictorInput) -> None:
        """Fit on training data.

        :param data: Training batch; its responses give the dataset mean and the per-entity effects.
        """
        y = data.response
        codes, n_categories = self._codes(data)
        self._dataset_mean = float(np.mean(y))
        self._effects = additive_effects(codes, n_categories, y, baseline=self._dataset_mean)

    def _predict(self, data: PredictorInput) -> np.ndarray:
        """Predict responses for each pair in the batch.

        :param data: Pairs to score.
        :returns: Dataset mean plus the effect of each pair's entity; the dataset mean for an entity without a
            category (an all-zero row) or without training pairs.
        :raises RuntimeError: If the predictor has not been fitted.
        """
        if self._dataset_mean is None or self._effects is None:
            msg = "Call fit before predict"
            raise RuntimeError(msg)
        codes, _ = self._codes(data)
        return self._dataset_mean + effects_of(codes, self._effects)

    def _codes(self, data: PredictorInput) -> tuple[np.ndarray, int]:
        """Return the category code of each pair's entity.

        :param data: Batch to read.
        :returns: One code per pair (``-1`` for an all-zero row) and the number of categories.
        """
        return require_pair_codes(data, side=self._feature_side)
