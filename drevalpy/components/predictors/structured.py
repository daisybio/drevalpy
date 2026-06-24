"""Base helpers for predictors that consume structured pair batches."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.pair_batch import PairBatch
from drevalpy.components.pair_context import PairContext
from drevalpy.components.predictors.base import Predictor
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset


class StructuredPredictor(Predictor):
    """Predictor that consumes featurizer outputs via `~drevalpy.components.pair_batch.PairBatch`."""

    uses_features: ClassVar[bool] = False
    uses_structured_features: ClassVar[bool] = True

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        pair_context: PairContext | None = None,
    ) -> None:
        msg = f"{type(self).__name__} requires fit_structured()"
        raise RuntimeError(msg)

    def predict(
        self,
        x: np.ndarray,
        *,
        pair_context: PairContext | None = None,
    ) -> np.ndarray:
        msg = f"{type(self).__name__} requires predict_structured()"
        raise RuntimeError(msg)

    def build(self, hyperparameters: dict[str, Any], input_dims: dict[str, Any]) -> None:
        self._hyperparameters = dict(hyperparameters)
        self._input_dims = dict(input_dims)

    def fit_structured(
        self,
        batch: PairBatch,
        *,
        output: DrugResponseDataset | None = None,
        cell_line_input: FeatureDataset | None = None,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
    ) -> None:
        """Fit on a structured featurized batch."""

    def predict_structured(
        self,
        batch: PairBatch,
        *,
        cell_line_input: FeatureDataset | None = None,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """Predict on a structured featurized batch."""
        return np.full(len(batch.cell_line_ids), np.nan, dtype=np.float64)
