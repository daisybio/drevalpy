"""Literature and neural models wrapped as monolithic component predictors."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.pair_context import PairContext
from drevalpy.components.config import PredictionMode
from drevalpy.components.predictors.base import Predictor
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.drp_model import DRPModel


class LegacyStackPredictor(Predictor):
    """Delegate training and prediction to an existing :class:`~drevalpy.models.drp_model.DRPModel`."""

    uses_features: ClassVar[bool] = False
    uses_raw_features: ClassVar[bool] = True
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})

    def __init__(self, model_cls: type[DRPModel]) -> None:
        self._model_cls = model_cls
        self._model: DRPModel | None = None

    def build(self, hyperparameters: dict[str, Any], input_dims: dict[str, Any]) -> None:
        _ = input_dims
        self._model = self._model_cls()
        self._model.build_model(hyperparameters)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        pair_context: PairContext | None = None,
    ) -> None:
        _ = x, y, pair_context

    def predict(
        self,
        x: np.ndarray,
        *,
        pair_context: PairContext | None = None,
    ) -> np.ndarray:
        _ = x, pair_context
        msg = "LegacyStackPredictor uses predict_raw()"
        raise RuntimeError(msg)

    def fit_raw(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None,
        *,
        output_earlystopping: DrugResponseDataset | None = None,
    ) -> None:
        if self._model is None:
            msg = "Call build() before fit_raw()"
            raise RuntimeError(msg)
        self._model.train(
            output,
            cell_line_input,
            drug_input,
            output_earlystopping=output_earlystopping,
        )

    def predict_raw(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None,
    ) -> np.ndarray:
        if self._model is None:
            return np.full(len(cell_line_ids), np.nan, dtype=np.float64)
        return self._model.predict(cell_line_ids, drug_ids, cell_line_input, drug_input)


def register_legacy_predictor(name: str, model_cls: type[DRPModel], *, description: str) -> None:
    """Register a literature model as a monolithic predictor component."""
    from drevalpy.components.registry import register_predictor

    @register_predictor(name, description=description, category="native")
    class _Legacy(LegacyStackPredictor):
        def __init__(self) -> None:
            super().__init__(model_cls)

    _Legacy.__name__ = f"Legacy{name}Predictor"
