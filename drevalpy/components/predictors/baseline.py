"""Feature-free baseline predictors."""

from __future__ import annotations

from typing import Any, ClassVar

from drevalpy.components.config import PredictionMode
from drevalpy.components.predictors.base import Predictor


class BaselinePredictor(Predictor):
    """Predictors that do not consume encoded feature matrices."""

    uses_features: ClassVar[bool] = False
    required_cell_line_contract = Predictor.required_cell_line_contract
    required_drug_contract = Predictor.required_drug_contract

    def build(self, hyperparameters: dict[str, Any], input_dims: dict[str, Any]) -> None:
        _ = input_dims
        self._hyperparameters = hyperparameters
        self._mode = PredictionMode(
            hyperparameters.get("prediction_mode", PredictionMode.REGRESSION)
        )
