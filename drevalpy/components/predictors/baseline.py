"""Feature-free baseline predictors."""

from __future__ import annotations

from typing import Any, ClassVar

from drevalpy.components.predictors.base import Predictor
from drevalpy.models.config import PredictionMode


class BaselinePredictor(Predictor):
    """Predictors that do not consume encoded feature matrices."""

    requires_drug_featurizer: ClassVar[bool] = False
    cell_line_contract = Predictor.cell_line_contract
    drug_contract = Predictor.drug_contract

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        super().__init__(hyperparameters)
        self._mode = PredictionMode(self._hyperparameters.get("prediction_mode", PredictionMode.REGRESSION))
