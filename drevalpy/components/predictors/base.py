"""Base class for predictors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from drevalpy.components.contracts import FeatureContract, FeatureFormat
from drevalpy.types.model_scope import ModelScope
from drevalpy.types.prediction_mode import PredictionMode

if TYPE_CHECKING:
    from drevalpy.components.model_input_batch import ModelInputBatch


class Predictor(ABC):
    """Train and predict drug response from a ``ModelInputBatch``."""

    cell_line_contract: ClassVar[FeatureContract] = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
    drug_contract: ClassVar[FeatureContract] = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
    requires_drug_featurizer: ClassVar[bool] = True
    routing_drug_featurizer: ClassVar[str | None] = None
    supports_early_stopping: ClassVar[bool] = False
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})
    supported_scopes: ClassVar[frozenset[ModelScope]] = frozenset({ModelScope.MULTI_DRUG})
    required_cell_line_blocks: ClassVar[tuple[str, ...]] = ()
    required_drug_blocks: ClassVar[tuple[str, ...]] = ()

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Store hyperparameters merged with class defaults."""
        self._hyperparameters: dict[str, Any] = {
            **self.get_default_hyperparameters(),
            **(hyperparameters or {}),
        }

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return tunable hyperparameter specs for HPO."""
        return {}

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        """Return default hyperparameter values from the HP space."""
        return {
            key: spec["default"]
            for key, spec in cls.get_hyperparameter_space().items()
            if isinstance(spec, dict) and "default" in spec
        }

    @abstractmethod
    def fit(self, batch: ModelInputBatch) -> None:
        """Fit on a featurized predictor input batch."""

    @abstractmethod
    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Predict response for each pair in *batch*."""

    def get_state(self) -> dict[str, object]:
        """Return serializable fitted state for legacy save/load bridges."""
        return {}

    def set_state(self, state: dict[str, object]) -> None:
        """Restore fitted state produced by `get_state`."""
        _ = state

    def is_fitted(self) -> bool:
        """Return whether the predictor has been fit."""
        return bool(self.get_state())
