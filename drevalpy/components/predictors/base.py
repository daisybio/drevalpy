"""Base class for predictors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from drevalpy.components.config import PredictionMode
from drevalpy.components.contracts import FeatureContract, FeatureKind

if TYPE_CHECKING:
    from drevalpy.components.pair_context import PairContext


class Predictor(ABC):
    """Train and predict drug response from featurized cell-line and drug features."""

    required_cell_line_contract: ClassVar[FeatureContract] = FeatureContract(kind=FeatureKind.DENSE)
    required_drug_contract: ClassVar[FeatureContract] = FeatureContract(kind=FeatureKind.DENSE)
    uses_features: ClassVar[bool] = True
    uses_structured_features: ClassVar[bool] = False
    requires_drug_featurizer: ClassVar[bool] = True
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset(PredictionMode)

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
    def build(self, hyperparameters: dict[str, Any], input_dims: dict[str, Any]) -> None:
        """Allocate the underlying estimator or module."""

    @abstractmethod
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        pair_context: PairContext | None = None,
    ) -> None:
        """Fit on feature rows (or empty rows for feature-free baselines)."""

    @abstractmethod
    def predict(
        self,
        x: np.ndarray,
        *,
        pair_context: PairContext | None = None,
    ) -> np.ndarray:
        """Predict response for feature rows."""

    def get_state(self) -> dict[str, object]:
        """Return serializable fitted state for legacy save/load bridges."""
        return {}

    def set_state(self, state: dict[str, object]) -> None:
        """Restore fitted state produced by :meth:`get_state`."""
        _ = state

    def is_fitted(self) -> bool:
        """Return whether the predictor has been fit."""
        return bool(self.get_state())
