"""Base classes for featurizers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np

from drevalpy.components.contracts import FeatureContract, FeatureKind
from drevalpy.datasets.dataset import FeatureDataset


class Featurizer(ABC):
    """Transform feature tables into per-entity representation matrices.

    Cell-line featurizers consume cell-line features; drug featurizers consume
    drug features. Both declare :attr:`output_contract` for predictor matching.
    """

    output_contract: ClassVar[FeatureContract] = FeatureContract(kind=FeatureKind.DENSE)

    @abstractmethod
    def fit(
        self,
        features: FeatureDataset,
        *,
        entity_ids: np.ndarray | None = None,
    ) -> Featurizer:
        """Fit on the entities given by *entity_ids* (or all entities when ``None``)."""

    @abstractmethod
    def transform(self, features: FeatureDataset, entity_ids: np.ndarray) -> np.ndarray:
        """Return a 2D feature matrix with one row per entity id in *entity_ids*."""

    def transform_blocks(
        self,
        features: FeatureDataset,
        entity_ids: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Return named feature blocks; default is a single ``default`` dense matrix."""
        return {"default": self.transform(features, entity_ids)}

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Feature dimension after :meth:`fit`."""

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

    def get_state(self) -> dict[str, object]:
        """Return serializable fitted state for legacy save/load bridges."""
        return {}

    def set_state(self, state: dict[str, object]) -> None:
        """Restore fitted state produced by :meth:`get_state`."""
        _ = state
