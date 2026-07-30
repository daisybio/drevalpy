"""Base classes for featurizers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np

from drevalpy.components.contracts import FeatureContract, FeatureFormat
from drevalpy.components.feature_block import FeatureBlock, numeric_feature_block
from drevalpy.components.featurizer_fit_context import FeaturizerFitContext
from drevalpy.datasets.dataset import FeatureDataset


class Featurizer(ABC):
    """Transform feature tables into per-entity representation payloads.

    Cell-line featurizers consume cell-line features; drug featurizers consume
    drug features. Both declare ``contract`` for predictor matching. Numeric
    featurizers return 2D matrices; graph and ragged featurizers return object
    arrays of payloads.
    """

    contract: ClassVar[FeatureContract] = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)

    @abstractmethod
    def fit(
        self,
        features: FeatureDataset,
        *,
        entity_ids: np.ndarray | None = None,
        context: FeaturizerFitContext | None = None,
    ) -> Featurizer:
        """Fit on the entities given by *entity_ids* (or all entities when ``None``)."""

    @abstractmethod
    def transform(self, features: FeatureDataset, entity_ids: np.ndarray) -> np.ndarray:
        """Return one payload row per entity id in *entity_ids*."""

    def transform_blocks(
        self,
        features: FeatureDataset,
        entity_ids: np.ndarray,
    ) -> dict[str, FeatureBlock]:
        """Return named feature blocks; default is a single ``default`` block."""
        return {
            "default": numeric_feature_block(self.transform(features, entity_ids)),
        }

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Feature dimension after `fit`."""

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

    @classmethod
    def load_features(cls, data_path: str, dataset_name: str, **kwargs: object) -> FeatureDataset:
        """Load the raw dataset required by this featurizer.

        Featurizers that require bespoke on-disk artifacts override this hook.
        Generic views continue to be loaded by the model data-loading layer.
        """
        _ = data_path, dataset_name, kwargs
        raise NotImplementedError

    def get_state(self) -> dict[str, object]:
        """Return serializable fitted state for legacy save/load bridges."""
        return {}

    def set_state(self, state: dict[str, object]) -> None:
        """Restore fitted state produced by `get_state`."""
        _ = state
