"""Shared constant (one-category / intercept) featurizer logic."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.datasets.dataset import FeatureDataset


class ConstantFeaturizerMixin:
    """Emit a single column of ones for every entity (no identity information)."""

    entity_id_only: ClassVar[bool] = True

    def fit(
        self,
        features: FeatureDataset,
        *,
        entity_ids: np.ndarray | None = None,
    ):
        _ = features, entity_ids
        return self

    def transform(self, features: FeatureDataset, entity_ids: np.ndarray) -> np.ndarray:
        _ = features
        return np.ones((len(entity_ids), 1), dtype=np.float32)

    def transform_blocks(
        self,
        features: FeatureDataset,
        entity_ids: np.ndarray,
    ) -> dict[str, np.ndarray]:
        return {"constant": self.transform(features, entity_ids)}

    @property
    def output_dim(self) -> int:
        return 1

    def get_state(self) -> dict[str, object]:
        return {}

    def set_state(self, state: dict[str, object]) -> None:
        _ = state
