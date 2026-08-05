"""Shared constant (one-category / intercept) featurizer logic."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.feature_block import FeatureBlock, numeric_feature_block
from drevalpy.components.featurizer_fit_context import FeaturizerFitContext
from drevalpy.datasets.dataset import FeatureDataset


class ConstantFeaturizerMixin:
    """Emit a single column of ones for every entity (no identity information)."""

    entity_id_only: ClassVar[bool] = True

    def fit(
        self,
        features: FeatureDataset,
        *,
        entity_ids: np.ndarray | None = None,
        context: FeaturizerFitContext | None = None,
    ):
        """Fit on training data.

        :param features: features.
        :param entity_ids: entity ids.
        :param context: context.
        :returns: Result.
        """
        _ = features, entity_ids, context
        return self

    def transform(self, features: FeatureDataset, entity_ids: np.ndarray) -> np.ndarray:
        """Transform inputs into feature payloads.

        :param features: features.
        :param entity_ids: entity ids.
        :returns: Result.
        """
        _ = features
        return np.ones((len(entity_ids), 1), dtype=np.float32)

    def transform_blocks(
        self,
        features: FeatureDataset,
        entity_ids: np.ndarray,
    ) -> dict[str, FeatureBlock]:
        """Transform blocks.

        :param features: features.
        :param entity_ids: entity ids.
        :returns: Result.
        """
        return {"constant": numeric_feature_block(self.transform(features, entity_ids))}

    @property
    def output_dim(self) -> int:
        """Return output feature dimension after fitting.

        :returns: Result.
        """
        return 1

    def get_state(self) -> dict[str, object]:
        """Return serializable fitted state.

        :returns: Result.
        """
        return {}

    def set_state(self, state: dict[str, object]) -> None:
        """Restore state from a prior ``get_state`` mapping.

        :param state: state.
        """
        _ = state
