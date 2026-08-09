"""Shared constant (one-category / intercept) featurizer logic."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.feature_block import FeatureBlock, numeric_feature_block
from drevalpy.components.feature_source import FeatureSource
from drevalpy.components.featurizer_fit_context import FeaturizerFitContext


class ConstantFeaturizerMixin:
    """Emit a single column of ones for every entity (no identity information)."""

    entity_id_only: ClassVar[bool] = True

    def fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        context: FeaturizerFitContext | None = None,
    ):
        """Fit on training data.

        :param source: Feature source providing views for the entity type.
        :param entity_ids: entity ids.
        :param context: context.
        :returns: Result.
        """
        _ = source, entity_ids, context
        return self

    def transform(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Transform inputs into feature payloads.

        :param source: Feature source providing views for the entity type.
        :param entity_ids: entity ids.
        :returns: Result.
        """
        _ = source
        return np.ones((len(entity_ids), 1), dtype=np.float32)

    def transform_blocks(
        self,
        source: FeatureSource,
        entity_ids: np.ndarray,
    ) -> dict[str, FeatureBlock]:
        """Transform blocks.

        :param source: Feature source providing views for the entity type.
        :param entity_ids: entity ids.
        :returns: Result.
        """
        return {"constant": numeric_feature_block(self.transform(source, entity_ids))}

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
