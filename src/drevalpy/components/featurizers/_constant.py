"""Shared constant (one-category / intercept) featurizer logic."""

from __future__ import annotations

from typing import ClassVar, Self

import numpy as np

from drevalpy.types.data.batch.feature_block import FeaturizerResult, numeric_feature_block
from drevalpy.types.data.feature_source import FeatureSource


class ConstantFeaturizerMixin:
    """Emit a single column of ones for every entity (no identity information)."""

    entity_id_only: ClassVar[bool] = True

    def _fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        pair_expanded_ids: np.ndarray | None = None,
        pair_expanded_es_ids: np.ndarray | None = None,
    ) -> Self:
        """Fit nothing: a constant column has no state.

        :param source: Unused.
        :param entity_ids: Unused.
        :param pair_expanded_ids: Unused.
        :param pair_expanded_es_ids: Unused.
        :returns: This featurizer.
        """
        _ = source, entity_ids, pair_expanded_ids, pair_expanded_es_ids
        return self

    def _feature_matrix(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Return an ``(len(entity_ids), 1)`` float32 matrix of ones.

        :param source: Unused.
        :param entity_ids: Identifiers to emit a row for.
        :returns: Column of ones, one row per identifier.
        """
        _ = source
        return np.ones((len(entity_ids), 1), dtype=np.float32)

    def _transform_features(
        self,
        source: FeatureSource,
        entity_ids: np.ndarray,
    ) -> FeaturizerResult:
        """Wrap the column of ones in a single ``constant`` block.

        :param source: Unused.
        :param entity_ids: Identifiers to emit a row for.
        :returns: Result with one numeric block named ``constant``.
        """
        matrix = self._feature_matrix(source, entity_ids)
        return FeaturizerResult(entity_ids, {"constant": numeric_feature_block(matrix)}, ("constant",))

    @property
    def output_dim(self) -> int:
        """Return the output dimension, always 1."""
        return 1

    def get_state(self) -> dict[str, object]:
        """Return the empty fitted state."""
        return {}

    def set_state(self, state: dict[str, object]) -> None:
        """Restore state from a prior ``get_state`` mapping.

        :param state: Ignored; there is no state to restore.
        """
        _ = state
