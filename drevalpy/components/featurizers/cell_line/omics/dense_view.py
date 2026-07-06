"""Shared base for dense single-view cell-line featurizers."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.contracts import FeatureKind
from drevalpy.components.featurizers._matrix import stack_view_matrix
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer


class DenseViewCellLineFeaturizer(CellLineFeaturizer):
    """Pass through one dense cell-line view without additional transformation."""

    _default_view: ClassVar[str]

    def __init__(self, *, view: str | None = None) -> None:
        self._view = view or self._default_view
        self._output_dim = 0

    def fit(
        self,
        features,
        *,
        entity_ids: np.ndarray | None = None,
    ) -> DenseViewCellLineFeaturizer:
        ids = entity_ids if entity_ids is not None else np.array(list(features.features.keys()))
        matrix = stack_view_matrix(features, self._view, ids)
        self._output_dim = int(matrix.shape[1])
        return self

    def transform(self, features, entity_ids: np.ndarray) -> np.ndarray:
        return stack_view_matrix(features, self._view, entity_ids).astype(np.float32)

    @property
    def output_dim(self) -> int:
        return self._output_dim
