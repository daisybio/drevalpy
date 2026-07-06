"""Generic raw dense pass-through featurizer for one omics view."""

from __future__ import annotations

import numpy as np

from drevalpy.components.contracts import FeatureKind
from drevalpy.components.featurizers._matrix import stack_view_matrix
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer


@register_cell_line_featurizer(
    "raw",
    description="Pass through one dense omics view without preprocessing.",
    category="native",
    contract=FeatureKind.DENSE,
)
class RawCellLineFeaturizer(CellLineFeaturizer):
    """Featurize one omics view as a dense matrix without transformation."""

    def __init__(self, *, view: str) -> None:
        if not view or not view.strip():
            msg = "raw featurizer requires an explicit view"
            raise ValueError(msg)
        self._view = view
        self._output_dim = 0

    def fit(
        self,
        features,
        *,
        entity_ids: np.ndarray | None = None,
    ) -> RawCellLineFeaturizer:
        ids = entity_ids if entity_ids is not None else np.array(list(features.features.keys()))
        matrix = stack_view_matrix(features, self._view, ids)
        self._output_dim = int(matrix.shape[1])
        return self

    def transform(self, features, entity_ids: np.ndarray) -> np.ndarray:
        return stack_view_matrix(features, self._view, entity_ids).astype(np.float32)

    @property
    def output_dim(self) -> int:
        return self._output_dim
