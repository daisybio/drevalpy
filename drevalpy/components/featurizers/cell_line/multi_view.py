"""Structured multi-view cell-line featurizers."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.contracts import FeatureContract, FeatureKind
from drevalpy.components.featurizers._matrix import stack_view_matrix
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer


@register_cell_line_featurizer(
    "multiViewStructured",
    description="Named dense cell-line views without concatenation.",
    category="native",
)
class MultiViewStructuredCellLineFeaturizer(CellLineFeaturizer):
    output_contract: ClassVar[FeatureContract] = FeatureContract(
        kind=FeatureKind.DENSE,
        scope="multi_view",
    )

    def __init__(self, *, views: list[str]) -> None:
        if not views:
            msg = "views must be a non-empty list"
            raise ValueError(msg)
        self._views = list(views)
        self._output_dim = 0
        self._view_dims: dict[str, int] = {}

    def fit(self, features, *, entity_ids: np.ndarray | None = None) -> MultiViewStructuredCellLineFeaturizer:
        ids = entity_ids if entity_ids is not None else np.array(list(features.features.keys()))
        self._view_dims = {}
        for view in self._views:
            matrix = stack_view_matrix(features, view, ids)
            self._view_dims[view] = int(matrix.shape[1])
        self._output_dim = sum(self._view_dims.values())
        return self

    def transform(self, features, entity_ids: np.ndarray) -> np.ndarray:
        blocks = self.transform_blocks(features, entity_ids)
        if not blocks:
            return np.empty((len(entity_ids), 0), dtype=np.float32)
        return np.concatenate([blocks[view] for view in self._views], axis=1).astype(np.float32)

    def transform_blocks(self, features, entity_ids: np.ndarray) -> dict[str, np.ndarray]:
        return {view: stack_view_matrix(features, view, entity_ids).astype(np.float32) for view in self._views}

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def view_dims(self) -> dict[str, int]:
        return dict(self._view_dims)
