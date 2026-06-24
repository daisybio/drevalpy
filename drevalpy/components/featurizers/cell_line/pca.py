"""PCA cell-line featurizer."""

from __future__ import annotations

from typing import Any

import numpy as np

from drevalpy.components.featurizers._matrix import stack_view_matrix
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer


@register_cell_line_featurizer(
    "pca",
    description="PCA compression of one dense cell-line view fit on training cell lines.",
    category="native",
)
class PCACellLineFeaturizer(CellLineFeaturizer):
    """Reduce one cell-line view with PCA."""

    def __init__(self, *, view: str = "gene_expression", n_components: int = 128) -> None:
        from sklearn.decomposition import PCA

        self._view = view
        self._n_components = int(n_components)
        self._pca = PCA(n_components=self._n_components)
        self._output_dim = 0

    def fit(
        self,
        features,
        *,
        entity_ids: np.ndarray | None = None,
    ) -> PCACellLineFeaturizer:
        ids = entity_ids if entity_ids is not None else np.array(list(features.features.keys()))
        matrix = stack_view_matrix(features, self._view, ids)
        n_components = min(self._n_components, matrix.shape[0], matrix.shape[1])
        self._pca.n_components = n_components
        self._pca.fit(matrix)
        self._output_dim = n_components
        return self

    def transform(self, features, entity_ids: np.ndarray) -> np.ndarray:
        matrix = stack_view_matrix(features, self._view, entity_ids)
        return self._pca.transform(matrix).astype(np.float32)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        return {
            "n_components": {"type": "int", "low": 8, "high": 512, "default": 128},
        }
