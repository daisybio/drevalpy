"""Methylation cell-line featurizer with scaling and PCA."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from drevalpy.components.contracts import FeatureKind
from drevalpy.components.featurizers._matrix import stack_view_matrix
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer
from drevalpy.data.preprocessing import prepare_expression_and_methylation


@register_cell_line_featurizer(
    "methylationPCA",
    description="Methylation view with scaling and PCA compression.",
    category="native",
    contract=FeatureKind.DENSE,
)
class MethylationPCACellLineFeaturizer(CellLineFeaturizer):
    """Match baseline methylation preprocessing used in multi-view models."""

    def __init__(self, *, view: str = "methylation", n_components: int = 100) -> None:
        self._view = view
        self._n_components = int(n_components)
        self._methylation_scaler = StandardScaler()
        self._methylation_pca = PCA(n_components=self._n_components)
        self._output_dim = 0
        self._is_fitted = False

    def fit(
        self,
        features,
        *,
        entity_ids: np.ndarray | None = None,
    ) -> MethylationPCACellLineFeaturizer:
        ids = entity_ids if entity_ids is not None else np.array(list(features.features.keys()))
        processed = prepare_expression_and_methylation(
            cell_line_input=features.copy(),
            cell_line_ids=np.unique(ids),
            training=True,
            methylation_scaler=self._methylation_scaler,
            methylation_pca=self._methylation_pca,
        )
        matrix = stack_view_matrix(processed, self._view, np.array(list(processed.features.keys())))
        self._output_dim = int(matrix.shape[1])
        self._is_fitted = True
        return self

    def transform(self, features, entity_ids: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            msg = "MethylationPCACellLineFeaturizer must be fit before transform"
            raise RuntimeError(msg)
        processed = prepare_expression_and_methylation(
            cell_line_input=features.copy(),
            cell_line_ids=np.unique(entity_ids),
            training=False,
            methylation_scaler=self._methylation_scaler,
            methylation_pca=self._methylation_pca,
        )
        return stack_view_matrix(processed, self._view, entity_ids).astype(np.float32)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        return {
            "n_components": {"type": "int", "low": 20, "high": 200, "default": 100},
        }

    def get_state(self) -> dict[str, object]:
        return {
            "methylation_scaler": self._methylation_scaler,
            "methylation_pca": self._methylation_pca,
            "fitted": self._is_fitted,
        }

    def set_state(self, state: dict[str, object]) -> None:
        scaler = state.get("methylation_scaler")
        if isinstance(scaler, StandardScaler):
            self._methylation_scaler = scaler
        pca = state.get("methylation_pca")
        if isinstance(pca, PCA):
            self._methylation_pca = pca
        if state.get("fitted"):
            self._is_fitted = True
