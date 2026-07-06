"""Scaled gene-expression featurizer for cell lines."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from sklearn.preprocessing import StandardScaler

from drevalpy.components.contracts import FeatureKind
from drevalpy.components.featurizers._matrix import stack_view_matrix
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer
from drevalpy.data.preprocessing import scale_gene_expression
from drevalpy.datasets.dataset import FeatureDataset


@register_cell_line_featurizer(
    "scaledGeneExpression",
    description="Landmark gene expression with arcsinh transform and scaling.",
    category="native",
    contract=FeatureKind.DENSE,
)
class ScaledGeneExpressionFeaturizer(CellLineFeaturizer):
    """Match sklearn baseline gene-expression preprocessing."""

    def __init__(self, *, view: str = "gene_expression") -> None:
        self._view = view
        self._scaler = StandardScaler()
        self._output_dim = 0
        self._fitted_features: FeatureDataset | None = None
        self._is_fitted = False

    def fit(
        self,
        features,
        *,
        entity_ids: np.ndarray | None = None,
    ) -> ScaledGeneExpressionFeaturizer:
        ids = entity_ids if entity_ids is not None else np.array(list(features.features.keys()))
        scaled = scale_gene_expression(
            cell_line_input=features.copy(),
            cell_line_ids=np.unique(ids),
            training=True,
            gene_expression_scaler=self._scaler,
        )
        self._fitted_features = scaled
        matrix = stack_view_matrix(scaled, self._view, np.array(list(scaled.features.keys())))
        self._output_dim = int(matrix.shape[1])
        self._is_fitted = True
        return self

    def transform(self, features, entity_ids: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            msg = "ScaledGeneExpressionFeaturizer must be fit before transform"
            raise RuntimeError(msg)
        scaled = scale_gene_expression(
            cell_line_input=features.copy(),
            cell_line_ids=np.unique(entity_ids),
            training=False,
            gene_expression_scaler=self._scaler,
        )
        return stack_view_matrix(scaled, self._view, entity_ids).astype(np.float32)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def get_state(self) -> dict[str, object]:
        return {
            "gene_expression_scaler": self._scaler,
            "fitted": self._is_fitted,
        }

    def set_state(self, state: dict[str, object]) -> None:
        scaler = state.get("gene_expression_scaler")
        if isinstance(scaler, StandardScaler):
            self._scaler = scaler
        if state.get("fitted"):
            self._is_fitted = True
            self._fitted_features = None
