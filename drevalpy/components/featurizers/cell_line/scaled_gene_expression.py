"""Scaled gene-expression featurizer for cell lines."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers.cell_line.base import DenseViewCellLineFeaturizer
from drevalpy.registry.cell_line_featurizer import register
from drevalpy.types.data.feature_source import FeatureSource


@register(
    "scaledGeneExpression",
    description="Landmark gene expression with arcsinh transform and scaling.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class ScaledGeneExpressionFeaturizer(DenseViewCellLineFeaturizer):
    """Match sklearn baseline gene-expression preprocessing."""

    input_views: ClassVar[tuple[str, ...]] = ("gene_expression",)
    requires_fit: ClassVar[bool] = True
    fit_on_unique_ids: ClassVar[bool] = True

    def __init__(self, *, view: str = "gene_expression") -> None:
        """Initialize instance state.

        :param view: view.
        """
        from sklearn.preprocessing import StandardScaler

        super().__init__(view=view)
        self._scaler = StandardScaler()

    def _fit_state(self, source: FeatureSource, entity_ids: np.ndarray) -> int:
        """Fit the scaler on arcsinh-transformed training rows.

        :param source: Feature source providing view matrices.
        :param entity_ids: Deduplicated cell-line identifiers to fit on.
        :returns: Output feature dimension.
        """
        matrix = np.arcsinh(self._raw_matrix(source, entity_ids))
        self._scaler.fit(matrix)
        return int(matrix.shape[1])

    def _compute_matrix(self, source: FeatureSource, matrix: np.ndarray) -> np.ndarray:
        """Arcsinh-transform and scale *matrix*.

        :param source: Feature source the matrix came from.
        :param matrix: Raw view matrix for the requested entity IDs.
        :returns: Scaled feature matrix.
        """
        _ = source
        return self._scaler.transform(np.arcsinh(matrix))

    def _block_name(self) -> str:
        """Publish under the canonical gene-expression block name.

        :returns: Block name.
        """
        return "gene_expression"

    def get_state(self) -> dict[str, object]:
        """Return serializable fitted state.

        :returns: Result.
        """
        if not self._is_fitted:
            return {}
        return {
            "gene_expression_scaler": self._scaler,
            "view": self._view,
            "output_dim": self._output_dim,
            "fitted": True,
        }

    def set_state(self, state: dict[str, object]) -> None:
        """Restore state from a prior ``get_state`` mapping.

        :param state: state.
        """
        from sklearn.preprocessing import StandardScaler

        scaler = state.get("gene_expression_scaler")
        if isinstance(scaler, StandardScaler):
            self._scaler = scaler
        view = state.get("view")
        if isinstance(view, str):
            self._view = view
        output_dim = state.get("output_dim")
        if isinstance(output_dim, int):
            self._output_dim = output_dim
        if state.get("fitted"):
            self._is_fitted = True
