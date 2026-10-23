"""Scaled gene-expression featurizer for samples."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers.sample.base import DenseViewSampleFeaturizer
from drevalpy.components.featurizers.sample.gene_lists import select_genes
from drevalpy.registry.sample_featurizer import register
from drevalpy.types.data.feature_source import FeatureSource


@register(
    "scaledGeneExpression",
    description="Landmark gene expression with arcsinh transform and scaling.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class ScaledGeneExpressionFeaturizer(DenseViewSampleFeaturizer):
    """Match sklearn baseline gene-expression preprocessing."""

    input_views: ClassVar[tuple[str, ...]] = ("gene_expression",)
    requires_fit: ClassVar[bool] = True
    fit_on_unique_ids: ClassVar[bool] = True

    def __init__(self, *, view: str = "gene_expression", gene_list: str | None = None) -> None:
        """Bind the gene-expression view to scale.

        :param view: Name of the view to arcsinh-transform and standardize.
        :param gene_list: Name of a shipped gene list to keep, in list order; the view must hold every
            gene of it. ``None`` keeps the whole view.
        """
        from sklearn.preprocessing import StandardScaler

        super().__init__(view=view)
        self._gene_list = gene_list
        self._scaler = StandardScaler()

    def _names_to_fit(self, source: FeatureSource) -> tuple[str, ...] | None:
        """Bind the transform to the genes of the gene list, or to the whole view without one.

        :param source: Feature source providing view matrices.
        :returns: Gene names in gene-list order, or every feature name of the view.
        """
        if self._gene_list is None:
            return super()._names_to_fit(source)
        return select_genes(source, self._view, self._gene_list)

    def _fetch_hyperparameters(self) -> dict[str, Any] | None:
        """Match a stored variant only when it was built for the same gene list.

        :returns: The gene-list setting, or ``None`` for the whole view.
        """
        return None if self._gene_list is None else {"gene_list": self._gene_list}

    def _fit_state(self, source: FeatureSource, entity_ids: np.ndarray) -> int:
        """Fit the scaler on arcsinh-transformed training rows.

        :param source: Feature source providing view matrices.
        :param entity_ids: Deduplicated sample identifiers to fit on.
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
        """Return serializable fitted state."""
        if not self._is_fitted:
            return {}
        return {
            "gene_expression_scaler": self._scaler,
            "gene_list": self._gene_list,
            **self._dense_state(),
            "fitted": True,
        }

    def set_state(self, state: dict[str, object]) -> None:
        """Restore state from a prior ``get_state`` mapping.

        :param state: Mapping previously returned by ``get_state``.
        """
        from sklearn.preprocessing import StandardScaler

        scaler = state.get("gene_expression_scaler")
        if isinstance(scaler, StandardScaler):
            self._scaler = scaler
        gene_list = state.get("gene_list")
        if gene_list is None or isinstance(gene_list, str):
            self._gene_list = gene_list
        self._restore_dense_state(state)
