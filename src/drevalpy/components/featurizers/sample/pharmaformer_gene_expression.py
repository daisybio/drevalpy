"""PharmaFormer gene-expression preprocessing featurizer."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers.sample.base import DenseViewSampleFeaturizer
from drevalpy.components.featurizers.sample.gene_lists import select_genes
from drevalpy.registry.sample_featurizer import register
from drevalpy.types.data.batch.feature_block import BlockSpec
from drevalpy.types.data.feature_source import FeatureSource


@register(
    "pharmaFormerGeneExpression",
    description="Reduced landmark genes scaled for PharmaFormer.",
    output_format=FeatureFormat.NUMERIC_MATRIX,
)
class PharmaFormerGeneExpressionFeaturizer(DenseViewSampleFeaturizer):
    """Apply the PharmaFormer StandardScaler then MinMaxScaler sequence, fitted once per training sample."""

    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX),)
    input_views: ClassVar[tuple[str, ...]] = ("gene_expression",)
    requires_fit: ClassVar[bool] = True
    fit_on_unique_ids: ClassVar[bool] = True

    def __init__(self, *, gene_list: str | None = None) -> None:
        """Initialize StandardScaler and MinMaxScaler pipelines.

        :param gene_list: Name of a shipped gene list to keep, in list order; the view must hold every
            gene of it. ``None`` keeps the whole view.
        """
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        super().__init__()
        self._gene_list = gene_list
        self._scaler = StandardScaler()
        self._minmax = MinMaxScaler()

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
        """Fit both scalers on the unique training samples, so a sample with many pairs counts once.

        :param source: Feature source providing view matrices.
        :param entity_ids: Deduplicated training sample IDs.
        :returns: Output feature dimension.
        """
        matrix = self._raw_matrix(source, entity_ids)
        self._minmax.fit(self._scaler.fit_transform(matrix))
        return int(matrix.shape[1])

    def _compute_matrix(self, source: FeatureSource, matrix: np.ndarray) -> np.ndarray:
        """Apply the fitted scalers to *matrix*.

        :param source: Feature source the matrix came from.
        :param matrix: Raw view matrix for the requested entity IDs.
        :returns: Scaled feature matrix.
        """
        _ = source
        return self._minmax.transform(self._scaler.transform(matrix))

    def get_state(self) -> dict[str, object]:
        """Serialize scaler state and feature names.

        :returns: Fitted state mapping, or empty dict before fitting.
        """
        if not self._is_fitted:
            return {}
        return {
            "scaler": self._scaler,
            "minmax": self._minmax,
            "gene_list": self._gene_list,
            "view_feature_names": self._view_names,
            "output_dim": self._output_dim,
            "fitted": True,
        }

    def set_state(self, state: dict[str, object]) -> None:
        """Restore scaler state from ``get_state``.

        :param state: Mapping previously returned by ``get_state``.
        """
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        scaler, minmax = state.get("scaler"), state.get("minmax")
        if isinstance(scaler, StandardScaler):
            self._scaler = scaler
        if isinstance(minmax, MinMaxScaler):
            self._minmax = minmax
        gene_list = state.get("gene_list")
        if gene_list is None or isinstance(gene_list, str):
            self._gene_list = gene_list
        self._restore_view_names(state)
        output_dim = state.get("output_dim")
        if isinstance(output_dim, int):
            self._output_dim = output_dim
        # Not _restore_dense_state: here an absent "fitted" key resets the flag to False.
        self._is_fitted = bool(state.get("fitted"))
