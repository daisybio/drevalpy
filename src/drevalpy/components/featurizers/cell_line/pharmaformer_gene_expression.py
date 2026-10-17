"""PharmaFormer gene-expression preprocessing featurizer."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers.cell_line.base import DenseViewCellLineFeaturizer
from drevalpy.registry.cell_line_featurizer import register
from drevalpy.types.data.batch.feature_block import BlockSpec
from drevalpy.types.data.feature_source import FeatureSource


@register(
    "pharmaFormerGeneExpression",
    description="Reduced landmark genes scaled for PharmaFormer.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class PharmaFormerGeneExpressionFeaturizer(DenseViewCellLineFeaturizer):
    """Apply the PharmaFormer StandardScaler then MinMaxScaler sequence."""

    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX),)
    input_views: ClassVar[tuple[str, ...]] = ("gene_expression",)
    requires_fit: ClassVar[bool] = True

    def __init__(self) -> None:
        """Initialize StandardScaler and MinMaxScaler pipelines."""
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        super().__init__()
        self._scaler = StandardScaler()
        self._minmax = MinMaxScaler()
        self._feature_names: tuple[str, ...] | None = None

    def _fit_entity_ids(
        self,
        source: FeatureSource,
        entity_ids: np.ndarray | None,
        pair_expanded_ids: np.ndarray | None,
        pair_expanded_es_ids: np.ndarray | None,
    ) -> np.ndarray:
        """Fit on the pair-expanded training IDs, matching the reference pipeline.

        :param source: Feature source providing view matrices.
        :param entity_ids: Unused; PharmaFormer fits on the pair-expanded IDs.
        :param pair_expanded_ids: Training entity IDs with duplicates per response pair.
        :param pair_expanded_es_ids: Unused early-stopping IDs.
        :returns: The pair-expanded training IDs.
        :raises ValueError: If *pair_expanded_ids* is missing.
        """
        _ = source, entity_ids, pair_expanded_es_ids
        if pair_expanded_ids is None:
            raise ValueError("pharmaFormerGeneExpression requires pair_expanded_ids")
        return pair_expanded_ids

    def _on_precomputed_fit(self, source: FeatureSource) -> None:
        """Record the source's feature names alongside a stored variant.

        :param source: Feature source carrying the stored variant.
        """
        self._feature_names = source.get_feature_names(self._view)

    def _fit_state(self, source: FeatureSource, entity_ids: np.ndarray) -> int:
        """Fit both scalers on the pair-expanded training rows.

        :param source: Feature source providing view matrices.
        :param entity_ids: Pair-expanded training entity IDs.
        :returns: Output feature dimension.
        """
        matrix = self._raw_matrix(source, entity_ids)
        self._minmax.fit(self._scaler.fit_transform(matrix))
        self._feature_names = source.get_feature_names(self._view)
        return int(matrix.shape[1])

    def _compute_matrix(self, source: FeatureSource, matrix: np.ndarray) -> np.ndarray:
        """Apply the fitted scalers to *matrix*.

        :param source: Feature source the matrix came from.
        :param matrix: Raw view matrix for the requested entity IDs.
        :returns: Scaled feature matrix.
        """
        _ = source
        return self._minmax.transform(self._scaler.transform(matrix))

    def _block_feature_names(self, source: FeatureSource) -> tuple[str, ...] | None:
        """Return the feature names captured at fit time.

        :param source: Feature source (unused; names are recorded during fit).
        :returns: Recorded feature names, or ``None``.
        """
        _ = source
        return self._feature_names

    def get_state(self) -> dict[str, object]:
        """Serialize scaler state and feature names.

        :returns: Fitted state mapping, or empty dict before fitting.
        """
        if not self._is_fitted:
            return {}
        return {
            "scaler": self._scaler,
            "minmax": self._minmax,
            "feature_names": self._feature_names,
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
        names = state.get("feature_names")
        if isinstance(names, tuple):
            self._feature_names = tuple(str(name) for name in names)
        output_dim = state.get("output_dim")
        if isinstance(output_dim, int):
            self._output_dim = output_dim
        self._is_fitted = bool(state.get("fitted"))
