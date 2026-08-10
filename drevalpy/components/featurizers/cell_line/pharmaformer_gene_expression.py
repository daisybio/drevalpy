"""PharmaFormer gene-expression preprocessing featurizer."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from drevalpy.components.core.batch.feature_block import BlockSpec, FeatureBlock, numeric_feature_block
from drevalpy.components.core.contracts.contracts import FeatureFormat
from drevalpy.components.core.features.feature_source import FeatureSource
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer


@register_cell_line_featurizer(
    "pharmaFormerGeneExpression",
    description="Reduced landmark genes scaled for PharmaFormer.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class PharmaFormerGeneExpressionFeaturizer(CellLineFeaturizer):
    """Apply the PharmaFormer StandardScaler then MinMaxScaler sequence."""

    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX),)
    input_views: ClassVar[tuple[str, ...]] = ("gene_expression",)

    def __init__(self) -> None:
        """Initialize StandardScaler and MinMaxScaler pipelines."""
        self._scaler = StandardScaler()
        self._minmax = MinMaxScaler()
        self._feature_names: tuple[str, ...] | None = None
        self._output_dim = 0
        self._is_fitted = False

    def fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        pair_expanded_ids: np.ndarray | None = None,
        pair_expanded_es_ids: np.ndarray | None = None,
    ) -> PharmaFormerGeneExpressionFeaturizer:
        """Fit StandardScaler and MinMaxScaler on pair-expanded training ids.

        :param source: Feature source providing view matrices.
        :param entity_ids: Unused; training ids come from *pair_expanded_ids*.
        :param pair_expanded_ids: Training entity IDs with duplicates per response pair.
        :param pair_expanded_es_ids: Unused early-stopping IDs.
        :returns: Fitted featurizer instance.
        :raises ValueError: If *pair_expanded_ids* is missing.
        """
        _ = entity_ids, pair_expanded_es_ids
        if pair_expanded_ids is None:
            raise ValueError("pharmaFormerGeneExpression requires pair_expanded_ids")
        matrix = source.get_view_matrix("gene_expression", pair_expanded_ids)
        self._minmax.fit(self._scaler.fit_transform(matrix))
        self._feature_names = source.get_feature_names("gene_expression")
        self._output_dim = int(matrix.shape[1])
        self._is_fitted = True
        return self

    def transform(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Apply fitted scalers to gene-expression rows.

        :param source: Feature source providing view matrices.
        :param entity_ids: Cell-line identifiers to transform.
        :returns: Scaled float matrix.
        :raises RuntimeError: If called before ``fit``.
        """
        if not self._is_fitted:
            raise RuntimeError("PharmaFormerGeneExpressionFeaturizer must be fit before transform")
        matrix = source.get_view_matrix("gene_expression", entity_ids)
        return self._minmax.transform(self._scaler.transform(matrix)).astype(np.float32)

    def transform_blocks(self, source: FeatureSource, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        """Return a single ``gene_expression`` numeric block.

        :param source: Feature source providing view matrices.
        :param entity_ids: Cell-line identifiers to transform.
        :returns: Mapping with one numeric block.
        """
        return {
            "gene_expression": numeric_feature_block(
                self.transform(source, entity_ids),
                feature_names=self._feature_names,
            )
        }

    @property
    def output_dim(self) -> int:
        """Return landmark gene count after fitting.

        :returns: Output feature dimensionality.
        """
        return self._output_dim

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
