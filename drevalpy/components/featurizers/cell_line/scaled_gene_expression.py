"""Scaled gene-expression featurizer for cell lines."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from sklearn.preprocessing import StandardScaler

from drevalpy.components.core.batch.feature_block import FeatureBlock, numeric_feature_block
from drevalpy.components.core.contracts.contracts import FeatureFormat
from drevalpy.components.core.features.feature_source import FeatureSource
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer


@register_cell_line_featurizer(
    "scaledGeneExpression",
    description="Landmark gene expression with arcsinh transform and scaling.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class ScaledGeneExpressionFeaturizer(CellLineFeaturizer):
    """Match sklearn baseline gene-expression preprocessing."""

    input_views: ClassVar[tuple[str, ...]] = ("gene_expression",)

    def __init__(self, *, view: str = "gene_expression") -> None:
        """Initialize instance state.

        :param view: view.
        """
        self._view = view
        self._scaler = StandardScaler()
        self._output_dim = 0
        self._is_fitted = False

    def _fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        pair_expanded_ids: np.ndarray | None = None,
        pair_expanded_es_ids: np.ndarray | None = None,
    ) -> ScaledGeneExpressionFeaturizer:
        """Fit on training data.

        :param source: Feature source providing view matrices.
        :param entity_ids: entity ids.
        :param pair_expanded_ids: Unused training IDs with duplicates.
        :param pair_expanded_es_ids: Unused early-stopping IDs.
        :returns: Result.
        """
        _ = pair_expanded_ids, pair_expanded_es_ids
        ids = entity_ids if entity_ids is not None else source.identifiers
        mdata = getattr(source, "mdata", None)
        precomputed = self.fetch(mdata, ids) if mdata is not None else None
        if precomputed is not None:
            self._output_dim = int(precomputed.shape[1])
            self._is_fitted = True
            return self
        matrix = np.arcsinh(source.get_view_matrix(self._view, np.unique(ids)))
        self._scaler.fit(matrix)
        self._output_dim = int(matrix.shape[1])
        self._is_fitted = True
        return self

    def _transform(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Transform inputs into feature payloads.

        :param source: Feature source providing view matrices.
        :param entity_ids: entity ids.
        :returns: Result.
        :raises RuntimeError: Raised on invalid input.
        """
        if not self._is_fitted:
            msg = "ScaledGeneExpressionFeaturizer must be fit before transform"
            raise RuntimeError(msg)
        mdata = getattr(source, "mdata", None)
        precomputed = self.fetch(mdata, entity_ids) if mdata is not None else None
        if precomputed is not None:
            return precomputed.astype(np.float32)
        matrix = np.arcsinh(source.get_view_matrix(self._view, entity_ids))
        return self._scaler.transform(matrix).astype(np.float32)

    def _transform_blocks(self, source: FeatureSource, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        """Transform blocks.

        :param source: Feature source providing view matrices.
        :param entity_ids: entity ids.
        :returns: Result.
        :raises RuntimeError: Raised on invalid input.
        """
        if not self._is_fitted:
            msg = "ScaledGeneExpressionFeaturizer must be fit before transform"
            raise RuntimeError(msg)
        return {
            "gene_expression": numeric_feature_block(
                self._transform(source, entity_ids),
                feature_names=source.get_feature_names(self._view),
            )
        }

    @property
    def output_dim(self) -> int:
        """Return output feature dimension after fitting.

        :returns: Result.
        """
        return self._output_dim

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
