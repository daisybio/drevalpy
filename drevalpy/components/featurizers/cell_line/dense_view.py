"""Shared base for dense single-view cell-line featurizers."""

from __future__ import annotations

import numpy as np

from drevalpy.components.core.batch.feature_block import FeatureBlock, numeric_feature_block
from drevalpy.components.featurizers._feature_source import FeatureSource
from drevalpy.components.featurizers._matrix import feature_names_for_view, stack_view_matrix
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer


class DenseViewCellLineFeaturizer(CellLineFeaturizer):
    """Pass through one dense cell-line view without additional transformation."""

    def __init__(self, *, view: str | None = None) -> None:
        """Initialize instance state.

        :param view: view.
        """
        self._view = view or self.resolve_input_views()[0]
        self._output_dim = 0

    def _fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        pair_expanded_ids: np.ndarray | None = None,
        pair_expanded_es_ids: np.ndarray | None = None,
    ) -> DenseViewCellLineFeaturizer:
        """Fit on training data.

        :param source: Feature source providing views for the entity type.
        :param entity_ids: entity ids.
        :param pair_expanded_ids: Unused training IDs with duplicates.
        :param pair_expanded_es_ids: Unused early-stopping IDs.
        :returns: Result.
        """
        _ = pair_expanded_ids, pair_expanded_es_ids
        ids = entity_ids if entity_ids is not None else source.identifiers
        mdata = getattr(source, "mdata", None)
        matrix = self.fetch(mdata, ids) if mdata is not None else None
        if matrix is not None:
            self._output_dim = int(matrix.shape[1])
            return self
        try:
            matrix = stack_view_matrix(source, self._view, ids)
            self._output_dim = int(matrix.shape[1])
            return self
        except (KeyError, TypeError, ValueError):
            if self.precompute and hasattr(self, "_compute_from_source"):
                probe = self._compute_from_source(source, ids[:1])
                self._output_dim = int(probe.shape[1])
                return self
            raise

    def _transform(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Transform inputs into feature payloads.

        :param source: Feature source providing views for the entity type.
        :param entity_ids: entity ids.
        :returns: Result.
        """
        mdata = getattr(source, "mdata", None)
        matrix = self.fetch(mdata, entity_ids) if mdata is not None else None
        if matrix is not None:
            return matrix.astype(np.float32)
        try:
            return stack_view_matrix(source, self._view, entity_ids).astype(np.float32)
        except (KeyError, TypeError, ValueError):
            if self.precompute and hasattr(self, "_compute_from_source"):
                return self._compute_from_source(source, entity_ids).astype(np.float32)
            raise

    def _transform_blocks(self, source: FeatureSource, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        """Transform blocks.

        :param source: Feature source providing views for the entity type.
        :param entity_ids: entity ids.
        :returns: Result.
        """
        return {
            self._view: numeric_feature_block(
                self._transform(source, entity_ids),
                feature_names=feature_names_for_view(source, self._view),
            )
        }

    @property
    def output_dim(self) -> int:
        """Return output feature dimension after fitting.

        :returns: Result.
        """
        return self._output_dim
