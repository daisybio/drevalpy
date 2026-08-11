"""Single-view drug featurizer."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.core.batch.feature_block import FeatureBlock, numeric_feature_block
from drevalpy.components.featurizers._feature_source import FeatureSource
from drevalpy.components.featurizers._matrix import stack_view_matrix
from drevalpy.components.featurizers.drug.base import DrugFeaturizer
from drevalpy.components.registry import register_drug_featurizer


@register_drug_featurizer(
    "view",
    description="Pass through one dense drug view from a FeatureSource.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class ViewDrugFeaturizer(DrugFeaturizer):
    """Featurize one drug view without additional transformation."""

    input_views: ClassVar[tuple[str, ...]] = ("morgan_fingerprint",)

    def __init__(self, *, view: str = "morgan_fingerprint") -> None:
        """Initialize instance state.

        :param view: view.
        """
        self._view = view
        self._output_dim = 0

    def _fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        pair_expanded_ids: np.ndarray | None = None,
        pair_expanded_es_ids: np.ndarray | None = None,
    ) -> ViewDrugFeaturizer:
        """Fit on training data.

        :param source: Feature source providing drug views.
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

        :param source: Feature source providing drug views.
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

        :param source: Feature source providing drug views.
        :param entity_ids: entity ids.
        :returns: Result.
        """
        block_name = self._view
        if hasattr(self, "output_block_specs") and self.output_block_specs:
            block_name = self.output_block_specs[0].name
        return {
            block_name: numeric_feature_block(
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
