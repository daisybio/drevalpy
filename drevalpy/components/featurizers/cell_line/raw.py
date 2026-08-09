"""Generic raw dense pass-through featurizer for one omics view."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.feature_block import FeatureBlock, numeric_feature_block
from drevalpy.components.feature_source import FeatureSource
from drevalpy.components.featurizer_fit_context import FeaturizerFitContext
from drevalpy.components.featurizers._matrix import feature_names_for_view, stack_view_matrix
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer


@register_cell_line_featurizer(
    "raw",
    description="Pass through one dense omics view without preprocessing.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class RawCellLineFeaturizer(CellLineFeaturizer):
    """Featurize one omics view as a dense matrix without transformation."""

    requires_view: ClassVar[bool] = True

    def __init__(self, *, view: str) -> None:
        """Initialize instance state.

        :param view: view.
        :raises ValueError: Raised on invalid input.
        """
        if not view or not view.strip():
            msg = "raw featurizer requires an explicit view"
            raise ValueError(msg)
        self._view = view
        self._output_dim = 0

    def fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        context: FeaturizerFitContext | None = None,
    ) -> RawCellLineFeaturizer:
        """Fit on training data.

        :param source: Feature source providing views for the entity type.
        :param entity_ids: entity ids.
        :param context: context.
        :returns: Result.
        """
        _ = context
        ids = entity_ids if entity_ids is not None else source.identifiers
        matrix = stack_view_matrix(source, self._view, ids)
        self._output_dim = int(matrix.shape[1])
        return self

    def transform(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Transform inputs into feature payloads.

        :param source: Feature source providing views for the entity type.
        :param entity_ids: entity ids.
        :returns: Result.
        """
        return stack_view_matrix(source, self._view, entity_ids).astype(np.float32)

    def transform_blocks(self, source: FeatureSource, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        """Transform blocks.

        :param source: Feature source providing views for the entity type.
        :param entity_ids: entity ids.
        :returns: Result.
        """
        return {
            self._view: numeric_feature_block(
                self.transform(source, entity_ids),
                feature_names=feature_names_for_view(source, self._view),
            )
        }

    @property
    def output_dim(self) -> int:
        """Return output feature dimension after fitting.

        :returns: Result.
        """
        return self._output_dim
