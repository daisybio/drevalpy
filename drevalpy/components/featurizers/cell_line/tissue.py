"""Tissue metadata featurizer for cell lines."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.feature_block import FeatureBlock, metadata_feature_block, numeric_feature_block
from drevalpy.components.feature_source import FeatureSource
from drevalpy.components.featurizer_fit_context import FeaturizerFitContext
from drevalpy.components.featurizers._one_hot import OneHotCategoryEncoder
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer
from drevalpy.datasets.utils import TISSUE_IDENTIFIER


def _tissue_label(source: FeatureSource, entity_id: str) -> str | None:
    raw = source.get_entity_view(str(entity_id), TISSUE_IDENTIFIER)
    if raw is None:
        return None
    return str(np.asarray(raw).reshape(-1)[0])


@register_cell_line_featurizer(
    "tissue",
    description="One-hot encoding of tissue or lineage labels for cell-line entities.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class TissueFeaturizer(CellLineFeaturizer):
    """Map each cell line to a dense one-hot tissue vector."""

    input_views: ClassVar[tuple[str, ...]] = ()

    def __init__(self, *, allow_missing: bool = False) -> None:
        """Initialize instance state.

        :param allow_missing: allow missing.
        """
        self._encoder = OneHotCategoryEncoder()
        self._allow_missing = bool(allow_missing)

    def fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        context: FeaturizerFitContext | None = None,
    ) -> TissueFeaturizer:
        """Fit on training data.

        :param source: Feature source providing views for the entity type.
        :param entity_ids: entity ids.
        :param context: context.
        :returns: Result.
        :raises ValueError: Raised on invalid input.
        """
        _ = context
        ids = entity_ids if entity_ids is not None else source.identifiers
        available: list[str] = []
        for entity_id in ids:
            label = _tissue_label(source, str(entity_id))
            if label is None:
                if not self._allow_missing:
                    msg = "TissueFeaturizer requires tissue annotations in cell_line_input"
                    raise ValueError(msg)
                continue
            available.append(label)
        if not available:
            if self._allow_missing:
                self._encoder.fit_categories(np.array([], dtype=str))
                return self
            msg = "TissueFeaturizer requires tissue annotations in cell_line_input"
            raise ValueError(msg)
        self._encoder.fit_categories(np.asarray(available, dtype=str))
        return self

    def transform(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Transform inputs into feature payloads.

        :param source: Feature source providing views for the entity type.
        :param entity_ids: entity ids.
        :returns: Result.
        :raises ValueError: Raised on invalid input.
        """
        if self._encoder.output_dim == 0:
            return np.empty((len(entity_ids), 0), dtype=np.float32)
        categories: list[str] = []
        for entity_id in entity_ids:
            label = _tissue_label(source, str(entity_id))
            if label is None:
                if not self._allow_missing:
                    msg = "TissueFeaturizer requires tissue annotations in cell_line_input"
                    raise ValueError(msg)
                categories.append("__missing__")
            else:
                categories.append(label)
        return self._encoder.transform(np.asarray(categories, dtype=str))

    def transform_blocks(
        self,
        source: FeatureSource,
        entity_ids: np.ndarray,
    ) -> dict[str, FeatureBlock]:
        """Transform blocks.

        :param source: Feature source providing views for the entity type.
        :param entity_ids: entity ids.
        :returns: Result.
        """
        return {
            "tissue": numeric_feature_block(self.transform(source, entity_ids)),
            "tissue_categories": metadata_feature_block(
                np.asarray(self._encoder.categories, dtype=str),
            ),
        }

    @property
    def output_dim(self) -> int:
        """Return output feature dimension after fitting.

        :returns: Result.
        """
        return self._encoder.output_dim

    def get_state(self) -> dict[str, object]:
        """Return serializable fitted state.

        :returns: Result.
        """
        return self._encoder.get_state()

    def set_state(self, state: dict[str, object]) -> None:
        """Restore state from a prior ``get_state`` mapping.

        :param state: state.
        """
        self._encoder.set_state(state)
