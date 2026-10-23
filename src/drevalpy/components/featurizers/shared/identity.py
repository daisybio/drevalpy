"""One-hot identity featurizer, shared by both entity sides."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers._one_hot import OneHotCategoryEncoder
from drevalpy.components.featurizers._side_binding import register_for_sides
from drevalpy.components.featurizers.base import Featurizer
from drevalpy.types.data.batch.feature_block import (
    BlockSpec,
    FeaturizerResult,
    metadata_feature_block,
    numeric_feature_block,
)
from drevalpy.types.data.feature_source import FeatureSource


@register_for_sides(
    "identity",
    description={
        "sample": "One-hot encoding of sample entity identifiers.",
        "drug": "One-hot encoding of drug entity identifiers.",
    },
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class SharedIdentityFeaturizer(Featurizer):
    """Encode entity IDs as dense one-hot vectors."""

    entity_id_only: ClassVar[bool] = True
    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (BlockSpec("identity", FeatureFormat.NUMERIC_MATRIX),)

    def __init__(self) -> None:
        """Initialize instance state."""
        self._encoder = OneHotCategoryEncoder()

    def _fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        pair_expanded_ids: np.ndarray | None = None,
        pair_expanded_es_ids: np.ndarray | None = None,
    ) -> SharedIdentityFeaturizer:
        """Learn the category order from the training entity IDs.

        :param source: Feature source; only consulted for its identifiers.
        :param entity_ids: Subset of entity identifiers to fit on; ``None`` uses all source identifiers.
        :param pair_expanded_ids: Unused.
        :param pair_expanded_es_ids: Unused.
        :returns: This featurizer, fitted.
        """
        _ = pair_expanded_ids, pair_expanded_es_ids
        ids = entity_ids if entity_ids is not None else source.identifiers
        self._encoder.fit_categories(ids)
        return self

    def _feature_matrix(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Return the one-hot matrix for *entity_ids*.

        :param source: Unused.
        :param entity_ids: Identifiers to encode; unseen identifiers become all-zero rows.
        :returns: Float32 matrix of shape ``(len(entity_ids), output_dim)``.
        """
        _ = source
        return self._encoder.transform(entity_ids)

    def _transform_features(
        self,
        source: FeatureSource,
        entity_ids: np.ndarray,
    ) -> FeaturizerResult:
        """Emit the one-hot ``identity`` block and the fitted category labels as metadata.

        :param source: Unused.
        :param entity_ids: Identifiers to encode.
        :returns: Result with ``identity`` and ``identity_categories`` blocks.
        """
        matrix = self._feature_matrix(source, entity_ids)
        return FeaturizerResult(
            entity_ids,
            {
                "identity": numeric_feature_block(matrix),
                "identity_categories": metadata_feature_block(
                    np.asarray(self._encoder.categories, dtype=str),
                ),
            },
            ("identity",),
        )

    @property
    def output_dim(self) -> int:
        """Return output feature dimension after fitting."""
        return self._encoder.output_dim

    def get_state(self) -> dict[str, object]:
        """Return serializable fitted state."""
        return self._encoder.get_state()

    def set_state(self, state: dict[str, object]) -> None:
        """Restore state from a prior ``get_state`` mapping.

        :param state: Mapping previously returned by ``get_state``.
        """
        self._encoder.set_state(state)
