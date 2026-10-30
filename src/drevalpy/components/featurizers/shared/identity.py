"""One-hot identity featurizer, for either entity side."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

import numpy as np

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers._one_hot import OneHotCategoryEncoder
from drevalpy.components.featurizers.featurizer import Featurizer
from drevalpy.registry.featurizer import register
from drevalpy.types.data.batch.feature_block import (
    BlockSpec,
    FeaturizerResult,
    global_feature_block,
    numeric_feature_block,
)
from drevalpy.types.data.feature_source import FeatureSource


@register("identity", description="One-hot encoding of entity identifiers.")
class IdentityFeaturizer(Featurizer):
    """Encode entity IDs as dense one-hot vectors over the training entities."""

    def __init__(self) -> None:
        """Initialize instance state."""
        self._encoder = OneHotCategoryEncoder()

    @classmethod
    def output_blocks(cls, config: Mapping[str, Any]) -> tuple[BlockSpec, ...]:
        """Return the ``identity`` block.

        :param config: Unused construction kwargs.
        :returns: One numeric ``identity`` block.
        """
        _ = config
        return (BlockSpec("identity", FeatureFormat.NUMERIC_MATRIX),)

    def fit(
        self,
        source: FeatureSource,
        train_ids: np.ndarray,
        *,
        train_pair_ids: np.ndarray | None = None,
        es_pair_ids: np.ndarray | None = None,
    ) -> Self:
        """Learn the category order from the training entity IDs.

        :param source: Unused.
        :param train_ids: Training entity IDs.
        :param train_pair_ids: Unused.
        :param es_pair_ids: Unused.
        :returns: This featurizer, fitted.
        """
        _ = source, train_pair_ids, es_pair_ids
        self._encoder.fit_categories(train_ids)
        return self

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
        """Emit the one-hot ``identity`` block and the fitted category labels as metadata.

        :param source: Unused.
        :param ids: Identifiers to encode; unseen identifiers become all-zero rows.
        :returns: Result with ``identity`` and ``identity_categories`` blocks.
        """
        _ = source
        return FeaturizerResult(
            ids,
            {
                "identity": numeric_feature_block(self._encoder.transform(ids)),
                "identity_categories": global_feature_block(np.asarray(self._encoder.categories, dtype=str)),
            },
            ("identity",),
        )
