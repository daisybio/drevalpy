"""One-hot encoding of entity IDs, registered as ``oneHot`` on both sides."""

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
from drevalpy.types.data.feature_source import FeatureSource, Side

#: Name of the featurizer and of the block it emits.
ONE_HOT = "oneHot"


class OneHotFeaturizer(Featurizer):
    """Encode entity IDs as dense one-hot vectors over the training entities."""

    def __init__(self) -> None:
        """Initialize instance state."""
        self._encoder = OneHotCategoryEncoder()

    @classmethod
    def output_blocks(cls, config: Mapping[str, Any]) -> tuple[BlockSpec, ...]:
        """Return the ``oneHot`` block.

        :param config: Unused construction kwargs.
        :returns: One numeric ``oneHot`` block.
        """
        _ = config
        return (BlockSpec(ONE_HOT, FeatureFormat.NUMERIC_MATRIX),)

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
        """Emit the ``oneHot`` block and the fitted category labels as metadata.

        :param source: Unused.
        :param ids: Identifiers to encode; unseen identifiers become all-zero rows.
        :returns: Result with ``oneHot`` and ``oneHot_categories`` blocks.
        """
        _ = source
        return FeaturizerResult(
            ids,
            {
                ONE_HOT: numeric_feature_block(self._encoder.transform(ids)),
                f"{ONE_HOT}_categories": global_feature_block(np.asarray(self._encoder.categories, dtype=str)),
            },
            (ONE_HOT,),
        )


@register(ONE_HOT, side=Side.SAMPLE, description="One-hot encoding of sample identifiers.")
class SampleOneHotFeaturizer(OneHotFeaturizer):
    """One-hot encoding of sample IDs."""


@register(ONE_HOT, side=Side.DRUG, description="One-hot encoding of drug identifiers.")
class DrugOneHotFeaturizer(OneHotFeaturizer):
    """One-hot encoding of drug IDs."""
