"""One-hot encoding of entity IDs, registered as ``oneHot`` on both sides."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

import numpy as np

from drevalpy.components.contracts.feature_format import BlockSpec
from drevalpy.components.featurizers.featurizer import Featurizer
from drevalpy.registry.featurizer import register
from drevalpy.types.data.feature_block import FeatureBlock, FeaturizerOutput
from drevalpy.types.data.feature_source import FeatureSource, Side

#: Name of the featurizer and of the block it emits.
ONE_HOT = "oneHot"


class OneHotCategoryEncoder:
    """Fit a fixed category vocabulary and emit dense one-hot rows."""

    def __init__(self) -> None:
        """Initialize instance state."""
        self._category_to_index: dict[str, int] = {}

    @property
    def categories(self) -> list[str]:
        """Return fitted category labels in index order."""
        return [category for category, _ in sorted(self._category_to_index.items(), key=lambda item: item[1])]

    @property
    def output_dim(self) -> int:
        """Return the number of fitted categories."""
        return len(self._category_to_index)

    def fit_categories(self, categories: np.ndarray) -> None:
        """Learn the category vocabulary from observed values.

        :param categories: Observed values; each is stringified and deduplicated.
        """
        unique = sorted({str(category) for category in np.asarray(categories).reshape(-1)})
        self._category_to_index = {category: index for index, category in enumerate(unique)}

    def transform(self, categories: np.ndarray, *, unknown_zero: bool = True) -> np.ndarray:
        """Encode values as one-hot rows against the fitted vocabulary.

        :param categories: Values to encode, one per output row.
        :param unknown_zero: Emit an all-zero row for unseen categories instead of raising.
        :returns: Float32 matrix of shape ``(len(categories), output_dim)``.
        :raises KeyError: If a category is unseen and *unknown_zero* is false.
        """
        if self.output_dim == 0:
            return np.empty((len(categories), 0), dtype=np.float32)
        matrix = np.zeros((len(categories), self.output_dim), dtype=np.float32)
        for row, category in enumerate(np.asarray(categories).reshape(-1)):
            index = self._category_to_index.get(str(category))
            if index is None:
                if not unknown_zero:
                    msg = f"Unknown category {category!r} for one-hot featurizer"
                    raise KeyError(msg)
                continue
            matrix[row, index] = 1.0
        return matrix


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
        return (BlockSpec(ONE_HOT),)

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

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerOutput:
        """Emit the ``oneHot`` block and the fitted category labels as metadata.

        :param source: Unused.
        :param ids: Identifiers to encode; unseen identifiers become all-zero rows.
        :returns: Result with ``oneHot`` and ``oneHot_categories`` blocks.
        """
        _ = source
        return FeaturizerOutput(
            ids,
            {
                ONE_HOT: FeatureBlock(self._encoder.transform(ids)),
                f"{ONE_HOT}_categories": FeatureBlock(
                    np.asarray(self._encoder.categories, dtype=str), entity_aligned=False
                ),
            },
        )


@register(ONE_HOT, side=Side.SAMPLE, description="One-hot encoding of sample identifiers.")
class SampleOneHotFeaturizer(OneHotFeaturizer):
    """One-hot encoding of sample IDs."""


@register(ONE_HOT, side=Side.DRUG, description="One-hot encoding of drug identifiers.")
class DrugOneHotFeaturizer(OneHotFeaturizer):
    """One-hot encoding of drug IDs."""
