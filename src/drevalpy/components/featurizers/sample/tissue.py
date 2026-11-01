"""Tissue metadata featurizer for samples."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Self

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

_MISSING_MESSAGE = "TissueFeaturizer requires tissue annotations in sample_input"


def _tissue_labels(source: FeatureSource, ids: np.ndarray) -> list[str | None]:
    """Return the tissue label of every sample, ``None`` where it has none."""
    raw = source.tissue(ids)
    if raw is None:
        return [None] * len(ids)
    return [None if label is None or (isinstance(label, float) and np.isnan(label)) else str(label) for label in raw]


@register("tissue", description="One-hot encoding of tissue or lineage labels for sample entities.")
class TissueFeaturizer(Featurizer):
    """Map each sample to a dense one-hot tissue vector."""

    sides: ClassVar[frozenset[Side]] = frozenset({"sample"})

    def __init__(self, *, allow_missing: bool = False) -> None:
        """Initialize instance state.

        :param allow_missing: Treat samples without a tissue annotation as an extra ``__missing__``
            category instead of raising.
        """
        self._encoder = OneHotCategoryEncoder()
        self._allow_missing = bool(allow_missing)

    @classmethod
    def output_blocks(cls, config: Mapping[str, Any]) -> tuple[BlockSpec, ...]:
        """Declare the one-hot ``tissue`` block; the global ``tissue_categories`` labels ride along undeclared.

        :param config: Unused.
        :returns: The ``tissue`` block spec.
        """
        _ = config
        return (BlockSpec("tissue", FeatureFormat.NUMERIC_MATRIX),)

    def fit(
        self,
        source: FeatureSource,
        train_ids: np.ndarray,
        *,
        train_pair_ids: np.ndarray | None = None,
        es_pair_ids: np.ndarray | None = None,
    ) -> Self:
        """Learn the tissue vocabulary from the training samples.

        :param source: Feature source of the sample side.
        :param train_ids: Training sample IDs.
        :param train_pair_ids: Unused.
        :param es_pair_ids: Unused.
        :returns: This featurizer, fitted.
        :raises ValueError: If a sample lacks a tissue annotation and *allow_missing* is false, or
            no sample has one.
        """
        _ = train_pair_ids, es_pair_ids
        available: list[str] = []
        for label in _tissue_labels(source, train_ids):
            if label is None:
                if not self._allow_missing:
                    raise ValueError(_MISSING_MESSAGE)
                continue
            available.append(label)
        if not available and not self._allow_missing:
            raise ValueError(_MISSING_MESSAGE)
        self._encoder.fit_categories(np.asarray(available, dtype=str))
        return self

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
        """Emit the one-hot ``tissue`` block and the fitted category labels as a global block.

        :param source: Feature source of the sample side.
        :param ids: Sample IDs, in output order.
        :returns: Result with ``tissue`` and ``tissue_categories`` blocks.
        :raises ValueError: If a sample lacks a tissue annotation and *allow_missing* is false.
        """
        if self._encoder.output_dim == 0:
            matrix = np.empty((len(ids), 0), dtype=np.float32)
        else:
            categories: list[str] = []
            for label in _tissue_labels(source, ids):
                if label is None:
                    if not self._allow_missing:
                        raise ValueError(_MISSING_MESSAGE)
                    label = "__missing__"
                categories.append(label)
            matrix = self._encoder.transform(np.asarray(categories, dtype=str))
        return FeaturizerResult(
            ids,
            {
                "tissue": numeric_feature_block(matrix),
                "tissue_categories": global_feature_block(np.asarray(self._encoder.categories, dtype=str)),
            },
            ("tissue",),
        )
