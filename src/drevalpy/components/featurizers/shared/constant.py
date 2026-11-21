"""Constant (one-category / intercept) featurizer, registered as ``constant`` on both sides."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

import numpy as np

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers.featurizer import Featurizer
from drevalpy.registry.featurizer import register
from drevalpy.types.data.batch.feature_block import BlockSpec, FeaturizerResult, numeric_feature_block
from drevalpy.types.data.feature_source import FeatureSource, Side


class ConstantFeaturizer(Featurizer):
    """Emit a single column of ones for every entity."""

    @classmethod
    def output_blocks(cls, config: Mapping[str, Any]) -> tuple[BlockSpec, ...]:
        """Return the ``constant`` block.

        :param config: Unused construction kwargs.
        :returns: One numeric ``constant`` block.
        """
        _ = config
        return (BlockSpec("constant", FeatureFormat.NUMERIC_MATRIX),)

    def fit(
        self,
        source: FeatureSource,
        train_ids: np.ndarray,
        *,
        train_pair_ids: np.ndarray | None = None,
        es_pair_ids: np.ndarray | None = None,
    ) -> Self:
        """Fit nothing: a constant column has no state.

        :param source: Unused.
        :param train_ids: Unused.
        :param train_pair_ids: Unused.
        :param es_pair_ids: Unused.
        :returns: This featurizer.
        """
        _ = source, train_ids, train_pair_ids, es_pair_ids
        return self

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
        """Return a ``(len(ids), 1)`` float32 ``constant`` block of ones.

        :param source: Unused.
        :param ids: Identifiers to emit a row for.
        :returns: Result with one numeric block named ``constant``.
        """
        _ = source
        matrix = np.ones((len(ids), 1), dtype=np.float32)
        return FeaturizerResult(ids, {"constant": numeric_feature_block(matrix)}, ("constant",))


@register("constant", side=Side.SAMPLE, description="Constant one-column intercept features with no sample identity.")
class SampleConstantFeaturizer(ConstantFeaturizer):
    """A column of ones for every sample."""


@register("constant", side=Side.DRUG, description="Constant one-column intercept features with no drug identity.")
class DrugConstantFeaturizer(ConstantFeaturizer):
    """A column of ones for every drug."""
