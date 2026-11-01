"""Concatenating featurizer, registered as ``concatFeaturizers`` on both sides.

Each child is fitted and transformed on the entities that are usable for every child, and the
children's numeric blocks are merged into one result.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

import numpy as np

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers._featurizer_label import CONCAT
from drevalpy.components.featurizers.featurizer import Featurizer
from drevalpy.registry.featurizer import register
from drevalpy.types.data.batch.feature_block import BlockSpec, FeaturizerResult, merge_feature_blocks
from drevalpy.types.data.feature_source import FeatureSource, Side


def _build_children(featurizers: list[Any], side: Side | None) -> list[Featurizer]:
    """Return the child featurizers: built instances as they are, configs instantiated.

    :param featurizers: Built featurizers, or config payloads (mappings or ``FeaturizerConfig``).
    :param side: Side of the concat featurizer, given to child configs that set none.
    :returns: Child featurizer instances.
    """
    if all(isinstance(item, Featurizer) for item in featurizers):
        return list(featurizers)
    from drevalpy.models.config.featurizer import FeaturizerConfig

    children = [
        item if isinstance(item, FeaturizerConfig) else FeaturizerConfig.parse(item, side) for item in featurizers
    ]
    parent = FeaturizerConfig(name=CONCAT, side=side or children[0].side, featurizers=tuple(children))
    return [child.create_instance() for child in parent.featurizers or ()]


class ConcatFeaturizer(Featurizer):
    """Fit child featurizers independently and concatenate their dense outputs."""

    def __init__(self, *, featurizers: list[Any] | None = None) -> None:
        """Bind the child featurizers.

        :param featurizers: Child featurizers, either built instances or configs of this side; must be non-empty.
        :raises ValueError: If *featurizers* is empty.
        """
        if not featurizers:
            msg = "featurizers must be a non-empty list"
            raise ValueError(msg)
        self._children = _build_children(featurizers, getattr(type(self), "side", None))

    @classmethod
    def output_blocks(cls, config: Mapping[str, Any]) -> tuple[BlockSpec, ...]:
        """Return no blocks of its own: the config layer collects the children's blocks.

        :param config: Unused construction kwargs.
        :returns: An empty tuple.
        """
        _ = config
        return ()

    @classmethod
    def views(cls, config: Mapping[str, Any]) -> tuple[str, ...]:
        """Return no views of its own: the config layer collects the children's views.

        :param config: Unused construction kwargs.
        :returns: An empty tuple.
        """
        _ = config
        return ()

    def valid_mask(self, source: FeatureSource, ids: np.ndarray) -> np.ndarray:
        """Require usable input for every child.

        :param source: Feature source of the featurizer's side.
        :param ids: Entity IDs to check.
        :returns: Boolean array with one entry per ID.
        """
        return np.logical_and.reduce([child.valid_mask(source, ids) for child in self._children])

    def fit(
        self,
        source: FeatureSource,
        train_ids: np.ndarray,
        *,
        train_pair_ids: np.ndarray | None = None,
        es_pair_ids: np.ndarray | None = None,
    ) -> Self:
        """Fit every child on the same usable entities.

        :param source: Feature source of the featurizer's side.
        :param train_ids: Training entity IDs usable for every child.
        :param train_pair_ids: Entity ID of every usable training pair.
        :param es_pair_ids: Entity ID of every usable early-stopping pair.
        :returns: This featurizer, fitted.
        :raises ValueError: If a child requires early stopping and no usable early-stopping pair is given.
        """
        for child in self._children:
            if child.requires_early_stopping and (es_pair_ids is None or len(es_pair_ids) == 0):
                raise ValueError(f"{type(child).__name__} requires a usable early-stopping holdout")
            child.fit(source, train_ids, train_pair_ids=train_pair_ids, es_pair_ids=es_pair_ids)
        return self

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
        """Transform with every child and merge their blocks.

        :param source: Feature source of the featurizer's side.
        :param ids: Entity IDs usable for every child, in output order.
        :returns: The children's feature blocks merged into one result.
        :raises ValueError: If a child emits a block that is not a numeric matrix.
        """
        child_blocks = []
        for child in self._children:
            blocks = child.transform(source, ids).blocks
            wrong = [name for name, block in blocks.items() if block.format != FeatureFormat.NUMERIC_MATRIX]
            if wrong:
                msg = f"concat featurizer child {type(child).__name__} emits non-numeric blocks {wrong}"
                raise ValueError(msg)
            child_blocks.append(blocks)
        return FeaturizerResult.from_blocks(merge_feature_blocks(*child_blocks), ids)


@register(CONCAT, side=Side.SAMPLE, description="Concatenate the dense outputs of several sample featurizers.")
class SampleConcatFeaturizer(ConcatFeaturizer):
    """Concatenation of sample featurizers."""


@register(CONCAT, side=Side.DRUG, description="Concatenate the dense outputs of several drug featurizers.")
class DrugConcatFeaturizer(ConcatFeaturizer):
    """Concatenation of drug featurizers."""
