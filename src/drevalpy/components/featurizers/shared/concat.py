"""Concatenating featurizer, registered as ``concatFeaturizers`` on both sides.

Each child is fitted and transformed on the entities that are usable for every child, and the
children's named numeric blocks are collected into one :class:`FeaturizerOutput`. Block
predictors read each child's block by name; matrix predictors get the blocks horizontally
concatenated via :attr:`FeaturizerOutput.matrix`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

import numpy as np

from drevalpy.components.featurizers.featurizer import CONCAT, Featurizer
from drevalpy.registry.featurizer import register
from drevalpy.types.data.feature_block import FeatureBlock, FeaturizerOutput
from drevalpy.types.data.feature_format import BlockSpec, FeatureFormat
from drevalpy.types.data.feature_source import FeatureSource, Side


class ConcatFeaturizer(Featurizer):
    """Fit child featurizers independently and collect their numeric blocks for block or matrix access."""

    def __init__(self, *, featurizers: list[Featurizer] | None = None) -> None:
        """Bind the child featurizers.

        A model builds the children from its config (``models/drp_model.py``), so this class never
        reads a config itself.

        :param featurizers: Built child featurizers; must be non-empty.
        :raises ValueError: If *featurizers* is empty.
        :raises TypeError: If a child is not a built featurizer.
        """
        if not featurizers:
            msg = "featurizers must be a non-empty list"
            raise ValueError(msg)
        unbuilt = [type(child).__name__ for child in featurizers if not isinstance(child, Featurizer)]
        if unbuilt:
            raise TypeError(f"concat featurizer children must be built featurizers, got {unbuilt}")
        self._children = list(featurizers)

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

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerOutput:
        """Transform with every child and combine their named numeric blocks into one result.

        Named blocks stay addressable individually in ``result.blocks``, and
        ``result.matrix`` concatenates all entity-aligned numeric blocks horizontally in child order.

        :param source: Feature source of the featurizer's side.
        :param ids: Entity IDs usable for every child, in output order.
        :returns: Combined feature blocks of all children.
        :raises ValueError: If a child emits a non-numeric block or two children emit the same block name.
        """
        merged: dict[str, FeatureBlock] = {}
        for child in self._children:
            blocks = child.transform(source, ids).blocks
            wrong = [name for name, block in blocks.items() if block.format != FeatureFormat.NUMERIC_MATRIX]
            if wrong:
                msg = f"concat featurizer child {type(child).__name__} emits non-numeric blocks {wrong}"
                raise ValueError(msg)
            for name, block in blocks.items():
                if name in merged:
                    msg = f"Duplicate featurizer block name {name!r}"
                    raise ValueError(msg)
                merged[name] = block
        return FeaturizerOutput(ids, merged)


@register(CONCAT, side=Side.SAMPLE, description="Concatenate the dense outputs of several sample featurizers.")
class SampleConcatFeaturizer(ConcatFeaturizer):
    """Concatenation of sample featurizers."""


@register(CONCAT, side=Side.DRUG, description="Concatenate the dense outputs of several drug featurizers.")
class DrugConcatFeaturizer(ConcatFeaturizer):
    """Concatenation of drug featurizers."""
