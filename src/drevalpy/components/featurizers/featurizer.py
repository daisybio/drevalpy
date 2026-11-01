"""The featurizer interface: declarations plus ``fit`` and ``transform``.

A featurizer turns the views of one entity side into a
:class:`~drevalpy.types.data.batch.feature_block.FeaturizerResult`. What it
declares on the class body is read without instantiating it: the views it
reads (``input_views``, empty for a featurizer that needs only entity IDs), the
raw views it computes from when a view is absent (``source_views``) and, through
:meth:`Featurizer.output_blocks` and :meth:`Featurizer.views`, the blocks it emits and the views it reads for a config.

The caller (``models/_feature_preparation.prepare_features``) owns everything around
``fit`` and ``transform``: which entities are usable (:meth:`Featurizer.valid_mask`),
the NaN-threshold warning and padding unusable rows back in. So ``fit`` and
``transform`` only ever see usable entity IDs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar, Self

import numpy as np

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.contracts.hyperparameter_space import TunableComponent
from drevalpy.types.data.batch.feature_block import BlockSpec, FeaturizerResult
from drevalpy.types.data.feature_source import FeatureSource, Side


class Featurizer(TunableComponent, ABC):
    """Fit on one entity side's training entities and transform entities into feature blocks."""

    #: Registry name, set by the registry.
    name: ClassVar[str]
    #: The entity side the featurizer reads, set by the registry.
    side: ClassVar[Side]
    #: Views the featurizer reads; empty when it needs only entity IDs.
    input_views: ClassVar[tuple[str, ...]] = ()
    #: Raw views the featurizer computes its input from when an input view is absent.
    source_views: ClassVar[tuple[str, ...]] = ()
    #: Fraction of unusable entities above which ``prepare_features`` logs a warning.
    nan_threshold: ClassVar[float] = 0.2
    #: Whether ``fit`` needs at least one usable early-stopping pair; ``prepare_features`` raises otherwise.
    requires_early_stopping: ClassVar[bool] = False

    @classmethod
    def output_blocks(cls, config: Mapping[str, Any]) -> tuple[BlockSpec, ...]:
        """Return the blocks the featurizer emits under *config*.

        The default is one numeric block named after the configured ``view``, or after
        the first declared input view; a featurizer that reads no view emits none.
        Featurizers with other outputs override this.

        :param config: Construction kwargs of the featurizer: its options, the defaults of its
            hyperparameter space and, when the config sets one, ``view``.
        :returns: Specs of the emitted blocks.
        """
        view = config.get("view")
        if not isinstance(view, str):
            view = cls.input_views[0] if cls.input_views else None
        if view is None:
            return ()
        return (BlockSpec(view, FeatureFormat.NUMERIC_MATRIX),)

    @classmethod
    def views(cls, config: Mapping[str, Any]) -> tuple[str, ...]:
        """Return the views the featurizer reads under *config*, so the data layer can load them.

        The default is the configured ``view``, else ``input_views``. A featurizer whose views
        depend on a hyperparameter overrides this.

        :param config: Construction kwargs, as for :meth:`output_blocks`.
        :returns: View names, empty when only entity IDs are needed.
        """
        view = config.get("view")
        if isinstance(view, str) and view.strip():
            return (view,)
        return cls.input_views

    def valid_mask(self, source: FeatureSource, ids: np.ndarray) -> np.ndarray:
        """Return which entities have usable input.

        The default reads the first input view and rejects entities whose row is all
        NaN. Every entity is usable when the featurizer reads no view, or when that
        view cannot be read as a numeric matrix.

        :param source: Feature source of the featurizer's side.
        :param ids: Entity IDs to check.
        :returns: Boolean array with one entry per ID, ``True`` for usable entities.
        """
        if not self.input_views:
            return np.ones(len(ids), dtype=bool)
        return all_nan_rows_mask(source, self.input_views[0], ids)

    @abstractmethod
    def fit(
        self,
        source: FeatureSource,
        train_ids: np.ndarray,
        *,
        train_pair_ids: np.ndarray | None = None,
        es_pair_ids: np.ndarray | None = None,
    ) -> Self:
        """Learn whatever ``transform`` needs from the training entities.

        :param source: Feature source of the featurizer's side.
        :param train_ids: Usable training entity IDs, unique, in first-seen order.
        :param train_pair_ids: Entity ID of every usable training pair, repeated per pair.
        :param es_pair_ids: Entity ID of every usable early-stopping pair, repeated per pair.
        :returns: This featurizer, fitted.
        """

    @abstractmethod
    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
        """Compute the feature blocks of *ids*.

        :param source: Feature source of the featurizer's side.
        :param ids: Usable entity IDs, in output order; may be empty.
        :returns: Blocks with one row per ID, in the order of *ids*.
        """


def all_nan_rows_mask(source: FeatureSource, view: str, ids: np.ndarray) -> np.ndarray:
    """Return which entities have a row of *view* that is not all NaN.

    Every entity counts as usable when *view* cannot be read as a numeric matrix.

    :param source: Feature source holding *view*.
    :param view: View to check.
    :param ids: Entity IDs to check.
    :returns: Boolean array with one entry per ID.
    """
    all_valid = np.ones(len(ids), dtype=bool)
    try:
        matrix = source.view(view, ids)
    except (KeyError, TypeError, ValueError):
        return all_valid
    if matrix.ndim != 2 or matrix.dtype.kind not in ("f", "i", "u"):
        return all_valid
    return ~np.all(np.isnan(matrix), axis=1)
