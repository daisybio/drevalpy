"""The featurizer interface: declarations plus ``fit`` and ``transform``.

A featurizer turns the views of one entity side into a
:class:`~drevalpy.types.data.batch.feature_block.FeaturizerResult`. What it
declares on the class body is read without instantiating it: the sides it
supports, the views it reads (``input_views``, empty for a featurizer that needs
only entity IDs), the raw views it computes from when a view is absent
(``source_views``), the key its pre-computed outputs are stored under, and, through
:meth:`Featurizer.output_blocks` and :meth:`Featurizer.views`, the blocks it emits and
the views it reads for a config.

The caller (``models/_feature_preparation.prepare_features``) owns everything around
``fit`` and ``transform``: which entities are usable (:meth:`Featurizer.valid_mask`),
the NaN-threshold warning, the stored-output lookup through
:meth:`~drevalpy.types.data.feature_source.FeatureSource.stored` with
:meth:`Featurizer.stored_params`, and padding unusable rows back in. So ``fit`` and
``transform`` only ever see usable entity IDs, and a featurizer whose output is stored
is neither fitted nor transformed: :meth:`Featurizer.from_stored` wraps the stored matrix.
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
    #: Entity sides the featurizer can be used on.
    sides: ClassVar[frozenset[Side]] = frozenset({"sample", "drug"})
    #: Views the featurizer reads; empty when it needs only entity IDs.
    input_views: ClassVar[tuple[str, ...]] = ()
    #: Raw views the featurizer computes its input from when an input view is absent.
    source_views: ClassVar[tuple[str, ...]] = ()
    #: Key pre-computed outputs are stored under. Registration sets it to the registry name
    #: when neither the class nor a base declares one.
    storage_key: ClassVar[str | None] = None
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

    def stored_params(self) -> Mapping[str, Any] | None:
        """Return the hyperparameters a stored output must have been computed with.

        :returns: Hyperparameters to match (``{}`` matches the parameter-free variant), or
            ``None`` (the default) when stored outputs are never looked up.
        """
        return None

    def from_stored(self, source: FeatureSource, ids: np.ndarray, matrix: np.ndarray) -> FeaturizerResult:
        """Wrap a stored output matrix as the result of :meth:`transform`.

        Called instead of ``fit`` and ``transform`` when :meth:`stored_params` is not ``None``
        and the source stores a matching variant. A featurizer that returns stored
        parameters overrides this, usually with
        :func:`~drevalpy.components.featurizers._dense.dense_result`.

        :param source: Feature source of the featurizer's side.
        :param ids: Usable entity IDs, in output order.
        :param matrix: Stored matrix with one row per ID.
        :returns: Blocks with one row per ID, in the order of *ids*.
        :raises NotImplementedError: Always, unless overridden.
        """
        _ = source, ids, matrix
        msg = f"{type(self).__name__} returns stored_params() but does not implement from_stored()"
        raise NotImplementedError(msg)

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
        matrix = source.get_view_matrix(view, ids)
    except (KeyError, TypeError, ValueError):
        return all_valid
    if matrix.ndim != 2 or matrix.dtype.kind not in ("f", "i", "u"):
        return all_valid
    return ~np.all(np.isnan(matrix), axis=1)


def usable(featurizer: Featurizer, source: FeatureSource, ids: np.ndarray) -> np.ndarray:
    """Return :meth:`Featurizer.valid_mask` of *featurizer* for *ids*.

    :param featurizer: Featurizer whose missing-input policy applies.
    :param source: Feature source of the featurizer's side.
    :param ids: Entity IDs to check; may repeat.
    :returns: Boolean array with one entry per ID.
    """
    return featurizer.valid_mask(source, ids)


def stored_output(featurizer: Featurizer, source: FeatureSource, ids: np.ndarray) -> np.ndarray | None:
    """Return the stored output of *featurizer* for *ids*, if the source stores a matching variant.

    Storage is looked up only when the featurizer has a ``storage_key`` and its
    :meth:`Featurizer.stored_params` is not ``None``. The decision depends on the source and
    the parameters only, so fit and transform on the same source make the same one.

    :param featurizer: Featurizer whose stored output is wanted.
    :param source: Feature source of the featurizer's side.
    :param ids: Entity IDs to align the stored rows to.
    :returns: Stored matrix with one row per ID, or ``None``.
    """
    params = featurizer.stored_params()
    if featurizer.storage_key is None or params is None:
        return None
    return source.stored(featurizer.storage_key, params, ids)


def fit_unless_stored(
    featurizer: Featurizer,
    source: FeatureSource,
    train_ids: np.ndarray,
    *,
    train_pair_ids: np.ndarray | None,
    es_pair_ids: np.ndarray | None,
) -> None:
    """Fit *featurizer* on usable training entities, unless its output is stored.

    :param featurizer: Featurizer to fit.
    :param source: Feature source of the featurizer's side.
    :param train_ids: Usable training entity IDs, unique, in first-seen order.
    :param train_pair_ids: Entity ID of every usable training pair.
    :param es_pair_ids: Entity ID of every usable early-stopping pair.
    """
    if stored_output(featurizer, source, train_ids) is None:
        featurizer.fit(source, train_ids, train_pair_ids=train_pair_ids, es_pair_ids=es_pair_ids)


def transform_or_stored(featurizer: Featurizer, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
    """Return the blocks of usable *ids*: the stored output when there is one, else :meth:`Featurizer.transform`.

    :param featurizer: Fitted featurizer.
    :param source: Feature source of the featurizer's side.
    :param ids: Usable entity IDs, in output order; may be empty.
    :returns: Blocks with one row per ID.
    """
    stored = stored_output(featurizer, source, ids)
    if stored is not None:
        return featurizer.from_stored(source, ids, stored)
    return featurizer.transform(source, ids)
