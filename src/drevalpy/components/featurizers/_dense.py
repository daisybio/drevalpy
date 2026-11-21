"""Dense featurizers: reading a view in fitted column order, and the precomputable featurizer base.

:func:`view_matrix` reads a view with its columns in the order recorded at fit,
so a source holding the same features in another order gives the same matrix and
one lacking a fitted feature raises :class:`FeatureMismatchError`; reading columns by
position instead would silently mix up genes. :func:`dense_result` wraps a matrix in the
single-block result.

A :class:`PrecomputableFeaturizer` needs no fitting: it computes one dense row per entity from
its source views (SMILES, gene expression) and its settings, so the output can be computed
once and stored as a view, which :func:`drevalpy.data.precompute.precompute` does at the
default settings. It reads the view when the source holds it and either no settings are
recorded for it (a view the dataset ships, read as it is) or the recorded settings are its
own; otherwise it computes, so a tuned setting never reads a view computed with another.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar, Self

import numpy as np

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers.featurizer import Featurizer, all_nan_rows_mask
from drevalpy.types.data.batch.feature_block import BlockSpec, FeaturizerResult, numeric_feature_block
from drevalpy.types.data.feature_source import FeatureSource

_SHOWN_NAMES = 5


class FeatureMismatchError(ValueError):
    """A source view lacks features the featurizer was fitted on."""


def align_to_fitted(
    source: FeatureSource,
    view: str,
    matrix: np.ndarray,
    fitted_names: tuple[str, ...] | None,
) -> np.ndarray:
    """Return *matrix* with its columns in the order of *fitted_names*.

    The matrix is returned as is when the source names already equal the fitted
    ones, and when either side has no names to compare. Source columns that are not
    in *fitted_names* are dropped.

    :param source: Feature source *matrix* was read from.
    :param view: View *matrix* was read from.
    :param matrix: ``(n_entities, n_source_features)`` matrix of that view.
    :param fitted_names: Feature names recorded at fit, in fitted order.
    :returns: ``(n_entities, len(fitted_names))`` matrix.
    :raises FeatureMismatchError: If a fitted feature is absent from the source view.
    """
    if fitted_names is None:
        return matrix
    names = source.feature_names(view)
    if names is None or tuple(names) == fitted_names:
        return matrix
    position = {name: index for index, name in enumerate(names)}
    missing = [name for name in fitted_names if name not in position]
    if missing:
        shown = ", ".join(repr(name) for name in missing[:_SHOWN_NAMES])
        msg = (
            f"view {view!r} is missing {len(missing)} of the {len(fitted_names)} features "
            f"the featurizer was fitted on, e.g. {shown}"
        )
        raise FeatureMismatchError(msg)
    return matrix[:, [position[name] for name in fitted_names]]


def view_matrix(
    source: FeatureSource,
    view: str,
    ids: np.ndarray,
    fitted_names: tuple[str, ...] | None,
) -> np.ndarray:
    """Return the rows of *view* for *ids*, with columns in fitted order.

    :class:`FeatureMismatchError` is raised when a fitted feature is absent from the view.

    :param source: Feature source holding *view*.
    :param view: View to read.
    :param ids: Entity IDs, in row order.
    :param fitted_names: Feature names recorded at fit; ``None`` keeps the source's columns.
    :returns: Matrix with one row per ID.
    """
    return align_to_fitted(source, view, source.view(view, ids), fitted_names)


def dense_result(
    ids: np.ndarray,
    block: str,
    matrix: np.ndarray,
    names: tuple[str, ...] | None,
) -> FeaturizerResult:
    """Wrap *matrix* as the single numeric block of a result.

    :param ids: Entity IDs, one per row of *matrix*.
    :param block: Block name.
    :param matrix: Feature matrix; cast to float32.
    :param names: Column names of *matrix*, or ``None``.
    :returns: Result holding *matrix* as block *block*, projected as the matrix.
    """
    values = numeric_feature_block(matrix.astype(np.float32), feature_names=names)
    return FeaturizerResult(ids, {block: values}, (block,))


class PrecomputableFeaturizer(Featurizer):
    """A featurizer that reads its precomputed view when it matches its settings, and computes it otherwise."""

    #: Name of the emitted block; ``None`` names it after the view.
    block: ClassVar[str | None] = None

    def __init__(self, view: str, settings: Mapping[str, Any]) -> None:
        """Bind the view and the settings the output is computed with.

        :param view: View holding the precomputed output.
        :param settings: Hyperparameter values that decide the output, compared with the view's recorded ones.
        """
        self._view = view
        self._settings = dict(settings)
        self._view_names: tuple[str, ...] | None = None

    @property
    def view(self) -> str:
        """The view holding the precomputed output."""
        return self._view

    @property
    def settings(self) -> dict[str, Any]:
        """The hyperparameter values that decide the output, as recorded with a precomputed view."""
        return dict(self._settings)

    @classmethod
    def output_blocks(cls, config: Mapping[str, Any]) -> tuple[BlockSpec, ...]:
        """Return the single numeric block: :attr:`block`, or the view's name without one.

        :param config: Construction kwargs of the featurizer.
        :returns: One numeric block spec.
        """
        if cls.block is None:
            return super().output_blocks(config)
        return (BlockSpec(cls.block, FeatureFormat.NUMERIC_MATRIX),)

    def reads_view(self, source: FeatureSource) -> bool:
        """Return whether *source* holds the view computed with this featurizer's settings, or unrecorded.

        :param source: Feature source of the featurizer's side.
        :returns: ``True`` to read the view, ``False`` to compute.
        """
        if not source.has_view(self._view):
            return False
        recorded = source.view_settings(self._view)
        return recorded is None or recorded == self._settings

    def valid_mask(self, source: FeatureSource, ids: np.ndarray) -> np.ndarray:
        """Reject entities whose view row is all NaN when the view is read; all are usable when computed.

        :param source: Feature source of the featurizer's side.
        :param ids: Entity IDs to check.
        :returns: Boolean array with one entry per ID.
        """
        if self.reads_view(source):
            return all_nan_rows_mask(source, self._view, ids)
        return np.ones(len(ids), dtype=bool)

    def fit(
        self,
        source: FeatureSource,
        train_ids: np.ndarray,
        *,
        train_pair_ids: np.ndarray | None = None,
        es_pair_ids: np.ndarray | None = None,
    ) -> Self:
        """Record the view's feature names when the view is read; nothing is learned.

        :param source: Feature source of the featurizer's side.
        :param train_ids: Unused.
        :param train_pair_ids: Unused.
        :param es_pair_ids: Unused.
        :returns: This featurizer.
        """
        _ = train_ids, train_pair_ids, es_pair_ids
        self._view_names = source.feature_names(self._view) if self.reads_view(source) else None
        return self

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
        """Read the rows of *ids* from the view, or compute them.

        :param source: Feature source of the featurizer's side.
        :param ids: Entity IDs, in output order.
        :returns: Result with the single block.
        """
        block = self.block or self._view
        if self.reads_view(source):
            names = self._view_names or source.feature_names(self._view)
            return dense_result(ids, block, view_matrix(source, self._view, ids, names), names)
        return dense_result(ids, block, self.compute(source, ids), None)

    @abstractmethod
    def compute(self, source: FeatureSource, ids: np.ndarray) -> np.ndarray:
        """Compute the output from the source views.

        :param source: Feature source of the featurizer's side.
        :param ids: Entity IDs, in row order.
        :returns: Matrix with one row per ID.
        """
