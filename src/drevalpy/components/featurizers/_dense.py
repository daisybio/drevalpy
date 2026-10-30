"""Plain functions for featurizers that read a dense view and emit one numeric block.

:func:`view_matrix` reads a view with its columns in the order recorded at fit,
so a source holding the same features in another order gives the same matrix and
one lacking a fitted feature raises :class:`FeatureMismatchError`; reading columns by
position instead would silently mix up genes. :func:`dense_result` wraps a matrix in the
single-block result.
"""

from __future__ import annotations

import numpy as np

from drevalpy.types.data.batch.feature_block import FeaturizerResult, numeric_feature_block
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
    names = source.get_feature_names(view)
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
    return align_to_fitted(source, view, source.get_view_matrix(view, ids), fitted_names)


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
