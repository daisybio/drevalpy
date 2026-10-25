"""Name-based column alignment shared by every featurizer that reads a dense view.

A fitted featurizer remembers the feature names of the view it was fitted on. At
transform time the source may hold the same features in another order, or extra
ones; :func:`align_to_fitted` reorders the source matrix to the fitted order and
raises :class:`FeatureMismatchError` when a fitted feature is absent. Reading
columns by position instead would silently mix up genes.
"""

from __future__ import annotations

import numpy as np

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
