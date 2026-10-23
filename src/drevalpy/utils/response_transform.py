"""Response transformations selectable by name, and the group-aware ``GroupMeanCenterer``.

The sklearn scalers are global monotone rescalings of the response. ``GroupMeanCenterer`` is
different: it subtracts a *conditional* mean (by default the per-drug mean of the training
scope) so a model learns the residual sample x drug structure, and adds the mean back to the
predictions. It is a plain class with ``get_params``/``set_params`` rather than a
``sklearn.base.BaseEstimator`` subclass, because a base class would have to import sklearn
when ``import drevalpy`` runs (see ``tests/test_import_cost_policy.py``); ``sklearn.base.clone``
only needs the two parameter methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from drevalpy.errors import InvalidInputError, UnknownNameError

if TYPE_CHECKING:
    from sklearn.base import TransformerMixin

#: ASCII unit separator, used to join the group levels into one lookup key. Neither drug ids nor
#: tissue names contain it, so the joined keys cannot collide.
_KEY_SEPARATOR = "\x1f"

#: Group levels a ``GroupMeanCenterer`` can use, in the order the pipeline reads them from the data.
GROUP_LEVELS = ("drug", "tissue")


def _composite_keys(labels: np.ndarray, n_levels: int) -> np.ndarray:
    """Join the first *n_levels* columns of *labels* into one key per value.

    :param labels: Group labels, shape ``(n, n_group_levels)``.
    :param n_levels: Number of leading columns to use.
    :returns: Joined keys, shape ``(n,)``.
    """
    keys = labels[:, 0].astype(str)
    for index in range(1, n_levels):
        keys = np.char.add(np.char.add(keys, _KEY_SEPARATOR), labels[:, index].astype(str))
    return keys


def _as_label_matrix(groups: Any) -> np.ndarray:
    """Bring group labels into the shape ``(n, n_group_levels)``.

    :param groups: Labels of shape ``(n,)`` for one level or ``(n, n_group_levels)``.
    :returns: Two-dimensional labels.
    """
    labels = np.asarray(groups)
    return labels.reshape(-1, 1) if labels.ndim == 1 else labels


class GroupMeanCenterer:
    """Subtract the mean response of the group a pair belongs to, usually its drug.

    It follows the sklearn transformer protocol, but ``fit``, ``transform`` and
    ``inverse_transform`` take an additional ``groups`` argument. The class attribute
    ``requires_groups`` tells the model pipeline to build the label columns named by ``levels``
    and pass them as ``groups``.

    With several levels the means are nested from the most specific combination down to the
    global mean. For ``levels=("drug", "tissue")`` a pair is centred on the mean of its
    (drug, tissue) combination if that combination occurred in ``fit``, otherwise on the mean of
    its drug, otherwise on the global training mean. That keeps the transformer safe for unseen
    drugs, unseen tissues (it degenerates to per-drug centring) and cross-study prediction.
    """

    requires_groups = True

    def __init__(self, levels: tuple[str, ...] = ("drug",)) -> None:
        """Store the group levels; state is created in :meth:`fit`.

        :param levels: Group levels from coarse to fine; each is one of :data:`GROUP_LEVELS`. The nested
            means are estimated for every prefix of this tuple.
        """
        self.levels = levels

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the constructor parameters, as ``sklearn.base.clone`` expects.

        :param deep: Ignored; there are no nested estimators.
        :returns: Mapping with the key ``levels``.
        """
        return {"levels": self.levels}

    def set_params(self, **params: Any) -> GroupMeanCenterer:
        r"""Set constructor parameters, as ``sklearn.base.clone`` expects.

        :param \**params: New parameter values; only ``levels`` exists.
        :returns: This transformer.
        :raises InvalidInputError: If a name is not a parameter.
        """
        for name, value in params.items():
            if name != "levels":
                raise InvalidInputError(f"Invalid parameter {name!r} for GroupMeanCenterer; the only one is 'levels'")
            self.levels = value
        return self

    def fit(self, X: Any, y: Any = None, groups: Any = None) -> GroupMeanCenterer:  # noqa: N803
        """Estimate the global mean and the mean of every group and parent group.

        :param X: Response values, shape ``(n,)`` or ``(n, 1)``.
        :param y: Ignored, present for sklearn compatibility.
        :param groups: Group label per pair, shape ``(n,)`` or ``(n, n_levels)``. Without it only the
            global mean is estimated and the transformer is plain mean centring.
        :returns: The fitted transformer.
        :raises InvalidInputError: If a level is unknown, or the labels do not have one column per level.
        """
        unknown = [level for level in self.levels if level not in GROUP_LEVELS]
        if unknown:
            raise InvalidInputError(f"Unknown group level {unknown[0]!r}; choose from {list(GROUP_LEVELS)}")
        values = np.asarray(X, dtype=float).reshape(-1)
        self.global_mean_ = float(values.mean()) if values.size else 0.0
        self.level_keys_: list[np.ndarray] = []
        self.level_means_: list[np.ndarray] = []
        self.n_fields_ = 0
        if groups is None or values.size == 0:
            return self
        labels = _as_label_matrix(groups)
        if labels.shape[1] != len(self.levels):
            raise InvalidInputError(f"GroupMeanCenterer{self.levels} was given {labels.shape[1]} group column(s)")
        self.n_fields_ = labels.shape[1]
        for n_levels in range(self.n_fields_, 0, -1):
            keys, inverse = np.unique(_composite_keys(labels, n_levels), return_inverse=True)
            self.level_keys_.append(keys)
            self.level_means_.append(np.bincount(inverse, weights=values) / np.bincount(inverse))
        return self

    def _offsets(self, groups: Any, n_values: int) -> np.ndarray:
        """Look up the offset to subtract or add for every value.

        :param groups: Group label per value, or None for the global mean.
        :param n_values: Number of values.
        :returns: Offsets, shape ``(n_values,)``.
        :raises ValueError: If the transformer is unfitted, or *groups* has another number of columns than ``fit`` saw.
        """
        if not hasattr(self, "global_mean_"):
            raise ValueError("This GroupMeanCenterer is not fitted; call fit first")
        offsets = np.full(n_values, self.global_mean_)
        if groups is None or not self.level_keys_:
            return offsets
        labels = _as_label_matrix(groups)
        if labels.shape[1] != self.n_fields_:
            raise ValueError(
                f"The transformer was fitted on {self.n_fields_} group level(s) but was given {labels.shape[1]}."
            )
        pending = np.ones(n_values, dtype=bool)
        for level, (keys, means) in enumerate(zip(self.level_keys_, self.level_means_, strict=True)):
            if not pending.any():
                break
            value_keys = _composite_keys(labels, self.n_fields_ - level)
            positions = np.clip(np.searchsorted(keys, value_keys), 0, keys.size - 1)
            found = pending & (keys[positions] == value_keys)
            offsets[found] = means[positions[found]]
            pending &= ~found
        return offsets

    def transform(self, X: Any, groups: Any = None) -> np.ndarray:  # noqa: N803
        """Subtract the group mean.

        :param X: Response or prediction values.
        :param groups: Group label per value.
        :returns: Residuals, same shape as *X*.
        """
        values = np.asarray(X, dtype=float)
        flat = values.reshape(-1)
        return (flat - self._offsets(groups, flat.size)).reshape(values.shape)

    def inverse_transform(self, X: Any, groups: Any = None) -> np.ndarray:  # noqa: N803
        """Add the group mean back, returning values on the original response scale.

        :param X: Residual values.
        :param groups: Group label per value.
        :returns: Values on the original scale, same shape as *X*.
        """
        values = np.asarray(X, dtype=float)
        flat = values.reshape(-1)
        return (flat + self._offsets(groups, flat.size)).reshape(values.shape)


def get_response_transformation(
    response_transformation: str | None,
) -> TransformerMixin | GroupMeanCenterer | None:
    """Return the response transformer for a pipeline option.

    ``"drug_mean"`` subtracts the per-drug mean of the training scope. ``"drug_tissue_mean"`` does the
    same per drug and tissue, falling back to the per-drug mean for a (drug, tissue) combination the
    training scope did not contain. Both add the mean back to the predictions.

    :param response_transformation: ``None`` or ``"none"`` (in any letter case) for no transformation, or one of
        ``"standard"``, ``"minmax"``, ``"robust"``, ``"drug_mean"`` and ``"drug_tissue_mean"``.
    :returns: A new, unfitted transformer, or ``None`` for no transformation.
    :raises UnknownNameError: If *response_transformation* is not recognized.
    """
    if response_transformation is None or response_transformation.lower() == "none":
        return None
    if response_transformation == "drug_mean":
        return GroupMeanCenterer(("drug",))
    if response_transformation == "drug_tissue_mean":
        return GroupMeanCenterer(("drug", "tissue"))
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    scalers = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
    if response_transformation in scalers:
        return scalers[response_transformation]()
    raise UnknownNameError(
        f"Unknown response transformation {response_transformation}. Choose from 'None', 'standard', 'minmax', "
        "'robust', 'drug_mean', 'drug_tissue_mean'"
    )
