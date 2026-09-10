"""
Group-aware response transformations (conditional-mean residualization).

The transformations shipped with drevalpy ("standard", "minmax", "robust") are *global*
monotone rescalings of the response. Because the evaluation metrics of interest are
rank-based (Spearman) and additionally mean-centered per drug and per cell line
("normalized" metrics), such global rescalings cannot change the reported scores.

``GroupMeanCenterer`` is different: it subtracts a *conditional* mean (by default the
per-drug mean, estimated on the training fold only) from the training target, so the
model spends its capacity on the residual drug x cell line structure instead of on the
drug main effect. At prediction time the group mean is added back, so predictions are
returned on the original response scale.

The group can be built from more than one field, e.g. drug and tissue. The fields are
ordered from coarse to fine, and the estimated means are nested: a sample whose
(drug, tissue) combination was not seen during ``fit`` falls back to the mean of its drug,
and only a sample whose drug is unknown as well falls back to the global training mean.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

#: ASCII unit separator, used to join the group fields into one lookup key. Neither drug
#: ids nor tissue names contain it, so the joined keys cannot collide.
_KEY_SEPARATOR = "\x1f"


def _composite_keys(labels: np.ndarray, n_fields: int) -> np.ndarray:
    """
    Join the first ``n_fields`` columns of ``labels`` into one key per sample.

    :param labels: group labels, shape (n_samples, n_group_fields)
    :param n_fields: number of leading columns to use
    :returns: joined keys, shape (n_samples,)
    """
    keys = labels[:, 0].astype(str)
    for index in range(1, n_fields):
        keys = np.char.add(np.char.add(keys, _KEY_SEPARATOR), labels[:, index].astype(str))
    return keys


class GroupMeanCenterer(BaseEstimator, TransformerMixin):
    """
    Subtract the mean response of the group a sample belongs to (usually its drug).

    The class follows the sklearn transformer protocol, but its ``fit``/``transform``/
    ``inverse_transform`` methods accept an additional ``groups`` argument. The
    ``requires_groups`` class attribute tells
    :class:`~drevalpy.datasets.dataset.DrugResponseDataset` to supply the columns named in
    ``group_fields`` for that argument.

    With more than one group field the means are nested, from the most specific
    combination down to the global mean: with ``group_fields=("drug_ids", "tissue")`` a
    sample is centered on the mean of its (drug, tissue) combination if that combination
    occurred during ``fit``, otherwise on the mean of its drug, otherwise on the global
    training mean. This keeps the transformer safe for LDO (unseen drugs), LTO (unseen
    tissues, where it degenerates to per-drug centering) and cross-study prediction.
    """

    requires_groups = True

    def __init__(self, group_fields: tuple[str, ...] = ("drug_ids",)):
        """
        Initialize the transformer. State is created in :meth:`fit`.

        :param group_fields: attributes of
            :class:`~drevalpy.datasets.dataset.DrugResponseDataset` that form the group
            key, ordered from coarse to fine. The nested means are estimated for every
            prefix of this tuple.
        """
        self.group_fields = group_fields
        self.global_mean_: float = 0.0
        #: keys per nesting level, most specific level first
        self.level_keys_: list[np.ndarray] = []
        #: mean per key, aligned with :attr:`level_keys_`
        self.level_means_: list[np.ndarray] = []
        self.n_fields_: int = 0

    def fit(self, X, y=None, groups=None) -> "GroupMeanCenterer":
        """
        Estimate the global mean and the mean of every group and parent group.

        :param X: response values, shape (n,) or (n, 1)
        :param y: ignored, present for sklearn compatibility
        :param groups: group label per sample, shape (n,) or (n, n_group_fields). If None,
            only the global mean is estimated and the transformer degenerates to plain
            mean-centering.
        :returns: the fitted transformer
        """
        values = np.asarray(X, dtype=float).reshape(-1)
        self.global_mean_ = float(values.mean()) if values.size else 0.0
        self.level_keys_ = []
        self.level_means_ = []
        self.n_fields_ = 0

        if groups is None or values.size == 0:
            return self

        labels = _as_label_matrix(groups)
        self.n_fields_ = labels.shape[1]
        for n_fields in range(self.n_fields_, 0, -1):
            keys, inverse = np.unique(_composite_keys(labels, n_fields), return_inverse=True)
            sums = np.bincount(inverse, weights=values)
            counts = np.bincount(inverse)
            self.level_keys_.append(keys)
            self.level_means_.append(sums / counts)
        return self

    def _offsets(self, groups, n_samples: int) -> np.ndarray:
        """
        Look up the offset to subtract/add for every sample.

        :param groups: group label per sample, or None
        :param n_samples: number of samples
        :returns: offsets, shape (n_samples,)
        :raises ValueError: if the number of group fields differs from the one used in fit
        """
        offsets = np.full(n_samples, self.global_mean_)
        if groups is None or not self.level_keys_:
            return offsets

        labels = _as_label_matrix(groups)
        if labels.shape[1] != self.n_fields_:
            raise ValueError(
                f"The transformer was fitted on {self.n_fields_} group field(s) but was given {labels.shape[1]}."
            )

        # Most specific level first; every sample keeps the first offset it finds.
        pending = np.ones(n_samples, dtype=bool)
        for level, (keys, means) in enumerate(zip(self.level_keys_, self.level_means_)):
            if not pending.any():
                break
            sample_keys = _composite_keys(labels, self.n_fields_ - level)
            positions = np.clip(np.searchsorted(keys, sample_keys), 0, keys.size - 1)
            found = pending & (keys[positions] == sample_keys)
            offsets[found] = means[positions[found]]
            pending &= ~found
        return offsets

    def transform(self, X, groups=None) -> np.ndarray:
        """
        Subtract the group mean.

        :param X: response or prediction values
        :param groups: group label per sample, e.g. drug ids
        :returns: residuals, same shape as X
        """
        values = np.asarray(X, dtype=float)
        flat = values.reshape(-1)
        return (flat - self._offsets(groups, flat.size)).reshape(values.shape)

    def inverse_transform(self, X, groups=None) -> np.ndarray:
        """
        Add the group mean back, returning values on the original response scale.

        :param X: residual values
        :param groups: group label per sample, e.g. drug ids
        :returns: values on the original scale, same shape as X
        """
        values = np.asarray(X, dtype=float)
        flat = values.reshape(-1)
        return (flat + self._offsets(groups, flat.size)).reshape(values.shape)


def _as_label_matrix(groups) -> np.ndarray:
    """
    Bring the group labels into the shape (n_samples, n_group_fields).

    :param groups: group labels, shape (n,) for a single field or (n, n_group_fields)
    :returns: two-dimensional group labels
    """
    labels = np.asarray(groups)
    if labels.ndim == 1:
        return labels.reshape(-1, 1)
    return labels
