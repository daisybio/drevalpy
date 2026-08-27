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
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class GroupMeanCenterer(BaseEstimator, TransformerMixin):
    """
    Subtract the mean response of the group a sample belongs to (usually its drug).

    The class follows the sklearn transformer protocol, but its ``fit``/``transform``/
    ``inverse_transform`` methods accept an additional ``groups`` argument. The
    ``requires_groups`` class attribute tells
    :class:`~drevalpy.datasets.dataset.DrugResponseDataset` to supply the drug ids for
    that argument.

    Groups that were not seen during ``fit`` fall back to the global training mean, which
    makes the transformer safe for LDO (unseen drugs) and cross-study prediction.
    """

    requires_groups = True

    def __init__(self):
        """Initialize the transformer. State is created in :meth:`fit`."""
        self.global_mean_: float = 0.0
        self.group_keys_: np.ndarray = np.array([])
        self.group_means_: np.ndarray = np.array([])

    def fit(self, X, y=None, groups=None) -> "GroupMeanCenterer":
        """
        Estimate the global mean and the per-group means.

        :param X: response values, shape (n,) or (n, 1)
        :param y: ignored, present for sklearn compatibility
        :param groups: group label per sample, e.g. drug ids. If None, only the global
            mean is estimated and the transformer degenerates to plain mean-centering.
        :returns: the fitted transformer
        """
        values = np.asarray(X, dtype=float).reshape(-1)
        self.global_mean_ = float(values.mean()) if values.size else 0.0

        if groups is None or values.size == 0:
            self.group_keys_ = np.array([])
            self.group_means_ = np.array([])
            return self

        labels = np.asarray(groups).reshape(-1)
        keys, inverse = np.unique(labels, return_inverse=True)
        sums = np.bincount(inverse, weights=values)
        counts = np.bincount(inverse)
        self.group_keys_ = keys
        self.group_means_ = sums / counts
        return self

    def _offsets(self, groups, n_samples: int) -> np.ndarray:
        """
        Look up the offset to subtract/add for every sample.

        :param groups: group label per sample, or None
        :param n_samples: number of samples
        :returns: offsets, shape (n_samples,)
        """
        if groups is None or self.group_keys_.size == 0:
            return np.full(n_samples, self.global_mean_)

        labels = np.asarray(groups).reshape(-1)
        positions = np.clip(np.searchsorted(self.group_keys_, labels), 0, self.group_keys_.size - 1)
        known = self.group_keys_[positions] == labels
        return np.where(known, self.group_means_[positions], self.global_mean_)

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
