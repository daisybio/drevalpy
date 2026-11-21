"""mRMR feature selection and the neighbour-capped kNN regressor of ``singleDrugKnn``.

The classes derive from scikit-learn bases, so this module imports ``sklearn`` at module
scope. Nothing on ``import drevalpy`` imports it: ``sklearn_models.py`` loads it inside
``_make_estimator``. See ``tests/test_import_cost_policy.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsRegressor


class MRMRSelector(TransformerMixin, BaseEstimator):
    """Supervised feature selection by greedy mRMR (minimum redundancy, maximum relevance).

    Relevance is the F-statistic of the univariate linear fit of the response on a column;
    redundancy is the mean absolute Pearson correlation to the columns chosen so far (floored
    at ``floor``). The next column maximizes relevance divided by redundancy. Columns without
    variance have no relevance and are never chosen. If no column has any relevance, all
    columns are kept.

    Used as a step of an sklearn pipeline, the selection is fitted on the training rows only.
    """

    def __init__(self, k: int = 20, floor: float = 0.001) -> None:
        """Initialize the selector.

        :param k: Number of columns to select (capped at the number of columns).
        :param floor: Lower bound of the redundancy, avoids division by (almost) zero.
        """
        self.k = k
        self.floor = floor

    def _relevance(self, centered: np.ndarray, norm: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return the F-statistic of the univariate linear fit of *y* on each column.

        :param centered: Column-centered feature matrix.
        :param norm: Euclidean norm of each centered column.
        :param y: Response vector.
        :returns: One non-negative value per column; zero for columns without variance.
        """
        n, p = centered.shape
        varies = norm > 0
        y_centered = y - y.mean()
        y_norm = np.sqrt((y_centered**2).sum())
        r = np.zeros(p)
        if y_norm > 0:
            np.divide(centered.T @ y_centered, norm * y_norm, out=r, where=varies)
        r2 = np.clip(r**2, 0.0, 1.0 - 1e-15)
        return np.where(varies, r2 / (1.0 - r2) * (n - 2), 0.0)

    def fit(self, X: Any, y: Any) -> MRMRSelector:  # noqa: N803
        """Select the columns on the given rows.

        :param X: Feature matrix (rows x columns), finite.
        :param y: Response vector of the same rows, finite.
        :returns: The fitted selector; ``selected_`` holds the column indices in selection order.
        :raises ValueError: If ``X`` or ``y`` contains non-finite values or ``k`` is smaller than one.
        """
        x = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if self.k < 1:
            msg = f"k must be at least 1, not {self.k}."
            raise ValueError(msg)
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            msg = "mRMR needs finite values in X and y; remove missing responses first."
            raise ValueError(msg)
        p = x.shape[1]
        centered = x - x.mean(axis=0)
        norm = np.sqrt((centered**2).sum(axis=0))
        relevance = self._relevance(centered, norm, y)

        unit = np.zeros_like(centered)
        np.divide(centered, norm, out=unit, where=norm > 0)  # unit columns: correlation = dot product
        candidates = [j for j in range(p) if relevance[j] > 0]
        chosen: list[int] = []
        redundancy_sum = np.zeros(p)
        for _ in range(min(self.k, len(candidates))):
            cand = np.array(candidates, dtype=int)
            if chosen:
                denominator = redundancy_sum[cand] / len(chosen)
                denominator = np.where(denominator == 1.0, np.inf, denominator)
                score = relevance[cand] / denominator
            else:
                score = relevance[cand]
            best = int(cand[int(np.argmax(score))])
            chosen.append(best)
            candidates.remove(best)
            redundancy_sum += np.maximum(np.abs(unit.T @ unit[:, best]), self.floor)
        self.selected_ = np.array(chosen if chosen else range(p), dtype=int)
        return self

    def transform(self, X: Any) -> np.ndarray:  # noqa: N803
        """Keep the selected columns.

        :param X: Feature matrix with the same columns as in ``fit``.
        :returns: Matrix with the selected columns in selection order.
        """
        return np.asarray(X)[:, self.selected_]


class NeighborCappedKNN(KNeighborsRegressor):
    """``KNeighborsRegressor`` that asks for no more neighbours than it has training rows.

    A drug with fewer training pairs than ``n_neighbors`` would otherwise fail at predict time.
    """

    def fit(self, X: Any, y: Any) -> NeighborCappedKNN:  # noqa: N803
        """Fit on the given rows after lowering ``n_neighbors`` to their number.

        :param X: Training feature matrix.
        :param y: Training responses.
        :returns: The fitted regressor.
        """
        self.n_neighbors = min(self.n_neighbors, max(len(X), 1))
        super().fit(X, y)
        return self
