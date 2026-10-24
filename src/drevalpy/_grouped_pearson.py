"""Streaming per-group Pearson correlation, the primitive under :mod:`drevalpy._grouped_evaluation`.

Everything works on ``np.bincount`` sums, so computing one correlation per drug or sample
is a handful of vectorised passes and folds can be folded in one at a time without ever
coexisting in memory. Which groups are undefined is decided here: fewer than
:data:`MIN_GROUP_SIZE` observations, or a variable that
:func:`drevalpy.evaluation.ranges_are_constant` calls constant. See
:mod:`drevalpy._grouped_evaluation` for how that rule is applied to every metric.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from drevalpy.evaluation import ranges_are_constant

#: Correlation is undefined below this many observations in a group.
MIN_GROUP_SIZE: Final[int] = 2


class PearsonSums:
    """Streaming sufficient statistics for a grouped Pearson correlation.

    Holds the six ``bincount`` accumulators (count, sum x, sum y, sum x^2,
    sum y^2, sum xy) and the per-group range of each variable for a fixed group axis,
    so folds can be folded in one at a time and never coexist in memory. The ranges
    decide whether a group is constant: a variance recovered from the sums is not exactly
    zero for a constant series, and would turn a constant prediction into a spurious score.
    """

    __slots__ = ("_count", "_max_x", "_max_y", "_min_x", "_min_y", "_sx", "_sxx", "_sxy", "_sy", "_syy", "n_groups")

    def __init__(self, n_groups: int) -> None:
        """Create empty accumulators for a fixed number of groups.

        :param n_groups: Size of the group axis; codes passed to :meth:`add` index into it.
        """
        self.n_groups = n_groups
        self._count = np.zeros(n_groups, dtype=np.int64)
        self._sx = np.zeros(n_groups, dtype=np.float64)
        self._sy = np.zeros(n_groups, dtype=np.float64)
        self._sxx = np.zeros(n_groups, dtype=np.float64)
        self._syy = np.zeros(n_groups, dtype=np.float64)
        self._sxy = np.zeros(n_groups, dtype=np.float64)
        self._min_x = np.full(n_groups, np.inf, dtype=np.float64)
        self._max_x = np.full(n_groups, -np.inf, dtype=np.float64)
        self._min_y = np.full(n_groups, np.inf, dtype=np.float64)
        self._max_y = np.full(n_groups, -np.inf, dtype=np.float64)

    def add(self, codes: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
        """Accumulate one batch of observations.

        Rows with a negative code (no matching group) or a non-finite value on either
        axis are dropped.

        :param codes: Group index per observation.
        :param x: First variable, aligned with ``codes``.
        :param y: Second variable, aligned with ``codes``.
        """
        codes = np.asarray(codes)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        keep = (codes >= 0) & np.isfinite(x) & np.isfinite(y)
        if not keep.all():
            codes, x, y = codes[keep], x[keep], y[keep]
        if codes.size == 0:
            return
        n = self.n_groups
        self._count += np.bincount(codes, minlength=n)
        self._sx += np.bincount(codes, weights=x, minlength=n)
        self._sy += np.bincount(codes, weights=y, minlength=n)
        self._sxx += np.bincount(codes, weights=x * x, minlength=n)
        self._syy += np.bincount(codes, weights=y * y, minlength=n)
        self._sxy += np.bincount(codes, weights=x * y, minlength=n)
        np.minimum.at(self._min_x, codes, x)
        np.maximum.at(self._max_x, codes, x)
        np.minimum.at(self._min_y, codes, y)
        np.maximum.at(self._max_y, codes, y)

    @property
    def counts(self) -> np.ndarray:
        """Number of accumulated observations per group."""
        return self._count

    def y_varies(self) -> np.ndarray:
        """Whether ``y`` is non-constant in each group, by :func:`drevalpy.evaluation.ranges_are_constant`.

        :returns: Boolean array of length ``n_groups``; ``False`` for empty groups.
        """
        return (self._count > 0) & ~ranges_are_constant(self._max_y, self._min_y)

    def varies(self) -> np.ndarray:
        """Whether both variables are non-constant in each group, by :func:`drevalpy.evaluation.ranges_are_constant`.

        :returns: Boolean array of length ``n_groups``; ``False`` for empty groups.
        """
        return (self._count > 0) & ~ranges_are_constant(self._max_x, self._min_x) & self.y_varies()

    def sum_squares_of_y(self) -> np.ndarray:
        """Total sum of squares of ``y`` about its group mean.

        :returns: ``float64`` array of length ``n_groups``; ``0`` for empty groups.
        """
        safe = np.maximum(self._count, 1)
        return np.clip(self._syy - self._sy * self._sy / safe, 0.0, None)

    def correlations(self, *, min_count: int = MIN_GROUP_SIZE) -> np.ndarray:
        """Reduce the accumulated sums to one correlation per group.

        Groups with fewer than ``min_count`` observations, or with no variance on either
        axis (see the module docstring), yield NaN.

        :param min_count: Minimum observations required for a finite value.
        :returns: ``float64`` array of length ``n_groups``.
        """
        counts = self._count.astype(np.float64)
        enough = self._count >= max(min_count, MIN_GROUP_SIZE)
        safe = np.where(enough, counts, 1.0)

        cov = self._sxy - self._sx * self._sy / safe
        var_x = self._sxx - self._sx * self._sx / safe
        var_y = self._syy - self._sy * self._sy / safe

        # Rounding can drive a genuinely small variance slightly negative.
        denominator = np.sqrt(np.clip(var_x, 0.0, None) * np.clip(var_y, 0.0, None))
        valid = enough & self.varies() & (denominator > 0.0)

        out = np.full(self.n_groups, np.nan, dtype=np.float64)
        np.divide(cov, denominator, out=out, where=valid)
        return np.clip(out, -1.0, 1.0, out=out)
