"""The input of every metric: finite (prediction, response) pairs, as one group or split into groups.

A pooled score scores one :class:`Pairs`; a per-group score splits the pairs by label with
:func:`group_pairs` and scores each group the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Final

import numpy as np

from ._grouping import defined_labels

#: Absolute tolerance of :func:`is_constant`; ``np.isclose`` adds its default relative tolerance (``1e-5``).
CONSTANT_ATOL: Final[float] = 1e-6


def is_constant(values: np.ndarray) -> bool:
    """Decide whether a non-empty series is constant: ``np.isclose(max, min, atol=CONSTANT_ATOL)``.

    Only the maximum and minimum are compared, so the verdict does not depend on the order of the
    values. The relative tolerance applies, so a series offset by ``1e5`` may span about ``1`` and
    still be constant.

    :param values: Non-empty, finite values.
    :returns: Whether the series counts as constant.
    """
    return bool(np.isclose(np.max(values), np.min(values), atol=CONSTANT_ATOL))


def _finite(y_pred: object, y_true: object, *aligned: object) -> list[np.ndarray]:
    r"""Check the shapes and keep the pairs where both the prediction and the response are finite.

    :param y_pred: Prediction per pair.
    :param y_true: Observed response per pair, in the same order.
    :param \*aligned: Further per-pair arrays (labels) filtered alike.
    :returns: ``y_pred`` and ``y_true`` as ``float64``, then *aligned*, all restricted to the finite pairs.
    :raises ValueError: If the inputs are not one-dimensional and equally long.
    """
    converted = [np.asarray(array) for array in (y_pred, y_true, *aligned)]
    if converted[0].ndim != 1 or any(array.shape != converted[0].shape for array in converted):
        raise ValueError("Predictions, response and labels must be one-dimensional and equally long")
    y_pred, y_true = converted[0].astype(np.float64), converted[1].astype(np.float64)
    usable = np.isfinite(y_pred) & np.isfinite(y_true)
    return [y_pred[usable], y_true[usable], *(array[usable] for array in converted[2:])]


@dataclass(frozen=True, eq=False)
class Pairs:
    """Finite (prediction, response) pairs scored as one group: ``float64``, one-dimensional, equally long.

    Build it with :meth:`finite` from untrusted arrays. The two constancy facts the undefined-score
    rule reads are computed at most once, however many metrics score the pairs.
    """

    #: Predictions, finite.
    y_pred: np.ndarray
    #: Observed responses, finite, aligned with :attr:`y_pred`.
    y_true: np.ndarray

    @classmethod
    def finite(cls, y_pred: object, y_true: object) -> Pairs:
        """Keep the pairs where both the prediction and the response are finite.

        :param y_pred: Prediction per pair.
        :param y_true: Observed response per pair, in the same order.
        :returns: The usable pairs, possibly none.
        """
        return cls(*_finite(y_pred, y_true))

    def __len__(self) -> int:
        """Return the number of pairs."""
        return len(self.y_pred)

    @cached_property
    def prediction_is_constant(self) -> bool:
        """Whether the predictions are constant, see :func:`is_constant`; only read for non-empty pairs."""
        return is_constant(self.y_pred)

    @cached_property
    def response_is_constant(self) -> bool:
        """Whether the responses are constant, see :func:`is_constant`; only read for non-empty pairs."""
        return is_constant(self.y_true)


def group_pairs(
    labels: object,
    y_pred: object,
    y_true: object,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, list[Pairs]]:
    """Split the finite pairs that have a group by label.

    Raises ``ValueError`` if the inputs are not one-dimensional and equally long.

    :param labels: Group label per pair; a NaN label (see :func:`drevalpy.evaluation._grouping.defined_labels`)
        puts the pair in no group.
    :param y_pred: Prediction per pair.
    :param y_true: Observed response per pair.
    :param groups: A fixed, sorted group axis: pairs whose label is not on it are dropped, groups
        without pairs are kept. ``None`` uses the sorted labels of the finite pairs.
    :returns: The group axis and the pairs of each group, in input order within a group.
    """
    y_pred, y_true, labels = _finite(y_pred, y_true, labels)
    grouped = defined_labels(labels)
    labels, y_pred, y_true = labels[grouped], y_pred[grouped], y_true[grouped]
    groups = np.unique(labels) if groups is None else np.asarray(groups)
    if groups.size == 0:
        return groups, []
    codes = np.minimum(np.searchsorted(groups, labels), groups.size - 1)
    on_axis = np.asarray(groups[codes] == labels, dtype=bool)
    codes, y_pred, y_true = codes[on_axis], y_pred[on_axis], y_true[on_axis]
    order = np.argsort(codes, kind="stable")
    bounds = np.cumsum(np.bincount(codes, minlength=groups.size))[:-1]
    split = zip(np.split(y_pred[order], bounds), np.split(y_true[order], bounds), strict=True)
    return groups, [Pairs(prediction, response) for prediction, response in split]
