"""Sklearn response-value transformations for the evaluation pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.base import TransformerMixin, clone
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

if TYPE_CHECKING:
    from drevalpy.types.data.mudatalike import MuDataLike
    from drevalpy.types.data.split_mask import SplitMask


def get_response_transformation(
    response_transformation: str | None,
) -> TransformerMixin | None:
    """Return the sklearn response transformer for a pipeline option.

    :param response_transformation: One of ``"None"``, ``"standard"``, ``"minmax"``, or ``"robust"``.

    :returns: Fitted-ready sklearn transformer, or ``None`` for no transformation.

    :raises ValueError: If *response_transformation* is not recognized.

    :param response_transformation: response transformation.
    :returns: Result of the operation.
    """
    if (response_transformation == "None") or (response_transformation is None):
        return None
    if response_transformation == "standard":
        return StandardScaler()
    if response_transformation == "minmax":
        return MinMaxScaler()
    if response_transformation == "robust":
        return RobustScaler()
    raise ValueError(
        f"Unknown response transformation {response_transformation}. Choose from 'None', 'standard', 'minmax', 'robust'"
    )


def fit_response_transformation(
    prototype: TransformerMixin | None,
    mudataset: MuDataLike,
    scope: SplitMask,
) -> TransformerMixin | None:
    """Fit a clone of *prototype* on the raw responses inside *scope*.

    This is the only place a response transformer is fitted. Every consumer
    downstream receives the already-fitted instance (or ``None``) and only calls
    ``transform`` / ``inverse_transform``. Restricting the fit to a single scope -
    normally the training scope - is what keeps held-out responses out of the
    scaler's statistics, and cloning leaves the caller's prototype unfitted so it
    can be reused across folds.

    :param prototype: Unfitted transformer to clone, or ``None`` for no transformation.
    :param mudataset: Source of the raw response matrix.
    :param scope: Split mask selecting the pairs to fit on.

    :returns: A transformer fitted on the scope's non-NaN responses, or ``None`` when
        *prototype* is ``None``.
    """
    if prototype is None:
        return None
    fitted = clone(prototype)
    pairs = scope.pairs
    responses = mudataset.response_matrix[pairs[:, 0], pairs[:, 1]]
    fitted.fit(responses[~np.isnan(responses)].reshape(-1, 1))
    return fitted
