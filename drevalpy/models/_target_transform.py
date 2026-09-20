"""Shape-preserving target transformations owned by one fitted pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np

from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.split_mask import SplitMask


def validate_fit_scopes(dataset: Dataset, scope: SplitMask, stopping: SplitMask | None) -> None:
    """Check dataset-backed selections without imposing split-mode entity rules.

    :param dataset: Source data.
    :param scope: Training or prediction positions.
    :param stopping: Optional disjoint stopping positions.
    :raises TypeError: If inputs do not meet the public model contract.
    :raises ValueError: If scope shapes differ or fitting selections overlap.
    """
    if not isinstance(dataset, Dataset):
        raise TypeError("Model operations require a Dataset")
    for selection in (scope,) if stopping is None else (scope, stopping):
        if not isinstance(selection, SplitMask):
            raise TypeError("Model scope must be a SplitMask")
        if selection.shape != dataset.response_matrix.shape:
            raise ValueError("Model scope shape must match the dataset response matrix")
    if stopping is not None and np.any(scope.mask & stopping.mask):
        raise ValueError("Training and early-stopping scopes must be response-disjoint")


def fit_target_transform(prototype: Any, response: np.ndarray) -> Any:
    """Fit an independent sklearn-compatible transform on observed targets.

    :param prototype: Unfitted transformer, or None.
    :param response: Observed targets in the actual training scope.
    :returns: A fitted clone, or None.
    :raises TypeError: If the prototype does not implement the required methods.
    """
    if prototype is None:
        return None
    from sklearn.base import clone

    if not all(callable(getattr(prototype, name, None)) for name in ("fit", "transform", "inverse_transform")):
        raise TypeError("Response transformer must implement fit, transform, and inverse_transform")
    fitted = clone(prototype)
    if fitted is prototype:
        raise TypeError("Response transformer cloning must create an independent instance")
    values = np.asarray(response)
    fitted.fit(values[~np.isnan(values)].reshape(-1, 1))
    return fitted


def apply_target_transform(transformer: Any, values: np.ndarray, *, inverse: bool = False) -> np.ndarray:
    """Transform one target column while preserving missing response positions.

    :param transformer: Fitted transformer, or None.
    :param values: One value per requested response position.
    :param inverse: Return values in original response units.
    :returns: Aligned one-dimensional values.
    :raises ValueError: If the transformer changes target shape.
    """
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("Response values must be one-dimensional")
    valid = ~np.isnan(values)
    if transformer is None or not valid.any():
        return values
    operation = transformer.inverse_transform if inverse else transformer.transform
    transformed = np.asarray(operation(values[valid].reshape(-1, 1)))
    if transformed.shape != (int(valid.sum()), 1):
        raise ValueError("Response transformer must preserve the shape (n_responses, 1)")
    result = np.full(values.shape, np.nan, dtype=np.result_type(transformed.dtype, np.float64))
    result[valid] = transformed[:, 0]
    return result
