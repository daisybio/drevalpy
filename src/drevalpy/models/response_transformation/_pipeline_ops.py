"""Fit and apply a response transformation inside one model pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from drevalpy.errors import InvalidInputError

from ._protocol import ResponseTransformer

if TYPE_CHECKING:
    from drevalpy.types.data.batch.response_batch import ResponseBatch
    from drevalpy.types.data.dataset import Dataset


def group_labels(transformer: object, dataset: Dataset, response: ResponseBatch) -> np.ndarray | None:
    """Build the group label columns a group-aware transformer asks for.

    Drug labels come from the batch's drug ids and tissue labels from *dataset*, so the lookup is
    made on the dataset being fitted or predicted, which covers a concatenated cross-study dataset.

    :param transformer: Prototype or fitted transformer; one without ``requires_groups`` needs none.
    :param dataset: Dataset that supplies the sample metadata.
    :param response: Pairs the labels are built for.
    :returns: String labels of shape ``(len(response), len(levels))``, or None if no groups are needed.
    :raises InvalidInputError: If a level is unknown or the dataset has no ``tissue`` sample column.
    """
    if not getattr(transformer, "requires_groups", False):
        return None
    columns = []
    for level in getattr(transformer, "levels", ("drug",)):
        if level == "drug":
            columns.append(np.asarray(response.drug_ids).astype(str))
        elif level == "tissue":
            try:
                tissues = dataset.get_tissue(response.sample_ids)
            except KeyError as error:
                raise InvalidInputError(
                    f"{type(transformer).__name__} groups the response by tissue, but the dataset has no "
                    "'tissue' sample column. Use a dataset with tissues or the 'drug_mean' transformation."
                ) from error
            columns.append(np.asarray(tissues).astype(str))
        else:
            raise InvalidInputError(f"Unknown group level {level!r}; choose from 'drug' and 'tissue'")
    return np.stack(columns, axis=1)


def fit_response_transformation(
    prototype: ResponseTransformer | None, response: np.ndarray, groups: np.ndarray | None = None
) -> ResponseTransformer | None:
    """Fit an independent clone of *prototype* on the observed responses.

    :param prototype: Unfitted transformer, or None.
    :param response: Observed responses in the actual training scope.
    :param groups: Group labels aligned with *response*; passed only to a transformer with ``requires_groups``.
    :returns: A fitted clone, or None.
    :raises TypeError: If the prototype is not a :class:`ResponseTransformer`, or cloning does not create an
        independent instance.
    """
    if prototype is None:
        return None
    from sklearn.base import clone

    if not isinstance(prototype, ResponseTransformer):
        raise TypeError("Response transformer must implement fit, transform, and inverse_transform")
    fitted = clone(prototype)
    if fitted is prototype:
        raise TypeError("Response transformer cloning must create an independent instance")
    values = np.asarray(response)
    valid = ~np.isnan(values)
    fitted.fit(values[valid].reshape(-1, 1), **_group_kwargs(fitted, groups, valid))
    return fitted


def apply_response_transformation(
    transformer: ResponseTransformer | None,
    values: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    inverse: bool = False,
) -> np.ndarray:
    """Transform one response column while preserving missing response positions.

    :param transformer: Fitted transformer, or None.
    :param values: One value per requested response position.
    :param groups: Group labels aligned with *values*; passed only to a transformer with ``requires_groups``.
    :param inverse: Return values in original response units.
    :returns: Aligned one-dimensional values.
    :raises ValueError: If the transformer changes the response shape.
    """
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("Response values must be one-dimensional")
    valid = ~np.isnan(values)
    if transformer is None or not valid.any():
        return values
    operation = transformer.inverse_transform if inverse else transformer.transform
    transformed = np.asarray(operation(values[valid].reshape(-1, 1), **_group_kwargs(transformer, groups, valid)))
    if transformed.shape != (int(valid.sum()), 1):
        raise ValueError("Response transformer must preserve the shape (n_responses, 1)")
    result = np.full(values.shape, np.nan, dtype=np.result_type(transformed.dtype, np.float64))
    result[valid] = transformed[:, 0]
    return result


def _group_kwargs(transformer: object, groups: np.ndarray | None, valid: np.ndarray) -> dict[str, np.ndarray]:
    """Return the ``groups`` keyword for a group-aware transformer, restricted to the observed positions."""
    if groups is None or not getattr(transformer, "requires_groups", False):
        return {}
    return {"groups": np.asarray(groups)[valid]}
