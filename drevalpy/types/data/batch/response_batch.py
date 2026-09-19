"""Lightweight immutable container for response triples passed to predictors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sklearn.base import TransformerMixin

    from drevalpy.types.data.dataset import Dataset
    from drevalpy.types.data.split_mask import SplitMask


@dataclass(frozen=True, slots=True)
class ResponseBatch:
    """Lightweight immutable container for response triples passed to predictors.

    Provides ``.response``, ``.cell_line_ids``,
    ``.drug_ids``, and ``__len__``.
    """

    response: np.ndarray
    cell_line_ids: np.ndarray
    drug_ids: np.ndarray

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        scope: SplitMask,
        *,
        observed_only: bool = False,
        transformation: TransformerMixin | None = None,
    ) -> ResponseBatch:
        """Extract aligned responses and identifiers in the scope's pair order.

        :param dataset: Response matrix and entity identifiers.
        :param scope: Requested response positions, possibly shuffled.
        :param observed_only: Drop missing targets for training only.
        :param transformation: Fitted training-response transformation.
        :returns: Response triples, preserving missing targets by default.
        """
        pairs = scope.pairs
        values = np.asarray(dataset.response_matrix[pairs[:, 0], pairs[:, 1]], dtype=np.float64)
        if observed_only:
            keep = ~np.isnan(values)
            pairs, values = pairs[keep], values[keep]
        if transformation is not None and len(values):
            values = transformation.transform(values.reshape(-1, 1)).ravel()
        return cls(values, dataset.cell_line_ids[pairs[:, 0]], dataset.drug_ids[pairs[:, 1]])

    def __len__(self) -> int:
        """Return the number of response pairs.

        :returns: Length of the response array.
        """
        return len(self.response)
