"""Lightweight immutable container for response triples passed to predictors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from drevalpy.types.data.dataset import Dataset
    from drevalpy.types.data.split_mask import SplitMask


@dataclass(frozen=True, slots=True)
class ResponseBatch:
    """Lightweight immutable container for response triples passed to predictors.

    Provides ``.response``, ``.sample_ids``,
    ``.drug_ids``, and ``__len__``.
    """

    response: np.ndarray
    sample_ids: np.ndarray
    drug_ids: np.ndarray

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        scope: SplitMask,
        *,
        observed_only: bool = False,
    ) -> ResponseBatch:
        """Extract aligned responses and identifiers in the scope's pair order.

        :param dataset: Response matrix and entity identifiers.
        :param scope: Requested response positions, possibly shuffled.
        :param observed_only: Drop missing targets for training only.
        :returns: Response triples, preserving missing targets by default.
        """
        pairs = scope.pairs
        values = np.asarray(dataset.response_matrix[pairs[:, 0], pairs[:, 1]], dtype=np.float64)
        if observed_only:
            keep = ~np.isnan(values)
            pairs, values = pairs[keep], values[keep]
        return cls(values, dataset.sample_ids[pairs[:, 0]], dataset.drug_ids[pairs[:, 1]])

    def subset(self, mask: np.ndarray) -> ResponseBatch:
        """Keep the pairs selected by a boolean mask.

        :param mask: One boolean value per pair.
        :returns: Batch with the selected pairs, in their original order.
        """
        return ResponseBatch(self.response[mask], self.sample_ids[mask], self.drug_ids[mask])

    def __len__(self) -> int:
        """Return the number of response pairs.

        :returns: Length of the response array.
        """
        return len(self.response)
