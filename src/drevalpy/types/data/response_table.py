"""Lightweight immutable table of sample-drug response pairs."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from drevalpy.types.data.dataset import Dataset
    from drevalpy.types.data.selection_mask import SelectionMask


@dataclass(frozen=True, slots=True)
class ResponseTable:
    """Lightweight immutable container for sample-drug response pairs.

    Provides ``.response``, ``.sample_ids``, ``.drug_ids``, and ``__len__``.
    ``layers`` holds further response layers of the same pairs (such as ``regulation``)
    by name, one value per pair; every subset keeps them aligned with the response.
    """

    response: np.ndarray
    sample_ids: np.ndarray
    drug_ids: np.ndarray
    layers: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check that every layer holds one value per pair.

        :raises ValueError: If a layer is not one-dimensional with one value per pair.
        """
        for name, values in self.layers.items():
            if np.ndim(values) != 1 or len(values) != len(self.response):
                raise ValueError(f"Response layer {name!r} must be one-dimensional with one value per pair")

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        mask: SelectionMask,
        *,
        observed_only: bool = False,
        layers: Collection[str] = (),
    ) -> ResponseTable:
        """Extract aligned responses and identifiers in the mask's pair order.

        :param dataset: Response matrix and entity identifiers.
        :param mask: Requested response positions, possibly shuffled.
        :param observed_only: Drop missing targets for training only.
        :param layers: Response layers to read at the same pairs; names the dataset does not have are left out.
        :returns: Response table, preserving missing targets by default.
        """
        pairs = mask.pairs
        values = np.asarray(dataset.response_matrix[pairs[:, 0], pairs[:, 1]], dtype=np.float64)
        if observed_only:
            keep = ~np.isnan(values)
            pairs, values = pairs[keep], values[keep]
        available = set(dataset.measures())
        layer_values = {
            name: dataset.get_response_layer(name)[pairs[:, 0], pairs[:, 1]] for name in layers if name in available
        }
        return cls(values, dataset.sample_ids[pairs[:, 0]], dataset.drug_ids[pairs[:, 1]], layer_values)

    def __getitem__(self, mask: np.ndarray | slice) -> ResponseTable:
        """Select pairs by boolean/integer mask or slice.

        :param mask: Boolean mask, integer index array, or slice over the pairs.
        :returns: Response table with the selected entries.
        """
        return ResponseTable(
            self.response[mask],
            self.sample_ids[mask],
            self.drug_ids[mask],
            {name: values[mask] for name, values in self.layers.items()},
        )

    def __len__(self) -> int:
        """Return the number of response pairs.

        :returns: Length of the response array.
        """
        return len(self.response)
