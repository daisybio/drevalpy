"""Shared DataLoader factory with lazy pair-level index lookup."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class IndexedPairDataset(Dataset):
    """Looks up entity-level features by pair index on each access.

    Instead of materializing a full pair-level feature matrix upfront, this
    dataset stores compact entity-level matrices and performs the lookup
    per-sample in ``__getitem__``.
    """

    def __init__(
        self,
        *feature_specs: tuple[np.ndarray, np.ndarray],
        response: np.ndarray | None = None,
    ) -> None:
        """Initialize with entity matrices and corresponding pair indices.

        :param feature_specs: Each positional arg is a tuple of
            ``(entity_matrix, pair_index_array)`` where entity_matrix has shape
            ``[n_entities, d]`` and pair_index_array has shape ``[n_pairs]``.
        :param response: Optional pair-level response array of shape ``[n_pairs]``.
        """
        self._matrices = [torch.as_tensor(m, dtype=torch.float32) for m, _ in feature_specs]
        self._indices = [torch.as_tensor(i, dtype=torch.long) for _, i in feature_specs]
        self._response = torch.as_tensor(response, dtype=torch.float32) if response is not None else None
        self._n_pairs = int(self._indices[0].shape[0]) if self._indices else 0

    def __len__(self) -> int:
        """Return number of pairs.

        :returns: Dataset length.
        """
        return self._n_pairs

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        """Look up features for pair *idx*.

        :param idx: Pair index.
        :returns: Tuple of feature tensors, optionally followed by the response scalar.
        """
        feats = tuple(m[i[idx]] for m, i in zip(self._matrices, self._indices, strict=True))
        if self._response is not None:
            return (*feats, self._response[idx])
        return feats


def make_pair_loader(
    *feature_specs: tuple[np.ndarray, np.ndarray],
    response: np.ndarray | None = None,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Create a DataLoader that lazily indexes entity features per mini-batch.

    :param feature_specs: Each positional arg is ``(entity_matrix, pair_indices)``.
    :param response: Optional pair-level response vector.
    :param batch_size: Mini-batch size.
    :param shuffle: Whether to shuffle each epoch.
    :param drop_last: Whether to drop the last incomplete batch.
    :returns: A DataLoader yielding tuples of tensors.
    """
    ds = IndexedPairDataset(*feature_specs, response=response)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
