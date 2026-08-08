"""Shared DataLoader factory for predictors that train on pre-built numpy arrays."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def make_tensor_loader(
    *arrays: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Create a DataLoader from one or more numpy arrays.

    Each array is converted to a float32 tensor once upfront and wrapped in a
    TensorDataset.

    :param arrays: One or more numpy arrays of equal first-dimension length.
    :param batch_size: Mini-batch size.
    :param shuffle: Whether to shuffle each epoch.
    :param drop_last: Whether to drop the last incomplete batch.
    :returns: A DataLoader yielding tuples of tensors.
    """
    tensors = [torch.as_tensor(a, dtype=torch.float32) for a in arrays]
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
