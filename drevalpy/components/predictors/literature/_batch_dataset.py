"""Shared dataset helpers for structured literature predictors."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class PairMatrixDataset(Dataset):
    """Simple (X, y) dataset for dense pair-matrix training."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        """Store feature and target tensors for pair-matrix training.

        :param x: Feature matrix with one row per training pair.
        :param y: Target responses with one value per row in *x*.
        """
        self._x = torch.as_tensor(x, dtype=torch.float32)
        self._y = torch.as_tensor(y, dtype=torch.float32).reshape(-1)

    def __len__(self) -> int:
        """Return the number of training pairs.

        :returns: Number of rows in the dataset.
        """
        return len(self._y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the feature row and target for one pair.

        :param index: Zero-based row index.

        :returns: Feature tensor and scalar target tensor.
        """
        return self._x[index], self._y[index]
