"""Shared dataset helpers for structured literature predictors."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class PairMatrixDataset(Dataset):
    """Simple (X, y) dataset for dense pair-matrix training."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self._x = torch.as_tensor(x, dtype=torch.float32)
        self._y = torch.as_tensor(y, dtype=torch.float32).reshape(-1)

    def __len__(self) -> int:
        return len(self._y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._x[index], self._y[index]
