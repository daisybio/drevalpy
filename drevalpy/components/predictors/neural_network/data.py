"""Tensor dataset helpers for dense neural-network training."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class PairMatrixDataset(Dataset):
    """Expose a dense pair matrix and response vector as float tensors."""

    def __init__(self, features: np.ndarray, response: np.ndarray) -> None:
        """Initialize instance state.

        :param features: features.
        :param response: response.
        """
        self._features = torch.as_tensor(features, dtype=torch.float32)
        self._response = torch.as_tensor(response, dtype=torch.float32).reshape(-1)

    def __len__(self) -> int:
        """Return the number of response values.

        :returns: Number of training pairs in the dataset.
        """
        return len(self._response)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one feature row and response value.

        :param index: Zero-based pair index.
        :returns: Feature tensor and scalar response tensor for *index*.
        """
        return self._features[index], self._response[index]
