"""Protocol for Dataset-compatible objects.

This is used to simplify testing, so tests don't have to implement all features of the Dataset class.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class MuDataLike(Protocol):
    """Minimal interface expected from a Dataset-compatible object.

    This allows splitters and other components to work with the real Dataset
    or with any object satisfying the protocol for testing.
    """

    @property
    def sample_ids(self) -> np.ndarray:
        """1-D array of sample identifiers (obs_names of the response modality)."""
        ...

    @property
    def drug_ids(self) -> np.ndarray:
        """1-D array of drug identifiers (var_names of the response modality)."""
        ...

    @property
    def response_matrix(self) -> np.ndarray:
        """2-D float array (n_samples x n_drugs). NaN where no measurement."""
        ...

    def get_tissue(self, ids: np.ndarray) -> np.ndarray:
        """Return tissue labels for the given sample IDs."""
        ...

    def sample_column(self, column: str, ids: np.ndarray) -> np.ndarray:
        """Return one per-sample metadata column for the given sample IDs.

        IDs that are not samples of the dataset give NaN. A column the dataset
        does not have raises :class:`KeyError`, naming the available columns.
        """
        ...

    def response_layer_names(self) -> list[str]:
        """Names of the available response layers."""
        ...

    def get_response_layer(self, name: str) -> np.ndarray:
        """2-D float array (n_samples x n_drugs) for a named response layer."""
        ...
