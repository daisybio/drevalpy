"""Shared one-hot encoding helpers for categorical featurizers."""

from __future__ import annotations

import numpy as np


class OneHotCategoryEncoder:
    """Fit a fixed category vocabulary and emit dense one-hot rows."""

    def __init__(self) -> None:
        """Initialize instance state."""
        self._category_to_index: dict[str, int] = {}

    @property
    def categories(self) -> list[str]:
        """Return fitted category labels in index order."""
        return [category for category, _ in sorted(self._category_to_index.items(), key=lambda item: item[1])]

    @property
    def output_dim(self) -> int:
        """Return the number of fitted categories."""
        return len(self._category_to_index)

    def fit_categories(self, categories: np.ndarray) -> None:
        """Learn the category vocabulary from observed values.

        :param categories: Observed values; each is stringified and deduplicated.
        """
        unique = sorted({str(category) for category in np.asarray(categories).reshape(-1)})
        self._category_to_index = {category: index for index, category in enumerate(unique)}

    def transform(self, categories: np.ndarray, *, unknown_zero: bool = True) -> np.ndarray:
        """Encode values as one-hot rows against the fitted vocabulary.

        :param categories: Values to encode, one per output row.
        :param unknown_zero: Emit an all-zero row for unseen categories instead of raising.
        :returns: Float32 matrix of shape ``(len(categories), output_dim)``.
        :raises KeyError: If a category is unseen and *unknown_zero* is false.
        """
        if self.output_dim == 0:
            return np.empty((len(categories), 0), dtype=np.float32)
        matrix = np.zeros((len(categories), self.output_dim), dtype=np.float32)
        for row, category in enumerate(np.asarray(categories).reshape(-1)):
            index = self._category_to_index.get(str(category))
            if index is None:
                if not unknown_zero:
                    msg = f"Unknown category {category!r} for one-hot featurizer"
                    raise KeyError(msg)
                continue
            matrix[row, index] = 1.0
        return matrix

    def get_state(self) -> dict[str, object]:
        """Return serializable fitted state (the category labels)."""
        return {"categories": self.categories}

    def set_state(self, state: dict[str, object]) -> None:
        """Restore state from a prior ``get_state`` mapping.

        :param state: Mapping previously returned by ``get_state``.
        """
        categories = state.get("categories")
        if isinstance(categories, list):
            self._category_to_index = {str(category): index for index, category in enumerate(categories)}
