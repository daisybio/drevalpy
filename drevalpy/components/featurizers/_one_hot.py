"""Shared one-hot encoding helpers for categorical featurizers."""

from __future__ import annotations

import numpy as np


class OneHotCategoryEncoder:
    """Fit a fixed category vocabulary and emit dense one-hot rows."""

    def __init__(self) -> None:
        self._category_to_index: dict[str, int] = {}

    @property
    def categories(self) -> list[str]:
        return [category for category, _ in sorted(self._category_to_index.items(), key=lambda item: item[1])]

    @property
    def output_dim(self) -> int:
        return len(self._category_to_index)

    def fit_categories(self, categories: np.ndarray) -> None:
        unique = sorted({str(category) for category in np.asarray(categories).reshape(-1)})
        self._category_to_index = {category: index for index, category in enumerate(unique)}

    def transform(self, categories: np.ndarray, *, unknown_zero: bool = True) -> np.ndarray:
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
        return {"categories": self.categories}

    def set_state(self, state: dict[str, object]) -> None:
        categories = state.get("categories")
        if isinstance(categories, list):
            self._category_to_index = {str(category): index for index, category in enumerate(categories)}
