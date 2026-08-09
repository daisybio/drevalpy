"""Single 2D boolean mask for train/predict operations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SplitMask:
    """2D boolean mask defining which (cell_line, drug) pairs to operate on.

    The ``mask`` array has shape (n_cell_lines, n_drugs) with True at positions
    that should be included.
    """

    mask: np.ndarray

    def __post_init__(self) -> None:
        """Ensure mask is stored as a boolean numpy array."""
        object.__setattr__(self, "mask", np.asarray(self.mask, dtype=bool))

    @classmethod
    def from_pairs(cls, pairs: np.ndarray, shape: tuple[int, int]) -> SplitMask:
        """Construct from a (n_pairs, 2) index array and matrix shape."""
        mask = np.zeros(shape, dtype=bool)
        if len(pairs) > 0:
            mask[pairs[:, 0], pairs[:, 1]] = True
        return cls(mask)

    @property
    def pairs(self) -> np.ndarray:
        """Pair indices as (n_pairs, 2) array — computed from the mask."""
        return np.argwhere(self.mask)

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the underlying mask."""
        return self.mask.shape  # type: ignore[return-value]

    def __len__(self) -> int:
        """Number of True entries in the mask."""
        return int(self.mask.sum())

    def __or__(self, other: SplitMask) -> SplitMask:
        """Logical OR of two masks."""
        return SplitMask(self.mask | other.mask)

    def __and__(self, other: SplitMask) -> SplitMask:
        """Logical AND of two masks."""
        return SplitMask(self.mask & other.mask)

    def __invert__(self) -> SplitMask:
        """Logical NOT of the mask."""
        return SplitMask(~self.mask)

    def any(self) -> bool:
        """Whether any entry is True."""
        return bool(self.mask.any())

    def sum(self) -> int:
        """Number of True entries."""
        return int(self.mask.sum())

    def __eq__(self, other: object) -> bool:
        """Equality based on mask contents."""
        if not isinstance(other, SplitMask):
            return NotImplemented
        return np.array_equal(self.mask, other.mask)

    def __hash__(self) -> int:
        """Hash based on mask bytes."""
        return hash(self.mask.tobytes())
