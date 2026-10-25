"""Single 2D boolean mask for train/predict operations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SplitMask:
    """2D boolean mask defining which (sample, drug) pairs to operate on.

    The ``mask`` array has shape (n_samples, n_drugs) with True at positions
    that should be included. When ``pair_seed`` is set, ``.pairs`` returns
    the indices in a shuffled order (for robustness testing). The mask is
    read-only, so a mask can be shared without copying.
    """

    mask: np.ndarray
    pair_seed: int | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Store the mask as a read-only boolean array, copying it if the caller could still write to it."""
        mask = np.asarray(self.mask, dtype=bool)
        if mask.flags.writeable:
            mask = mask.copy()
            mask.flags.writeable = False
        object.__setattr__(self, "mask", mask)

    @classmethod
    def from_pairs(cls, pairs: np.ndarray, shape: tuple[int, int], pair_seed: int | None = None) -> SplitMask:
        """Construct from a (n_pairs, 2) index array and matrix shape.

        :param pairs: (n_pairs, 2) array of (sample, drug) indices.
        :param shape: Shape (n_samples, n_drugs) of the mask.
        :param pair_seed: Seed that shuffles the order of ``.pairs``; ``None`` keeps row-major order.
        :returns: The mask with True at *pairs*.
        """
        mask = np.zeros(shape, dtype=bool)
        if len(pairs) > 0:
            mask[pairs[:, 0], pairs[:, 1]] = True
        return cls(mask, pair_seed)

    def shuffled(self, seed: int) -> SplitMask:
        """Return a new SplitMask whose .pairs are in shuffled order.

        :param seed: Random seed for reproducible pair ordering.
        :returns: SplitMask with same content but shuffled .pairs output.
        """
        return SplitMask(mask=self.mask, pair_seed=seed)

    @property
    def pairs(self) -> np.ndarray:
        """Pair indices as (n_pairs, 2) array.

        Order is deterministic (row-major) unless a shuffle seed is set.
        """
        p = np.argwhere(self.mask)
        if self.pair_seed is not None:
            rng = np.random.default_rng(self.pair_seed)
            p = rng.permutation(p)
        return p

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the underlying mask."""
        return self.mask.shape  # type: ignore[return-value]

    def __len__(self) -> int:
        """Return the number of True entries in the mask."""
        return int(self.mask.sum())

    def __or__(self, other: SplitMask) -> SplitMask:
        """Logical OR of two masks."""
        seed = self._combined_seed(other)
        return SplitMask(self.mask | other.mask, seed)

    def __and__(self, other: SplitMask) -> SplitMask:
        """Logical AND of two masks."""
        seed = self._combined_seed(other)
        return SplitMask(self.mask & other.mask, seed)

    def __invert__(self) -> SplitMask:
        """Logical NOT of the mask."""
        return SplitMask(~self.mask, self.pair_seed)

    def _combined_seed(self, other: SplitMask) -> int | None:
        """Preserve pair-order provenance, refusing incompatible combinations."""
        if self.shape != other.shape:
            raise ValueError(f"Cannot combine mask shapes {self.shape} and {other.shape}")
        if self.pair_seed is not None and other.pair_seed is not None and self.pair_seed != other.pair_seed:
            raise ValueError("Cannot combine masks with conflicting pair-order seeds")
        return self.pair_seed if self.pair_seed is not None else other.pair_seed

    def sum(self) -> int:
        """Return the number of True entries."""
        return int(self.mask.sum())

    def __eq__(self, other: object) -> bool:
        """Equality based on mask contents."""
        if not isinstance(other, SplitMask):
            return NotImplemented
        return np.array_equal(self.mask, other.mask)
