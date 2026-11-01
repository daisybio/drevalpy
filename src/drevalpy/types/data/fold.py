"""A cross-validation fold: the pair masks to train, validate and test on."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from upath import UPath as Path

from .pair_mask import PairMask


@dataclass(frozen=True, slots=True)
class Fold:
    """One cross-validation fold: the pairs to train on, to validate on and to test on.

    Each field is a ``PairMask`` with shape (n_samples, n_drugs).
    This format is uniform across all split modes (LPO, LSO, LDO, LTO, LStO).
    """

    train: PairMask
    test: PairMask
    val: PairMask
    metadata: dict[str, Any] = field(default_factory=dict, hash=False)

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the response matrix (n_samples, n_drugs)."""
        return self.train.shape

    @property
    def train_val(self) -> PairMask:
        """Merged train | val mask for final retraining."""
        return self.train | self.val

    def early_stopping_mask(self, fraction: float = 0.25) -> tuple[PairMask, PairMask]:
        """Split the val mask into early-stopping and remaining validation.

        :param fraction: Fraction of val pairs to reserve for early stopping.
        :returns: Tuple of (early_stopping_mask, remaining_val_mask).
        """
        val_pairs = self.val.pairs
        n_es = max(1, int(len(val_pairs) * fraction))
        seed = self.val.pair_seed
        return (
            PairMask.from_pairs(val_pairs[:n_es], self.shape, seed),
            PairMask.from_pairs(val_pairs[n_es:], self.shape, seed),
        )

    def save(self, path: str | Path) -> None:
        """Save to a .npz file (compressed bool arrays + JSON-encoded metadata).

        :param path: Output file path (should end in .npz).
        """
        arrays: dict[str, np.ndarray] = {
            "train": self.train.mask,
            "test": self.test.mask,
            "val": self.val.mask,
            "pair_seeds": np.array(
                json.dumps({name: getattr(self, name).pair_seed for name in ("train", "test", "val")})
            ),
        }
        if self.metadata:
            arrays["_metadata"] = np.array(json.dumps(self.metadata))
        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with resolved.open("wb") as file:
            np.savez_compressed(file, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> Fold:
        """Load from a .npz file.

        :param path: Path to a .npz file saved by ``save()``.
        :returns: Reconstructed Fold with metadata.
        """
        data = np.load(Path(path), allow_pickle=False)
        metadata = json.loads(str(data["_metadata"])) if "_metadata" in data else {}
        seeds = json.loads(str(data["pair_seeds"])) if "pair_seeds" in data else {}
        return cls(
            train=PairMask(data["train"], seeds.get("train")),
            test=PairMask(data["test"], seeds.get("test")),
            val=PairMask(data["val"], seeds.get("val")),
            metadata=metadata,
        )

    def __repr__(self) -> str:
        """Return a formatted summary."""
        lines = [
            "Fold",
            f"    Shape: {self.shape}",
            f"    Train: {len(self.train)} pairs",
            f"    Test: {len(self.test)} pairs",
            f"    Val: {len(self.val)} pairs",
        ]
        if self.metadata:
            lines.append("    Metadata:")
            for k, v in self.metadata.items():
                lines.append(f"        {k}: {v}")
        return "\n".join(lines)
