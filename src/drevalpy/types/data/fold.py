"""A cross-validation fold: the pair masks to train, validate and test on."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from upath import UPath as Path

from .selection_mask import SelectionMask


@dataclass(frozen=True, slots=True)
class Fold:
    """One cross-validation fold: the pairs to train on, to validate on and to test on.

    Each field is a ``SelectionMask`` with shape (n_samples, n_drugs).
    This format is uniform across all split modes (LPO, LSO, LDO, LTO, LStO).
    """

    train: SelectionMask
    test: SelectionMask
    val: SelectionMask
    metadata: dict[str, Any] = field(default_factory=dict, hash=False)
    stopping: SelectionMask | None = None

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the response matrix (n_samples, n_drugs)."""
        return self.train.shape

    @property
    def train_val(self) -> SelectionMask:
        """Merged train | val mask for final retraining (excludes stopping if set)."""
        return self.train | self.val

    @property
    def fold_id(self) -> str:
        """Deterministic 12-character SHA-256 hex digest of the train, test, and full val masks."""
        full_val = self.val if self.stopping is None else (self.val | self.stopping)
        hasher = hashlib.sha256()
        hasher.update(self.train.mask.tobytes())
        hasher.update(self.test.mask.tobytes())
        hasher.update(full_val.mask.tobytes())
        return hasher.hexdigest()[:12]

    def save(self, path: str | Path) -> None:
        """Save to a .npz file (compressed bool arrays + JSON-encoded metadata).

        :param path: Output file path (should end in .npz).
        """
        pair_seeds: dict[str, int | None] = {name: getattr(self, name).pair_seed for name in ("train", "test", "val")}
        arrays: dict[str, np.ndarray] = {
            "train": self.train.mask,
            "test": self.test.mask,
            "val": self.val.mask,
        }
        if self.stopping is not None:
            arrays["stopping"] = self.stopping.mask
            pair_seeds["stopping"] = self.stopping.pair_seed
        arrays["pair_seeds"] = np.array(json.dumps(pair_seeds))
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
        stopping = SelectionMask(data["stopping"], seeds.get("stopping")) if "stopping" in data else None
        return cls(
            train=SelectionMask(data["train"], seeds.get("train")),
            test=SelectionMask(data["test"], seeds.get("test")),
            val=SelectionMask(data["val"], seeds.get("val")),
            metadata=metadata,
            stopping=stopping,
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
        if self.stopping is not None:
            lines.append(f"    Stopping: {len(self.stopping)} pairs")
        if self.metadata:
            lines.append("    Metadata:")
            for k, v in self.metadata.items():
                lines.append(f"        {k}: {v}")
        return "\n".join(lines)
