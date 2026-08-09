"""Unified split masks for cross-validation folds."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from upath import UPath as Path

from .split_mask import SplitMask


@dataclass(frozen=True, slots=True)
class SplitMasks:
    """Collection of train/test/val masks for a single cross-validation fold.

    Each field is a ``SplitMask`` with shape (n_cell_lines, n_drugs).
    This format is uniform across all split modes (LPO, LCO, LDO, LTO).
    """

    train: SplitMask
    test: SplitMask
    val: SplitMask
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the response matrix (n_cell_lines, n_drugs)."""
        return self.train.shape

    def save(self, path: str | Path) -> None:
        """Save to a .npz file (compressed bool arrays + JSON-encoded metadata).

        :param path: Output file path (should end in .npz).
        """
        arrays: dict[str, np.ndarray] = {
            "train": self.train.mask,
            "test": self.test.mask,
            "val": self.val.mask,
        }
        if self.metadata:
            arrays["_metadata"] = np.array(json.dumps(self.metadata))
        np.savez_compressed(Path(path), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> SplitMasks:
        """Load from a .npz file.

        :param path: Path to a .npz file saved by ``save()``.
        :returns: Reconstructed SplitMasks with metadata.
        """
        data = np.load(Path(path), allow_pickle=False)
        metadata = json.loads(str(data["_metadata"])) if "_metadata" in data else {}
        return cls(
            train=SplitMask(data["train"]),
            test=SplitMask(data["test"]),
            val=SplitMask(data["val"]),
            metadata=metadata,
        )

    def __repr__(self) -> str:
        """Formatted summary."""
        lines = [
            "SplitMasks",
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
