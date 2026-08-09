"""Unified 2D pair arrays for cross-validation folds."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from upath import UPath as Path


@dataclass(frozen=True, slots=True)
class SplitMasks:
    """2D pair arrays for a single cross-validation fold.

    Each array has shape (n_pairs, 2) where column 0 is the cell line index
    and column 1 is the drug index into the response matrix. This format is
    uniform across all split modes (LPO, LCO, LDO, LTO).
    """

    train: np.ndarray
    test: np.ndarray
    val: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Save to a .npz file (arrays + JSON-encoded metadata).

        :param path: Output file path (should end in .npz).
        """
        arrays: dict[str, np.ndarray] = {
            "train": self.train,
            "test": self.test,
            "val": self.val,
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
            train=data["train"],
            test=data["test"],
            val=data["val"],
            metadata=metadata,
        )

    def __repr__(self) -> str:
        """Formatted summary."""
        lines = [
            "SplitMasks",
            f"    Train: {len(self.train)} pairs",
            f"    Test: {len(self.test)} pairs",
            f"    Val: {len(self.val)} pairs",
        ]
        if self.metadata:
            lines.append("    Metadata:")
            for k, v in self.metadata.items():
                lines.append(f"        {k}: {v}")
        return "\n".join(lines)
