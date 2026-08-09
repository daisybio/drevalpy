"""Index arrays for a single cross-validation fold."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from upath import UPath as Path


@dataclass(frozen=True, slots=True)
class SplitMasks:
    """Index arrays for a single cross-validation fold.

    For LCO/LTO the drug indices are *None* (all drugs used for all splits).
    For LDO the cell line indices cover all cell lines and drug indices differ.
    For LPO both cell_line and drug indices are populated (paired).
    """

    train_cell_lines: np.ndarray
    test_cell_lines: np.ndarray
    val_cell_lines: np.ndarray

    train_drugs: np.ndarray | None = None
    test_drugs: np.ndarray | None = None
    val_drugs: np.ndarray | None = None

    def save(self, path: str | Path) -> None:
        """Save to a .npz file.

        :param path: Output file path (should end in .npz).
        """
        arrays: dict[str, np.ndarray] = {
            "train_cell_lines": self.train_cell_lines,
            "test_cell_lines": self.test_cell_lines,
            "val_cell_lines": self.val_cell_lines,
        }
        if self.train_drugs is not None:
            arrays["train_drugs"] = self.train_drugs
        if self.test_drugs is not None:
            arrays["test_drugs"] = self.test_drugs
        if self.val_drugs is not None:
            arrays["val_drugs"] = self.val_drugs
        np.savez_compressed(Path(path), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> SplitMasks:
        """Load from a .npz file.

        :param path: Path to a .npz file saved by ``save()``.
        :returns: Reconstructed SplitMasks.
        """
        data = np.load(Path(path))
        return cls(
            train_cell_lines=data["train_cell_lines"],
            test_cell_lines=data["test_cell_lines"],
            val_cell_lines=data["val_cell_lines"],
            train_drugs=data["train_drugs"] if "train_drugs" in data else None,
            test_drugs=data["test_drugs"] if "test_drugs" in data else None,
            val_drugs=data["val_drugs"] if "val_drugs" in data else None,
        )
