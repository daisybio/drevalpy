"""Cross-validation splitting strategies operating on MuData response matrices.

Provides ``SplitMasks`` (the per-fold index containers) and ``MuDataSplitter``
which generates folds for LPO, LCO, LDO, and LTO modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from sklearn.model_selection import GroupKFold, train_test_split


@runtime_checkable
class _MuDataLike(Protocol):
    """Minimal interface expected from a MuDataset-compatible object.

    This allows the splitter to be used with the real MuDataset (once built)
    or with any object satisfying the protocol for testing.
    """

    @property
    def cell_line_ids(self) -> np.ndarray:
        """1-D array of cell line identifiers (obs_names of the response modality)."""
        ...

    @property
    def drug_ids(self) -> np.ndarray:
        """1-D array of drug identifiers (var_names of the response modality)."""
        ...

    @property
    def response_matrix(self) -> np.ndarray:
        """2-D float array (n_cell_lines x n_drugs). NaN where no measurement."""
        ...

    def get_tissue(self, ids: np.ndarray) -> np.ndarray:
        """Return tissue labels for the given cell line IDs."""
        ...


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


class MuDataSplitter:
    """Generate cross-validation folds from a MuDataset-compatible object."""

    _MODES = frozenset({"LPO", "LCO", "LDO", "LTO"})

    def split(
        self,
        mudataset: _MuDataLike,
        mode: str,
        n_splits: int = 5,
        validation_ratio: float = 0.1,
        random_state: int = 42,
    ) -> list[SplitMasks]:
        """Return one ``SplitMasks`` per fold.

        Parameters
        ----------
        mudataset:
            Object exposing cell_line_ids, drug_ids, response_matrix, get_tissue.
        mode:
            One of "LPO", "LCO", "LDO", "LTO".
        n_splits:
            Number of GroupKFold splits.
        validation_ratio:
            Fraction of training groups/samples carved for validation.
        random_state:
            Seed for shuffling and validation splitting.
        """
        if mode not in self._MODES:
            raise ValueError(f"mode must be one of {sorted(self._MODES)}, got {mode!r}")

        if mode == "LPO":
            return self._leave_pair_out(mudataset, n_splits, validation_ratio, random_state)
        if mode == "LCO":
            return self._leave_group_out(mudataset, "cell_line", n_splits, validation_ratio, random_state)
        if mode == "LDO":
            return self._leave_group_out(mudataset, "drug", n_splits, validation_ratio, random_state)
        return self._leave_group_out(mudataset, "tissue", n_splits, validation_ratio, random_state)

    # ------------------------------------------------------------------
    # LPO: pairs of (cell_line_idx, drug_idx) where response is non-NaN
    # ------------------------------------------------------------------

    def _leave_pair_out(
        self,
        mudataset: _MuDataLike,
        n_splits: int,
        validation_ratio: float,
        random_state: int,
    ) -> list[SplitMasks]:
        response = mudataset.response_matrix
        cl_ids = mudataset.cell_line_ids
        drug_ids = mudataset.drug_ids

        # Valid pairs: row, col indices where response is not NaN
        row_idx, col_idx = np.where(~np.isnan(response))

        rng = np.random.default_rng(random_state)
        perm = rng.permutation(len(row_idx))
        row_idx = row_idx[perm]
        col_idx = col_idx[perm]

        # Groups = cell_line + drug string (prevents leakage from replicates)
        groups = np.array([f"{cl_ids[r]}_{drug_ids[c]}" for r, c in zip(row_idx, col_idx, strict=True)])

        gkf = GroupKFold(n_splits=n_splits)
        folds: list[SplitMasks] = []

        for train_pos, test_pos in gkf.split(row_idx, groups=groups):
            if validation_ratio > 0:
                train_pos, val_pos = train_test_split(
                    train_pos,
                    test_size=validation_ratio,
                    shuffle=True,
                    random_state=random_state,
                )
            else:
                val_pos = np.array([], dtype=np.intp)

            folds.append(
                SplitMasks(
                    train_cell_lines=row_idx[train_pos],
                    test_cell_lines=row_idx[test_pos],
                    val_cell_lines=row_idx[val_pos],
                    train_drugs=col_idx[train_pos],
                    test_drugs=col_idx[test_pos],
                    val_drugs=col_idx[val_pos],
                )
            )
        return folds

    # ------------------------------------------------------------------
    # Group-based: LCO, LDO, LTO
    # ------------------------------------------------------------------

    def _leave_group_out(
        self,
        mudataset: _MuDataLike,
        group_kind: str,
        n_splits: int,
        validation_ratio: float,
        random_state: int,
    ) -> list[SplitMasks]:
        cl_ids = mudataset.cell_line_ids
        drug_ids = mudataset.drug_ids

        if group_kind == "cell_line":
            unique_groups = cl_ids
        elif group_kind == "drug":
            unique_groups = drug_ids
        else:
            unique_groups = mudataset.get_tissue(cl_ids)

        rng = np.random.default_rng(random_state)
        perm = rng.permutation(len(unique_groups))
        shuffled_groups = unique_groups[perm]

        gkf = GroupKFold(n_splits=n_splits)
        # GroupKFold needs X with same length as groups -- use dummy array
        dummy = np.zeros(len(shuffled_groups))
        folds: list[SplitMasks] = []

        for train_pos, test_pos in gkf.split(dummy, groups=shuffled_groups):
            train_group_vals = shuffled_groups[train_pos]
            test_group_vals = shuffled_groups[test_pos]

            # Carve validation from training *groups* (not individual samples)
            unique_train = np.unique(train_group_vals)
            if validation_ratio > 0 and len(unique_train) > 1:
                keep_groups, val_groups = train_test_split(
                    unique_train,
                    test_size=validation_ratio,
                    shuffle=True,
                    random_state=random_state,
                )
            else:
                keep_groups = unique_train
                val_groups = np.array([], dtype=unique_train.dtype)

            fold = self._build_group_masks(
                group_kind,
                cl_ids,
                drug_ids,
                mudataset,
                train_groups=keep_groups,
                val_groups=val_groups,
                test_groups=np.unique(test_group_vals),
            )
            folds.append(fold)
        return folds

    @staticmethod
    def _build_group_masks(
        group_kind: str,
        cl_ids: np.ndarray,
        drug_ids: np.ndarray,
        mudataset: _MuDataLike,
        *,
        train_groups: np.ndarray,
        val_groups: np.ndarray,
        test_groups: np.ndarray,
    ) -> SplitMasks:
        """Map group assignments back to cell-line / drug index arrays."""
        if group_kind == "cell_line":
            train_cl = np.where(np.isin(cl_ids, train_groups))[0]
            test_cl = np.where(np.isin(cl_ids, test_groups))[0]
            val_cl = np.where(np.isin(cl_ids, val_groups))[0]
            return SplitMasks(
                train_cell_lines=train_cl,
                test_cell_lines=test_cl,
                val_cell_lines=val_cl,
            )

        if group_kind == "drug":
            train_dr = np.where(np.isin(drug_ids, train_groups))[0]
            test_dr = np.where(np.isin(drug_ids, test_groups))[0]
            val_dr = np.where(np.isin(drug_ids, val_groups))[0]
            all_cl = np.arange(len(cl_ids))
            return SplitMasks(
                train_cell_lines=all_cl,
                test_cell_lines=all_cl,
                val_cell_lines=all_cl,
                train_drugs=train_dr,
                test_drugs=test_dr,
                val_drugs=val_dr,
            )

        # LTO: group on tissue, split cell lines by their tissue
        tissues = mudataset.get_tissue(cl_ids)
        train_cl = np.where(np.isin(tissues, train_groups))[0]
        test_cl = np.where(np.isin(tissues, test_groups))[0]
        val_cl = np.where(np.isin(tissues, val_groups))[0]
        return SplitMasks(
            train_cell_lines=train_cl,
            test_cell_lines=test_cl,
            val_cell_lines=val_cl,
        )
