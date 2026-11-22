"""Leakage groups: the unit of a split mode that must not straddle two parts of a split.

Every pair belongs to one group per mode:

======  ==============================
Mode    Group of a pair
======  ==============================
LSO     its source sample
LDO     its drug
LTO     the tissue of its sample
LStO    the study of its sample
LPO     its (source sample, drug)
======  ==============================

The source sample is the ``source_sample_id`` sample column that
:func:`drevalpy.data.concat` writes, so a sample measured in two studies is one group;
a dataset without that column uses the row ID. The splitters cut along these groups,
the validators check them, and the early-stopping split draws whole groups.

Everything that differs between the modes is one :class:`LeakageGroups` entry of
:data:`MODES`, so adding a mode is one entry there plus its name in :data:`Validation`.

Group IDs are numbered so the built-in folds of a dataset that is not concatenated
are the folds the per-mode code produced: by first appearance in row-major order for
LSO, LDO and LPO (row index, column index, flat pair index), and by sorted label for
LTO and LStO.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from drevalpy.types import Dataset

Validation = Literal["LSO", "LDO", "LPO", "LTO", "LStO"]

SOURCE_SAMPLE_COLUMN = "source_sample_id"
"""Sample column holding the ID a sample had in its own study before :func:`drevalpy.data.concat`."""


@dataclass(frozen=True)
class LeakageGroups:
    """How one split mode groups its pairs."""

    #: The groups in the plural, for messages (``"tissues"``).
    unit: str
    #: Group ID of every pair, shape (n_samples, n_drugs).
    groups: Callable[[Dataset], np.ndarray]
    #: Names of the groups of the pairs at the given (rows, columns), for messages.
    labels: Callable[[Dataset, np.ndarray, np.ndarray], list[str]]
    #: Mode whose groups separate train, validation and the early-stopping set. When it is not
    #: the mode itself, only the test part is held out by the mode's groups.
    cut: Validation


def pair_groups(dataset: Dataset, mode: Validation) -> np.ndarray:
    """Return the group ID of every pair under split *mode*.

    :param dataset: Dataset whose pairs are grouped; the sample columns ``source_sample_id``,
        ``tissue`` and ``study`` are read as the mode needs them.
    :param mode: Leakage constraint whose unit the groups are.
    :returns: ``int64`` array of shape (n_samples, n_drugs) with one group ID per pair.
    """
    return MODES[mode].groups(dataset)


def in_groups(groups: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Return whether each pair's group is one of *ids*.

    Equal to ``np.isin(groups, ids)``, through a lookup table over the group IDs, which is
    about ten times faster on a response matrix of a few hundred thousand pairs.

    :param groups: Group ID of every pair, as :func:`pair_groups` returns.
    :param ids: Group IDs to select.
    :returns: Boolean mask of the shape of *groups*.
    """
    selected = np.zeros(groups.max() + 1, dtype=bool)
    selected[ids] = True
    return selected[groups]


def _source_samples(dataset: Dataset) -> np.ndarray:
    """Return the source sample of every row, or the row ID without a ``source_sample_id`` column.

    :param dataset: Dataset whose rows are resolved.
    :returns: One string per row of the response matrix.
    """
    try:
        return dataset.sample_column(SOURCE_SAMPLE_COLUMN, dataset.sample_ids).astype(str)
    except KeyError:
        return np.asarray(dataset.sample_ids, dtype=str)


def _tissues(dataset: Dataset) -> np.ndarray:
    return np.asarray(dataset.get_tissue(dataset.sample_ids))


def _studies(dataset: Dataset) -> np.ndarray:
    return dataset.sample_column("study", dataset.sample_ids).astype(str)


def _first_appearance(values: np.ndarray) -> np.ndarray:
    """Return the ID of each value, numbering the distinct *values* 0, 1, ... by first appearance."""
    _, first, inverse = np.unique(values, return_index=True, return_inverse=True)
    rank = np.empty(len(first), dtype=np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(len(first))
    return rank[inverse.ravel()]


def _per_row(dataset: Dataset, row_groups: np.ndarray) -> np.ndarray:
    """Give every pair the group of its row."""
    return np.repeat(row_groups.astype(np.int64).reshape(-1, 1), dataset.response_matrix.shape[1], axis=1)


def _by_sorted_label(rows_of: Callable[[Dataset], np.ndarray]) -> Callable[[Dataset], np.ndarray]:
    """Group the pairs by a per-row label, numbering the labels in sorted order."""
    return lambda dataset: _per_row(dataset, np.unique(rows_of(dataset), return_inverse=True)[1])


def _sample_groups(dataset: Dataset) -> np.ndarray:
    return _per_row(dataset, _first_appearance(_source_samples(dataset)))


def _drug_groups(dataset: Dataset) -> np.ndarray:
    n_samples, n_drugs = dataset.response_matrix.shape
    return np.broadcast_to(np.arange(n_drugs, dtype=np.int64), (n_samples, n_drugs)).copy()


def _sample_drug_groups(dataset: Dataset) -> np.ndarray:
    n_samples, n_drugs = dataset.response_matrix.shape
    samples = _first_appearance(_source_samples(dataset))
    return _first_appearance((samples[:, None] * n_drugs + np.arange(n_drugs)).ravel()).reshape(n_samples, n_drugs)


def _row_labels(rows_of: Callable[[Dataset], np.ndarray]) -> Callable[[Dataset, np.ndarray, np.ndarray], list[str]]:
    """Name a pair's group by a per-row label."""
    return lambda dataset, rows, _columns: [str(label) for label in rows_of(dataset)[rows]]


def _drug_labels(dataset: Dataset, _rows: np.ndarray, columns: np.ndarray) -> list[str]:
    return [str(drug) for drug in np.asarray(dataset.drug_ids)[columns]]


def _sample_drug_labels(dataset: Dataset, rows: np.ndarray, columns: np.ndarray) -> list[str]:
    samples, drugs = _source_samples(dataset)[rows], np.asarray(dataset.drug_ids)[columns]
    return [f"({sample}, {drug})" for sample, drug in zip(samples, drugs, strict=True)]


MODES: dict[Validation, LeakageGroups] = {
    "LSO": LeakageGroups("samples", _sample_groups, _row_labels(_source_samples), cut="LSO"),
    "LDO": LeakageGroups("drugs", _drug_groups, _drug_labels, cut="LDO"),
    "LTO": LeakageGroups("tissues", _by_sorted_label(_tissues), _row_labels(_tissues), cut="LTO"),
    # A study holds many samples and drugs, so validation and early stopping are cut by
    # (source sample, drug): a sample measured in two training studies stays on one side.
    "LStO": LeakageGroups("studies", _by_sorted_label(_studies), _row_labels(_studies), cut="LPO"),
    "LPO": LeakageGroups("(sample, drug) pairs", _sample_drug_groups, _sample_drug_labels, cut="LPO"),
}
"""The leakage groups of every split mode."""
