"""Helpers for splitter tests: the pairs a test marks as failing and the pairs the folds cover."""

from __future__ import annotations

import numpy as np

from tests._fake_dataset import FakeResponseDataset


def first_measured_pairs(dataset: FakeResponseDataset, count: int) -> list[tuple[int, int]]:
    """Return the first *count* measured ``(row, column)`` pairs of *dataset*."""
    measured = np.argwhere(~np.isnan(dataset.response_matrix))
    return [(int(row), int(column)) for row, column in measured[:count]]


def covered_pairs(folds: list) -> np.ndarray:
    """Union every train, test and validation mask across *folds*."""
    covered = np.zeros(folds[0].train.shape, dtype=bool)
    for fold in folds:
        covered |= fold.train.mask | fold.test.mask | fold.val.mask
    return covered
