"""Data loading, splitting, and registries."""

from __future__ import annotations

from drevalpy.types import SplitMasks
from drevalpy.types.data.dataset import Dataset as Dataset

from .datasets import registry as dataset_registry
from .datasets.load import load
from .splitters import splitter_registry


def split(
    dataset: Dataset,
    mode: str,
    n_splits: int = 5,
    validation_ratio: float = 0.1,
    random_state: int = 42,
) -> list[SplitMasks]:
    """Split a dataset using a registered splitter.

    :param dataset: Dataset instance.
    :param mode: Splitter mode (e.g. "LCO", "LPO", "LDO", "LTO").
    :param n_splits: Number of CV folds.
    :param validation_ratio: Fraction of training data for validation.
    :param random_state: Seed for reproducibility.
    :returns: List of SplitMasks, one per fold.
    """
    splitter = splitter_registry.get(mode)
    folds = splitter(dataset, n_splits=n_splits, validation_ratio=validation_ratio, random_state=random_state)
    for fold in folds:
        fold.metadata.setdefault("dataset", dataset.name)
    return folds


__all__ = [
    "dataset_registry",
    "load",
    "split",
    "splitter_registry",
]
