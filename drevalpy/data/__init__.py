"""Data loading, splitting, and registries."""

from __future__ import annotations

from .datasets import registry as dataset_registry
from .datasets.load import load
from .splitters import splitter_registry
from .structures import SplitMasks


def split(
    dataset: str | Dataset,
    mode: str,
    n_splits: int = 5,
    validation_ratio: float = 0.1,
    random_state: int = 42,
) -> list[SplitMasks]:
    """Split a dataset using a registered splitter.

    :param dataset: Dataset instance, registered dataset name, or path to .h5mu file.
    :param mode: Splitter mode (e.g. "LCO", "LPO", "LDO", "LTO").
    :param n_splits: Number of CV folds.
    :param validation_ratio: Fraction of training data for validation.
    :param random_state: Seed for reproducibility.
    :returns: List of SplitMasks, one per fold.
    """
    from .structures.dataset import Dataset as _MuDataset

    mudataset = dataset if isinstance(dataset, _MuDataset) else load(dataset)
    splitter = splitter_registry.get(mode)
    folds = splitter(mudataset, n_splits=n_splits, validation_ratio=validation_ratio, random_state=random_state)
    for fold in folds:
        fold.metadata.setdefault("dataset", mudataset.name)
    return folds


# Lazy import to avoid circular ref
from .structures.dataset import Dataset as Dataset  # noqa: E402

__all__ = [
    "dataset_registry",
    "load",
    "load",
    "split",
    "splitter_registry",
]
