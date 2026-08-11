"""Data loading, splitting, and registries."""

from __future__ import annotations

import hashlib

from drevalpy.types import SplitMasks
from drevalpy.types.data.dataset import Dataset as Dataset

from .datasets.load import load


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
    from drevalpy.registry.splitter import splitter_registry

    splitter = splitter_registry.get(mode)
    folds = splitter(dataset, n_splits=n_splits, validation_ratio=validation_ratio, random_state=random_state)

    fold_ids: list[str] = []
    for i, fold in enumerate(folds):
        hasher = hashlib.sha256()
        hasher.update(fold.train.mask.tobytes())
        hasher.update(fold.test.mask.tobytes())
        hasher.update(fold.val.mask.tobytes())
        fold_id = hasher.hexdigest()[:12]
        fold_ids.append(fold_id)

        fold.metadata.setdefault("dataset", dataset.name)
        fold.metadata.setdefault("split_mode", mode)
        fold.metadata.setdefault("fold_index", i)
        fold.metadata.setdefault("fold_id", fold_id)

    if len(set(fold_ids)) != len(fold_ids):
        raise ValueError(f"Duplicate fold_ids generated: {fold_ids}")

    return folds


def __getattr__(name: str):
    """Lazy access to registry singletons to avoid circular imports."""
    if name == "dataset_registry":
        from drevalpy.registry.dataset import dataset_registry

        return dataset_registry
    if name == "splitter_registry":
        from drevalpy.registry.splitter import splitter_registry

        return splitter_registry
    raise AttributeError(f"module 'drevalpy.data' has no attribute {name!r}")


__all__ = [
    "dataset_registry",
    "load",
    "split",
    "splitter_registry",
]
