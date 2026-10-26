"""Data loading, splitting, and registries."""

from __future__ import annotations

import hashlib
from typing import Any

from drevalpy.types import SplitMasks
from drevalpy.types.data.dataset import Dataset as Dataset

from ._concat import concat
from ._synthetic import synthetic
from .datasets._load import load
from .quality import CurveQuality, curve_quality_mask


def split(
    dataset: Dataset,
    mode: str,
    n_splits: int = 5,
    validation_ratio: float = 0.1,
    random_state: int = 42,
    *,
    quality: CurveQuality | None = None,
) -> list[SplitMasks]:
    """Split a dataset using a registered splitter.

    :param dataset: Dataset instance.
    :param mode: Splitter mode or alias (e.g. "LSO", "LPO", "LDO", "LTO"). Metadata records the canonical mode.
    :param n_splits: Number of CV folds.
    :param validation_ratio: Fraction of training data for validation.
    :param random_state: Seed for reproducibility.
    :param quality: Curve-quality rule handed to the splitter as ``quality=``; ``None`` means
        ``CurveQuality.default()``. The built-in splitters leave out the pairs that fail it. The rule is
        recorded in each fold's metadata under ``quality`` as the one requested; a custom splitter may
        deliberately ignore it.
    :returns: List of SplitMasks, one per fold.
    :raises ValueError: If two folds produce the same fold ID.
    """
    from drevalpy.registry.splitter import splitter_registry

    rule = quality or CurveQuality.default()
    mode = splitter_registry.canonical(mode)
    splitter = splitter_registry.get(mode)
    folds = splitter(
        dataset, n_splits=n_splits, validation_ratio=validation_ratio, random_state=random_state, quality=rule
    )

    fold_ids: list[str] = []
    for i, fold in enumerate(folds):
        hasher = hashlib.sha256()
        hasher.update(fold.train.mask.tobytes())
        hasher.update(fold.test.mask.tobytes())
        hasher.update(fold.val.mask.tobytes())
        fold_id = hasher.hexdigest()[:12]
        fold_ids.append(fold_id)

        fold.metadata.setdefault("dataset", dataset.name)
        fold.metadata.setdefault("fold_index", i)
        fold.metadata.setdefault("fold_id", fold_id)
        fold.metadata["quality"] = rule.to_metadata()

    if len(set(fold_ids)) != len(fold_ids):
        raise ValueError(f"Duplicate fold_ids generated: {fold_ids}")

    return folds


def __getattr__(name: str) -> Any:
    """Lazy access to registry singletons to avoid circular imports."""
    if name == "dataset_registry":
        from drevalpy.registry.dataset import dataset_registry

        return dataset_registry
    if name == "splitter_registry":
        from drevalpy.registry.splitter import splitter_registry

        return splitter_registry
    raise AttributeError(f"module 'drevalpy.data' has no attribute {name!r}")


__all__ = [
    "CurveQuality",
    "concat",
    "curve_quality_mask",
    "dataset_registry",
    "load",
    "split",
    "splitter_registry",
    "synthetic",
]
