"""Top-level ``split`` entry point: resolve a registered splitter and stamp fold metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

from drevalpy.data.quality import CurveQuality
from drevalpy.types import Fold

if TYPE_CHECKING:
    from drevalpy.types.data.dataset import Dataset


def split(
    dataset: Dataset,
    mode: str,
    n_splits: int = 5,
    validation_ratio: float = 0.1,
    random_state: int = 42,
    *,
    quality: CurveQuality | None = None,
    measure: str | None = None,
) -> list[Fold]:
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
    :param measure: Response measure to split on, see :meth:`Dataset.with_measure`; ``None`` keeps the
        dataset's. Only pairs that have it are split. It is recorded in each fold's metadata under
        ``measure``, and a run on the fold must use the same measure.
    :returns: List of Fold, one per fold.
    :raises ValueError: If two folds produce the same fold ID.
    """
    from drevalpy.registry import splitter as splitters

    if measure is not None:
        dataset = dataset.with_measure(measure)
    rule = quality or CurveQuality.default()
    splitter = splitters.get(mode)
    folds = splitter(
        dataset, n_splits=n_splits, validation_ratio=validation_ratio, random_state=random_state, quality=rule
    )

    fold_ids = [fold.fold_id for fold in folds]
    for fold, fold_id in zip(folds, fold_ids, strict=True):
        fold.metadata.setdefault("dataset", dataset.name)
        fold.metadata.setdefault("fold_id", fold_id)
        fold.metadata["quality"] = rule.to_metadata()
        fold.metadata["measure"] = dataset.measure

    if len(set(fold_ids)) != len(fold_ids):
        raise ValueError(f"Duplicate fold_ids generated: {fold_ids}")

    return folds
