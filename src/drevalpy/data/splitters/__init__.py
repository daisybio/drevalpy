"""Cross-validation splitting strategies for drug response prediction.

Splitters are plain callables with the signature::

    (dataset: Dataset, n_splits: int, validation_ratio: float, random_state: int,
     *, quality: CurveQuality) -> list[Fold]

A splitter calls :func:`observed_mask` with the ``quality`` it receives to get the usable pairs.

Built-in modes (LPO, LSO, LDO, LTO, LStO) are registered via decorator on import of their
modules. Register custom splitters with :func:`drevalpy.registry.splitter.register` and look one
up with :func:`drevalpy.registry.splitter.get`::

    @register("MY_MODE", "Description", validation="LSO")
    def my_splitter(dataset, n_splits=5, validation_ratio=0.1, random_state=42, *, quality):
        observed = observed_mask(dataset, quality)
        ...
"""

from ._folds import observed_mask

__all__ = ["observed_mask"]
