"""Cross-validation splitting strategies for drug response prediction.

Splitters are plain callables with the signature::

    (dataset: Dataset, n_splits: int, validation_ratio: float, random_state: int,
     *, quality: CurveQuality) -> list[SplitMasks]

A splitter calls :func:`observed_mask` with the ``quality`` it receives to get the usable pairs.

Built-in modes (LPO, LSO, LDO, LTO, LStO) are registered via decorator on import.
Register custom splitters with :func:`drevalpy.registry.splitter.register`::

    @register("MY_MODE", "Description", validation="LSO")
    def my_splitter(dataset, n_splits=5, validation_ratio=0.1, random_state=42, *, quality):
        observed = observed_mask(dataset, quality)
        ...
"""

from drevalpy.registry.splitter import (
    Splitter,
    SplitValidationError,
    Validation,
    splitter_registry,
)
from drevalpy.registry.splitter import get as get_splitter

from ._folds import observed_mask as observed_mask
from .ldo import leave_drug_out as leave_drug_out
from .lpo import leave_pair_out as leave_pair_out
from .lso import leave_sample_out as leave_sample_out
from .lsto import leave_study_out as leave_study_out
from .lto import leave_tissue_out as leave_tissue_out

__all__ = [
    "SplitValidationError",
    "Splitter",
    "Validation",
    "get_splitter",
    "leave_drug_out",
    "leave_pair_out",
    "leave_sample_out",
    "leave_study_out",
    "leave_tissue_out",
    "observed_mask",
    "splitter_registry",
]
