"""Shared value types: the dataset, folds and pair masks, and results.

The names below are the ones user code imports; every other type is imported from its
defining module.
"""

from .data.dataset import Dataset
from .data.fold import Fold
from .data.pair_mask import PairMask

__all__ = ["Dataset", "Fold", "PairMask"]
