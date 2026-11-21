"""Shared value types: the dataset, folds and selection masks, and results.

The names below are the ones user code imports; every other type is imported from its
defining module.
"""

from .data.dataset import Dataset
from .data.fold import Fold
from .data.selection_mask import SelectionMask

__all__ = ["Dataset", "Fold", "SelectionMask"]
