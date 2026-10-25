"""Shared value types: the dataset, split masks and results.

The names below are the ones user code imports; every other type is imported from its
defining module.
"""

from .data.dataset import Dataset
from .data.split_mask import SplitMask
from .data.split_masks import SplitMasks

__all__ = ["Dataset", "SplitMask", "SplitMasks"]
