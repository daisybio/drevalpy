"""Core data structures for drevalpy."""

from .dataset import Dataset
from .mudatalike import MuDataLike
from .response_batch import ResponseBatch
from .split_mask import SplitMask
from .split_masks import SplitMasks
from .view_location import ViewLocation

EntityScope = SplitMask

__all__ = [
    "Dataset",
    "EntityScope",
    "MuDataLike",
    "ResponseBatch",
    "SplitMask",
    "SplitMasks",
    "ViewLocation",
]
