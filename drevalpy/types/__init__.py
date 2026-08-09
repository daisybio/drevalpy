"""Shared types and data structures for drevalpy."""

from drevalpy.types.dataset import Dataset
from drevalpy.types.literature_reference import LiteratureReference
from drevalpy.types.model_scope import ModelScope
from drevalpy.types.mudatalike import MuDataLike
from drevalpy.types.prediction_mode import PredictionMode
from drevalpy.types.response_batch import ResponseBatch
from drevalpy.types.split_mask import SplitMask
from drevalpy.types.split_masks import SplitMasks
from drevalpy.types.view_location import ViewLocation

EntityScope = SplitMask

__all__ = [
    "Dataset",
    "EntityScope",
    "LiteratureReference",
    "ModelScope",
    "MuDataLike",
    "PredictionMode",
    "ResponseBatch",
    "SplitMask",
    "SplitMasks",
    "ViewLocation",
]
