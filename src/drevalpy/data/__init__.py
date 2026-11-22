"""Data loading, splitting, concatenation and the synthetic dataset."""

from __future__ import annotations

from drevalpy.types.data.dataset import Dataset as Dataset

from ._concat import concat
from ._synthetic import synthetic
from .datasets._load import load
from .quality import CurveQuality
from .splitters._split import split

__all__ = [
    "CurveQuality",
    "concat",
    "load",
    "split",
    "synthetic",
]
