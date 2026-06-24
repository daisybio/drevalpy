"""Compatibility re-export for moved DIPK data utilities."""

from drevalpy.components.predictors.literature.impl.dipk.data_utils import (
    CollateFn,
    DIPKDataset,
    get_data,
    load_bionic_features,
)

__all__ = ["CollateFn", "DIPKDataset", "get_data", "load_bionic_features"]
