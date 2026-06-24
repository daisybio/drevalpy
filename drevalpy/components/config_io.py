"""Compatibility re-export — implementation lives in `drevalpy.models.config_io`."""

from drevalpy.models.config_io import (
    model_config_from_dict,
    model_config_from_spec,
    model_config_from_yaml,
)

__all__ = [
    "model_config_from_dict",
    "model_config_from_spec",
    "model_config_from_yaml",
]
