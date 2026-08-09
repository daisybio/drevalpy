"""Pydantic-based user configuration for drevalpy datasets.

The user config lives at ``<config_dir>/drevalpy.json`` (override with
``DREVALPY_CONFIG_DIR``) and is extensible -- additional top-level keys
can be added in the future without breaking existing configs.
"""

from .io import get_config_path, load_config, save_config
from .models import DataConfig, DatasetEntry, DrevalConfig, SourceEntry

__all__ = [
    "DataConfig",
    "DatasetEntry",
    "DrevalConfig",
    "SourceEntry",
    "get_config_path",
    "load_config",
    "save_config",
]
