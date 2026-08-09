"""Dataset registry: Pydantic models, config I/O, and registration API.

The user config lives at ``<config_dir>/drevalpy.json`` (override with
``DREVALPY_CONFIG_DIR``). Sources and datasets registered there are merged
with the built-in registry at import time.

Usage::

    from drevalpy.data import registry

    registry  # displays Rich table via __repr__
    registry.dataset_names  # programmatic access
    registry.register_source("my_s3", "s3://bucket/data")
    registry.register_dataset("MyStudy", source="my_s3", file="MyStudy.h5mu")
"""

from .io import config_lock, get_config_path, load_config, save_config
from .load import load
from .models import DatasetEntry, DrevalConfig, SourceEntry
from .registry import Registry

registry = Registry()

__all__ = [
    "DatasetEntry",
    "DrevalConfig",
    "Registry",
    "SourceEntry",
    "config_lock",
    "get_config_path",
    "load_config",
    "load",
    "registry",
    "save_config",
]
