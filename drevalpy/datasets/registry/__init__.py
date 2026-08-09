"""Dataset registry: Pydantic models, config I/O, and registration API.

The user config lives at ``<config_dir>/drevalpy.json`` (override with
``DREVALPY_CONFIG_DIR``). Sources and datasets registered there are merged
with the built-in registry at import time.

Usage::

    from drevalpy.datasets import registry

    registry.list_datasets()  # pretty-print table
    registry.dataset_names  # programmatic access
    registry.register_source("my_s3", "s3://bucket/data")
    registry.register_dataset("MyStudy", source="my_s3", file="MyStudy.h5mu")
"""

from .io import config_lock, get_config_path, load_config, save_config
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
    "registry",
    "save_config",
]
