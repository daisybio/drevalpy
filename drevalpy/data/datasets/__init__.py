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

from __future__ import annotations

from .load import load


def __getattr__(name: str):
    """Lazy imports from drevalpy.registry.dataset to avoid circular imports."""
    from drevalpy.registry.dataset import (
        DatasetEntry,
        DatasetRegistry,
        DrevalConfig,
        SourceEntry,
        config_lock,
        dataset_registry,
        get_config_path,
        load_config,
        register_dataset,
        register_source,
        save_config,
    )

    _lazy = {
        "DatasetEntry": DatasetEntry,
        "DatasetRegistry": DatasetRegistry,
        "DrevalConfig": DrevalConfig,
        "SourceEntry": SourceEntry,
        "config_lock": config_lock,
        "dataset_registry": dataset_registry,
        "get_config_path": get_config_path,
        "load_config": load_config,
        "register_dataset": register_dataset,
        "register_source": register_source,
        "registry": dataset_registry,
        "save_config": save_config,
    }
    if name in _lazy:
        return _lazy[name]
    raise AttributeError(f"module 'drevalpy.data.datasets' has no attribute {name!r}")


__all__ = [
    "DatasetEntry",
    "DatasetRegistry",
    "DrevalConfig",
    "SourceEntry",
    "config_lock",
    "dataset_registry",
    "get_config_path",
    "load",
    "load_config",
    "register_dataset",
    "register_source",
    "registry",
    "save_config",
]
