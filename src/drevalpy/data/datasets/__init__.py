"""Lazy re-export of the dataset registry and :func:`load`.

The registry classes, the config I/O functions and the ``registry`` singleton are defined in
:mod:`drevalpy.registry.dataset` and resolved on first attribute access, which avoids a circular
import. The user config lives at ``<config_dir>/drevalpy.json`` (override with
``DREVALPY_CONFIG_DIR``).

Usage::

    from drevalpy.data.datasets import registry

    registry  # displays Rich table via __repr__
    registry.dataset_names  # programmatic access
    registry.register_source("my_s3", "s3://bucket/data")
    registry.register_dataset("MyStudy", source="my_s3", file="MyStudy.h5mu")
"""

from __future__ import annotations

from typing import Any

from ._load import load


def __getattr__(name: str) -> Any:
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
