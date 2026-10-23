"""Dataset registry: register, discover, and manage dataset sources and entries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._io import config_lock, get_config_path, load_config, save_config
from ._models import DatasetEntry, DrevalConfig, SourceEntry
from ._registry import DatasetRegistry, dataset_registry

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "DatasetEntry",
    "DatasetRegistry",
    "DrevalConfig",
    "SourceEntry",
    "config_lock",
    "dataset_registry",
    "get_config_path",
    "list",
    "load_config",
    "register_dataset",
    "register_source",
    "save_config",
    "table",
]


def register_source(name: str, base_url: str, storage_options: dict | None = None, *, override: bool = False) -> None:
    """Register a custom source (base URL + optional fsspec storage options).

    An identical repeat is a no-op; a different entry under a taken name raises ``ValueError``
    unless ``override=True``.
    """
    dataset_registry.register_source(name, base_url, storage_options, override=override)


def register_dataset(name: str, source: str, file: str, *, override: bool = False) -> None:
    """Register a custom dataset under an existing source.

    An identical repeat is a no-op; a different entry under a taken name raises ``ValueError``
    unless ``override=True``.
    """
    dataset_registry.register_dataset(name, source, file, override=override)


def list() -> list[str]:
    """Return sorted list of all registered dataset names."""
    return dataset_registry.dataset_names


def table() -> pd.DataFrame:
    """Return registry contents as a DataFrame."""
    return dataset_registry.to_dataframe()
