"""Dataset registry: register, discover, and manage dataset sources and entries."""

from ._io import config_lock, get_config_path, load_config, save_config
from ._models import DatasetEntry, DrevalConfig, SourceEntry
from ._registry import DatasetRegistry, dataset_registry

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


def register_source(name: str, base_url: str, storage_options: dict | None = None) -> None:
    """Register a custom source (base URL + optional fsspec storage options)."""
    dataset_registry.register_source(name, base_url, storage_options)


def register_dataset(name: str, source: str, file: str) -> None:
    """Register a custom dataset under an existing source."""
    dataset_registry.register_dataset(name, source, file)


def list() -> list[str]:  # noqa: A001
    """Return sorted list of all registered dataset names."""
    return dataset_registry.dataset_names


def table():
    """Return registry contents as a DataFrame."""
    return dataset_registry.to_dataframe()
