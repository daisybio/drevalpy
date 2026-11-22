"""Dataset registry: register, discover, and manage dataset sources and entries."""

from __future__ import annotations

from ._registry import dataset_registry

__all__ = ["dataset_registry", "register_dataset", "register_source"]


def register_source(name: str, base_url: str, storage_options: dict | None = None, *, replace: bool = False) -> None:
    """Register a custom source (base URL + optional fsspec storage options).

    An identical repeat is a no-op; a different entry under a taken name raises ``ValueError``
    unless ``replace=True``.
    """
    dataset_registry.register_source(name, base_url, storage_options, replace=replace)


def register_dataset(name: str, source: str, file: str, *, replace: bool = False) -> None:
    """Register a custom dataset under an existing source.

    An identical repeat is a no-op; a different entry under a taken name raises ``ValueError``
    unless ``replace=True``.
    """
    dataset_registry.register(name, source, file, replace=replace)
