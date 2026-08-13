"""Dataset registry: built-in + user-registered sources and datasets."""

from __future__ import annotations

import json
from importlib import resources
from typing import TYPE_CHECKING, Any

from ._io import config_lock, load_config, save_config
from ._models import DatasetEntry, DrevalConfig, SourceEntry

if TYPE_CHECKING:
    import pandas as pd

_REGISTRY_JSON = "available_datasets.json"


class DatasetRegistry:
    """Dataset registry merging built-in and user-registered datasets.

    Built-in and custom entries are stored separately. The combined view
    is a computed property so it always reflects the current state.
    Mutation methods use file locking for atomic read-modify-write.
    """

    def __init__(self) -> None:
        """Initialize with lazy loading of built-in and custom registries."""
        self._builtin: DrevalConfig | None = None
        self._custom: DrevalConfig | None = None

    def _ensure_loaded(self) -> None:
        """Load registries on first access."""
        if self._builtin is None:
            registry_path = resources.files("drevalpy.data.datasets").joinpath(_REGISTRY_JSON)
            with registry_path.open(encoding="utf-8") as handle:
                raw = json.load(handle)
            self._builtin = DrevalConfig.from_raw(raw)
        if self._custom is None:
            self._custom = load_config()

    @property
    def sources(self) -> dict[str, SourceEntry]:
        """All sources (custom overrides built-in)."""
        self._ensure_loaded()
        return {**self._builtin.sources, **self._custom.sources}  # type: ignore[union-attr]

    @property
    def datasets(self) -> dict[str, DatasetEntry]:
        """All datasets (custom overrides built-in)."""
        self._ensure_loaded()
        return {**self._builtin.datasets, **self._custom.datasets}  # type: ignore[union-attr]

    @property
    def builtin_sources(self) -> dict[str, SourceEntry]:
        """Only built-in sources (read-only)."""
        self._ensure_loaded()
        return self._builtin.sources  # type: ignore[union-attr]

    @property
    def builtin_datasets(self) -> dict[str, DatasetEntry]:
        """Only built-in datasets (read-only)."""
        self._ensure_loaded()
        return self._builtin.datasets  # type: ignore[union-attr]

    @property
    def custom_sources(self) -> dict[str, SourceEntry]:
        """Only user-registered sources."""
        self._ensure_loaded()
        return self._custom.sources  # type: ignore[union-attr]

    @property
    def custom_datasets(self) -> dict[str, DatasetEntry]:
        """Only user-registered datasets."""
        self._ensure_loaded()
        return self._custom.datasets  # type: ignore[union-attr]

    @property
    def dataset_names(self) -> list[str]:
        """Sorted list of all registered dataset names (built-in + custom)."""
        return sorted(self.datasets)

    @property
    def source_names(self) -> list[str]:
        """Sorted list of all registered source names (built-in + custom)."""
        return sorted(self.sources)

    def to_dataframe(self) -> pd.DataFrame:
        """Return registry contents as a pandas DataFrame."""
        import pandas as pd

        rows = []
        for name in sorted(self.datasets):
            entry = self.datasets[name]
            origin = "custom" if name in self._custom.datasets else "built-in"
            rows.append({"Name": name, "Source": entry.source, "File": entry.file, "Origin": origin})
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        """Return a tabular string representation."""
        return self.to_dataframe().to_string(index=False)

    def _repr_html_(self) -> str:
        """HTML table for Jupyter notebooks."""
        return self.to_dataframe().to_html(index=False)

    def is_registered(self, name: str) -> bool:
        """Return whether ``name`` is a registered dataset.

        :param name: Dataset name to look up.
        :returns: ``True`` when ``name`` is registered.
        """
        return name in self.datasets

    def register_source(self, name: str, base_url: str, storage_options: dict[str, Any] | None = None) -> None:
        """Register a custom source (base URL + optional fsspec storage options).

        Atomically reads the config, applies the change, and writes back.

        :param name: Source name (used to reference from dataset entries).
        :param base_url: Base URL (any fsspec-compatible protocol: https, s3, gs, az, ...).
        :param storage_options: Optional dict passed to fsspec for auth/config.
        """
        entry = SourceEntry(url=base_url, storage_options=storage_options or {})
        with config_lock():
            config = load_config()
            config.sources[name] = entry
            save_config(config)
            self._custom = config

    def register_dataset(self, name: str, source: str, file: str) -> None:
        """Register a custom dataset under an existing source.

        Atomically reads the config, applies the change, and writes back.

        :param name: Dataset name (used with ``drevalpy.data.load``).
        :param source: Source name (must be registered).
        :param file: Filename of the .h5mu file at the source URL.
        :raises KeyError: If the source is not registered.
        """
        if source not in self.sources:
            raise KeyError(f"Source '{source}' not registered. Register it first with register_source().")

        with config_lock():
            config = load_config()
            config.datasets[name] = DatasetEntry(source=source, file=file)
            save_config(config)
            self._custom = config

    def unregister_dataset(self, name: str) -> None:
        """Remove a custom dataset registration.

        :param name: Dataset name to remove.
        :raises KeyError: If the dataset is not in the custom registry.
        """
        self._ensure_loaded()
        with config_lock():
            config = load_config()
            if name not in config.datasets:
                raise KeyError(f"Dataset '{name}' not in custom registry.")
            if name in self._builtin.datasets:  # type: ignore[union-attr]
                raise KeyError(f"Dataset '{name}' is built-in and cannot be unregistered.")
            del config.datasets[name]
            save_config(config)
            self._custom = config

    def unregister_source(self, name: str) -> None:
        """Remove a custom source registration.

        :param name: Source name to remove.
        :raises KeyError: If the source is not in the custom registry.
        :raises ValueError: If datasets still reference this source.
        """
        self._ensure_loaded()
        with config_lock():
            config = load_config()
            if name not in config.sources:
                raise KeyError(f"Source '{name}' not in custom registry.")
            if name in self._builtin.sources:  # type: ignore[union-attr]
                raise KeyError(f"Source '{name}' is built-in and cannot be unregistered.")

            referencing = [ds for ds, entry in config.datasets.items() if entry.source == name]
            if referencing:
                raise ValueError(f"Cannot remove source '{name}': still referenced by datasets {referencing}")

            del config.sources[name]
            save_config(config)
            self._custom = config

    def reload(self) -> None:
        """Re-read the custom registry from disk.

        Useful after external modifications to the config file.
        """
        self._ensure_loaded()
        self._custom = load_config()


dataset_registry = DatasetRegistry()
