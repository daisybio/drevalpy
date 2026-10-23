"""Dataset registry: built-in + user-registered sources and datasets."""

from __future__ import annotations

import json
from collections.abc import Callable
from importlib import resources
from typing import TYPE_CHECKING, Any, Literal

from drevalpy.errors import UnknownNameError
from drevalpy.registry._base import TableReprMixin

from ._io import config_lock, load_config, save_config
from ._models import DatasetEntry, DrevalConfig, SourceEntry

if TYPE_CHECKING:
    import pandas as pd

_REGISTRY_JSON = "available_datasets.json"

_Kind = Literal["sources", "datasets"]
_LABELS: dict[_Kind, str] = {"sources": "Source", "datasets": "Dataset"}


class DatasetRegistry(TableReprMixin):
    """Dataset registry merging built-in and user-registered datasets.

    Built-in and custom entries are stored separately. The combined view
    is a computed property so it always reflects the current state.
    Mutation methods use file locking for atomic read-modify-write.
    """

    def __init__(self) -> None:
        """Initialize with lazy loading of built-in and custom registries."""
        self._builtin: DrevalConfig | None = None
        self._custom: DrevalConfig | None = None

    def _ensure_loaded(self) -> tuple[DrevalConfig, DrevalConfig]:
        """Load the built-in and custom registries on first access.

        :returns: The built-in config and the custom config, in that order.
        """
        if self._builtin is None:
            registry_path = resources.files("drevalpy.data.datasets").joinpath(_REGISTRY_JSON)
            with registry_path.open(encoding="utf-8") as handle:
                self._builtin = DrevalConfig.from_raw(json.load(handle))
        if self._custom is None:
            self._custom = load_config()
        return self._builtin, self._custom

    @property
    def sources(self) -> dict[str, SourceEntry]:
        """All sources (custom overrides built-in)."""
        builtin, custom = self._ensure_loaded()
        return {**builtin.sources, **custom.sources}

    @property
    def datasets(self) -> dict[str, DatasetEntry]:
        """All datasets (custom overrides built-in)."""
        builtin, custom = self._ensure_loaded()
        return {**builtin.datasets, **custom.datasets}

    @property
    def builtin_sources(self) -> dict[str, SourceEntry]:
        """Only built-in sources (read-only)."""
        return self._ensure_loaded()[0].sources

    @property
    def builtin_datasets(self) -> dict[str, DatasetEntry]:
        """Only built-in datasets (read-only)."""
        return self._ensure_loaded()[0].datasets

    @property
    def custom_sources(self) -> dict[str, SourceEntry]:
        """Only user-registered sources."""
        return self._ensure_loaded()[1].sources

    @property
    def custom_datasets(self) -> dict[str, DatasetEntry]:
        """Only user-registered datasets."""
        return self._ensure_loaded()[1].datasets

    @property
    def dataset_names(self) -> list[str]:
        """Sorted list of all registered dataset names (built-in + custom)."""
        return sorted(self.datasets)

    @property
    def source_names(self) -> list[str]:
        """Sorted list of all registered source names (built-in + custom)."""
        return sorted(self.sources)

    def to_dataframe(self) -> pd.DataFrame:
        """Return registry contents as a pandas DataFrame.

        :returns: One row per dataset, with Name, Source, File and Origin columns.
        """
        import pandas as pd

        datasets = self.datasets
        custom = self.custom_datasets
        rows = []
        for name in sorted(datasets):
            entry = datasets[name]
            origin = "custom" if name in custom else "built-in"
            rows.append({"Name": name, "Source": entry.source, "File": entry.file, "Origin": origin})
        return pd.DataFrame(rows)

    def is_registered(self, name: str) -> bool:
        """Return whether ``name`` is a registered dataset.

        :param name: Dataset name to look up.
        :returns: ``True`` when ``name`` is registered.
        """
        return name in self.datasets

    def _register(self, kind: _Kind, name: str, entry: SourceEntry | DatasetEntry, *, override: bool) -> None:
        """Add one entry to the custom config under the file lock.

        Atomically reads the config, applies the change, and writes back. An entry identical to the one
        already in effect (built-in or custom) is a no-op and writes nothing.

        :param kind: Which table of the config to change.
        :param name: Entry name.
        :param entry: Entry to store under *name*.
        :param override: Replace an existing entry of the same name instead of raising.
        :raises ValueError: If *name* is already registered with a different entry and *override* is false.
        """
        builtin, _ = self._ensure_loaded()
        with config_lock():
            config = load_config()
            existing = {**getattr(builtin, kind), **getattr(config, kind)}.get(name)
            if existing == entry:
                self._custom = config
                return
            if existing is not None and not override:
                raise ValueError(f"{_LABELS[kind]} {name!r} already registered")
            getattr(config, kind)[name] = entry
            save_config(config)
            self._custom = config

    def _unregister(
        self, kind: _Kind, name: str, *, before_delete: Callable[[DrevalConfig], None] | None = None
    ) -> None:
        """Remove one entry from the custom config under the file lock.

        :param kind: Which table of the config to change.
        :param name: Entry name to remove.
        :param before_delete: Check run on the freshly loaded config that may raise to veto the removal.
        :raises UnknownNameError: If *name* is not a custom entry, or is a built-in one.
        """
        builtin, _ = self._ensure_loaded()
        label = _LABELS[kind]
        with config_lock():
            config = load_config()
            entries = getattr(config, kind)
            if name not in entries:
                raise UnknownNameError(f"{label} '{name}' not in custom registry.")
            if name in getattr(builtin, kind):
                raise UnknownNameError(f"{label} '{name}' is built-in and cannot be unregistered.")
            if before_delete is not None:
                before_delete(config)
            del entries[name]
            save_config(config)
            self._custom = config

    def register_source(
        self,
        name: str,
        base_url: str,
        storage_options: dict[str, Any] | None = None,
        *,
        override: bool = False,
    ) -> None:
        """Register a custom source (base URL + optional fsspec storage options).

        Registering an entry identical to the one already in effect (built-in or custom) is a no-op
        and writes nothing.

        :param name: Source name (used to reference from dataset entries).
        :param base_url: Base URL (any fsspec-compatible protocol: https, s3, gs, az, ...).
        :param storage_options: Optional dict passed to fsspec for auth/config.
        :param override: Replace an existing source of the same name instead of raising.
        :raises ValueError: If *name* is already registered with a different entry and *override* is false.
        """
        entry = SourceEntry(url=base_url, storage_options=storage_options or {})
        self._register("sources", name, entry, override=override)

    def register_dataset(self, name: str, source: str, file: str, *, override: bool = False) -> None:
        """Register a custom dataset under an existing source.

        Registering an entry identical to the one already in effect (built-in or custom) is a no-op
        and writes nothing.

        :param name: Dataset name (used with ``drevalpy.data.load``).
        :param source: Source name (must be registered).
        :param file: Filename of the .h5mu file at the source URL.
        :param override: Replace an existing dataset of the same name instead of raising.
        :raises UnknownNameError: If the source is not registered.
        :raises ValueError: If *name* is already registered with a different entry and *override* is false.
        """
        if source not in self.sources:
            raise UnknownNameError(f"Source '{source}' not registered. Register it first with register_source().")
        self._register("datasets", name, DatasetEntry(source=source, file=file), override=override)

    def unregister_dataset(self, name: str) -> None:
        """Remove a custom dataset registration.

        :param name: Dataset name to remove.
        :raises UnknownNameError: If the dataset is not in the custom registry.
        """
        self._unregister("datasets", name)

    def unregister_source(self, name: str) -> None:
        """Remove a custom source registration.

        :param name: Source name to remove.
        :raises UnknownNameError: If the source is not in the custom registry.
        :raises ValueError: If datasets still reference this source.
        """

        def reject_referenced(config: DrevalConfig) -> None:
            referencing = [ds for ds, entry in config.datasets.items() if entry.source == name]
            if referencing:
                raise ValueError(f"Cannot remove source '{name}': still referenced by datasets {referencing}")

        self._unregister("sources", name, before_delete=reject_referenced)

    def reload(self) -> None:
        """Re-read the custom registry from disk.

        Useful after external modifications to the config file.
        """
        self._ensure_loaded()
        self._custom = load_config()


dataset_registry = DatasetRegistry()
