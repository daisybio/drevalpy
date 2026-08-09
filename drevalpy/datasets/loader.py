"""Load built-in and custom MuDatasets with protocol-agnostic downloads."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

import fsspec

from ._paths import get_default_data_dir, resolve_h5mu_path
from .config import DataConfig, DatasetEntry, SourceEntry, load_config, save_config
from .mudataset import MuDataset

_REGISTRY_JSON = "available_datasets.json"


# ------------------------------------------------------------------
# Registry loading and merging
# ------------------------------------------------------------------


def _load_builtin_registry() -> DataConfig:
    """Load the built-in sources and datasets from the packaged JSON."""
    registry_path = resources.files(__package__).joinpath(_REGISTRY_JSON)
    with registry_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return DataConfig.from_raw(raw)


def _build_merged_registry() -> DataConfig:
    """Merge built-in and custom registries (custom overrides built-in)."""
    builtin = _load_builtin_registry()
    custom = load_config().data
    return DataConfig(
        sources={**builtin.sources, **custom.sources},
        datasets={**builtin.datasets, **custom.datasets},
    )


_REGISTRY: DataConfig = _build_merged_registry()


# ------------------------------------------------------------------
# Source resolution and download
# ------------------------------------------------------------------


def _download_h5mu(name: str) -> Path:
    """Download the .h5mu file for a registered dataset.

    :param name: Dataset name from the registry.
    :returns: Local path to the downloaded file.
    """
    from rich.progress import DownloadColumn, Progress, TimeRemainingColumn, TransferSpeedColumn

    entry = _REGISTRY.datasets[name]
    source = _REGISTRY.sources[entry.source]
    remote_path = f"{source.url}/{entry.file}"
    local_path = get_default_data_dir() / entry.file
    local_path.parent.mkdir(parents=True, exist_ok=True)

    fs, _, paths = fsspec.get_fs_token_paths(remote_path, storage_options=source.storage_options)
    size = fs.size(paths[0])

    with Progress(
        *Progress.get_default_columns(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(f"Downloading {name}", total=size)
        with fs.open(paths[0], "rb") as remote, open(local_path, "wb") as local:
            while chunk := remote.read(1024 * 64):
                local.write(chunk)
                progress.advance(task, len(chunk))

    return local_path


# ------------------------------------------------------------------
# Public API: listing and loading
# ------------------------------------------------------------------


def list_builtin_datasets() -> list[str]:
    """List all registered dataset names (built-in + custom).

    :returns: Sorted dataset names available for ``load_mudataset``.
    """
    return sorted(_REGISTRY.datasets)


def is_builtin_dataset(name: str) -> bool:
    """Return whether ``name`` is a registered dataset.

    :param name: Dataset name to look up in the registry.
    :returns: ``True`` when ``name`` is registered.
    """
    return name in _REGISTRY.datasets


def load_mudataset(dataset_name: str) -> MuDataset:
    """Load a registered or custom dataset as a MuDataset from its .h5mu file.

    Resolution order:

    1. If the .h5mu exists at the standard cache path, load it directly.
    2. If *dataset_name* is registered, download the .h5mu if needed and load.
    3. If *dataset_name* is a path to an existing .h5mu file, load it directly.

    :param dataset_name: Registered dataset name, or path to a .h5mu file.
    :returns: Loaded MuDataset.
    :raises FileNotFoundError: If the .h5mu file cannot be found or downloaded.
    """
    h5mu_path = resolve_h5mu_path(dataset_name)
    if h5mu_path.is_file():
        return MuDataset.from_file(h5mu_path)

    if dataset_name in _REGISTRY.datasets:
        entry = _REGISTRY.datasets[dataset_name]
        data_dir = get_default_data_dir()
        candidate = data_dir / entry.file
        if candidate.is_file():
            return MuDataset.from_file(candidate)
        downloaded = _download_h5mu(dataset_name)
        return MuDataset.from_file(downloaded)

    candidate_path = Path(dataset_name)
    if candidate_path.is_file() and candidate_path.suffix == ".h5mu":
        return MuDataset.from_file(candidate_path)

    raise FileNotFoundError(
        f"Cannot locate .h5mu for dataset '{dataset_name}'. "
        f"Checked: {h5mu_path}, registry ({list(_REGISTRY.datasets.keys())}), and direct path."
    )


# ------------------------------------------------------------------
# Public API: registration
# ------------------------------------------------------------------


def register_source(name: str, base_url: str, storage_options: dict[str, Any] | None = None) -> None:
    """Register a custom source (base URL + optional fsspec storage options).

    Persists to the user config file.

    :param name: Source name (used to reference from dataset entries).
    :param base_url: Base URL (any fsspec-compatible protocol: https, s3, gs, az, ...).
    :param storage_options: Optional dict passed to fsspec for auth/config.
    """
    global _REGISTRY
    config = load_config()
    entry = SourceEntry(url=base_url, storage_options=storage_options or {})
    config.data.sources[name] = entry
    save_config(config)
    _REGISTRY.sources[name] = entry


def register_dataset(name: str, source: str, file: str) -> None:
    """Register a custom dataset under an existing source.

    Persists to the user config file.

    :param name: Dataset name (used with ``load_mudataset``).
    :param source: Source name (must be registered).
    :param file: Filename of the .h5mu file at the source URL.
    :raises KeyError: If the source is not registered.
    """
    global _REGISTRY
    if source not in _REGISTRY.sources:
        raise KeyError(f"Source '{source}' not registered. Register it first with register_source().")

    config = load_config()
    entry = DatasetEntry(source=source, file=file)
    config.data.datasets[name] = entry
    save_config(config)
    _REGISTRY.datasets[name] = entry


def unregister_dataset(name: str) -> None:
    """Remove a custom dataset registration.

    :param name: Dataset name to remove.
    :raises KeyError: If the dataset is not in the custom registry.
    """
    global _REGISTRY
    config = load_config()
    if name not in config.data.datasets:
        raise KeyError(f"Dataset '{name}' not in custom registry.")

    del config.data.datasets[name]
    save_config(config)
    _REGISTRY.datasets.pop(name, None)


def unregister_source(name: str) -> None:
    """Remove a custom source registration.

    :param name: Source name to remove.
    :raises KeyError: If the source is not in the custom registry.
    :raises ValueError: If datasets still reference this source.
    """
    global _REGISTRY
    config = load_config()
    if name not in config.data.sources:
        raise KeyError(f"Source '{name}' not in custom registry.")

    referencing = [ds for ds, entry in config.data.datasets.items() if entry.source == name]
    if referencing:
        raise ValueError(f"Cannot remove source '{name}': still referenced by datasets {referencing}")

    del config.data.sources[name]
    save_config(config)
    _REGISTRY.sources.pop(name, None)
