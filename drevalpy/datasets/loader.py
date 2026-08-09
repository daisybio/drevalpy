"""Load built-in and custom MuDatasets with protocol-agnostic downloads."""

from __future__ import annotations

import json
import os
from importlib import resources
from pathlib import Path
from typing import Any

import fsspec
from platformdirs import user_config_dir

from ._paths import get_default_data_dir, resolve_h5mu_path
from .mudataset import MuDataset

_REGISTRY_JSON = "available_datasets.json"


# ------------------------------------------------------------------
# User config (persistent, extensible)
# ------------------------------------------------------------------


def _get_config_path() -> Path:
    """Return path to the user config file.

    Checks ``DREVALPY_CONFIG_DIR`` env var first, falls back to platformdirs.
    """
    env = os.environ.get("DREVALPY_CONFIG_DIR", "").strip()
    config_dir = Path(env) if env else Path(user_config_dir("drevalpy"))
    return config_dir / "drevalpy.json"


def _load_user_config() -> dict[str, Any]:
    """Read the full user config, returning {} if it doesn't exist."""
    path = _get_config_path()
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_user_config(config: dict[str, Any]) -> None:
    """Write the user config to disk, creating the directory if needed."""
    path = _get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


# ------------------------------------------------------------------
# Registry loading and merging
# ------------------------------------------------------------------


def _load_builtin_registry() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the built-in sources and datasets from the packaged JSON."""
    registry_path = resources.files(__package__).joinpath(_REGISTRY_JSON)
    with registry_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return raw.get("sources", {}), raw.get("datasets", {})


def _load_custom_registry() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load user-registered sources and datasets from the config file."""
    config = _load_user_config()
    data = config.get("data", {})
    return data.get("sources", {}), data.get("datasets", {})


def _build_merged_registry() -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge built-in and custom registries (custom overrides built-in)."""
    builtin_sources, builtin_datasets = _load_builtin_registry()
    custom_sources, custom_datasets = _load_custom_registry()
    return {**builtin_sources, **custom_sources}, {**builtin_datasets, **custom_datasets}


_SOURCES: dict[str, Any]
_DATASETS: dict[str, Any]
_SOURCES, _DATASETS = _build_merged_registry()


# ------------------------------------------------------------------
# Source resolution and download
# ------------------------------------------------------------------


def _resolve_source(name: str) -> tuple[str, dict[str, Any]]:
    """Return (base_url, storage_options) for a source name.

    :param name: Source name from the registry.
    :returns: Tuple of base URL and fsspec storage options.
    :raises KeyError: If the source is not registered.
    """
    if name not in _SOURCES:
        raise KeyError(f"Source '{name}' not registered. Available: {sorted(_SOURCES)}")
    raw = _SOURCES[name]
    if isinstance(raw, str):
        return raw, {}
    return raw["url"], raw.get("storage_options", {})


def _download_h5mu(name: str) -> Path:
    """Download the .h5mu file for a registered dataset.

    :param name: Dataset name from the registry.
    :returns: Local path to the downloaded file.
    """
    from rich.progress import DownloadColumn, Progress, TimeRemainingColumn, TransferSpeedColumn

    entry = _DATASETS[name]
    base_url, storage_options = _resolve_source(entry["source"])
    remote_path = f"{base_url}/{entry['file']}"
    local_path = get_default_data_dir() / entry["file"]
    local_path.parent.mkdir(parents=True, exist_ok=True)

    fs, _, paths = fsspec.get_fs_token_paths(remote_path, storage_options=storage_options)
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
    return sorted(_DATASETS)


def is_builtin_dataset(name: str) -> bool:
    """Return whether ``name`` is a registered dataset.

    :param name: Dataset name to look up in the registry.
    :returns: ``True`` when ``name`` is registered.
    """
    return name in _DATASETS


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

    if dataset_name in _DATASETS:
        data_dir = get_default_data_dir()
        candidate = data_dir / _DATASETS[dataset_name]["file"]
        if candidate.is_file():
            return MuDataset.from_file(candidate)
        downloaded = _download_h5mu(dataset_name)
        return MuDataset.from_file(downloaded)

    candidate_path = Path(dataset_name)
    if candidate_path.is_file() and candidate_path.suffix == ".h5mu":
        return MuDataset.from_file(candidate_path)

    raise FileNotFoundError(
        f"Cannot locate .h5mu for dataset '{dataset_name}'. "
        f"Checked: {h5mu_path}, registry ({list(_DATASETS.keys())}), and direct path."
    )


# ------------------------------------------------------------------
# Public API: registration
# ------------------------------------------------------------------


def register_source(name: str, base_url: str, storage_options: dict[str, Any] | None = None) -> None:
    """Register a custom source (base URL + optional fsspec storage options).

    Persists to the user config file at ``<config_dir>/drevalpy.json``.

    :param name: Source name (used to reference from dataset entries).
    :param base_url: Base URL (any fsspec-compatible protocol: https, s3, gs, az, ...).
    :param storage_options: Optional dict passed to fsspec for auth/config.
    """
    global _SOURCES
    config = _load_user_config()
    data = config.setdefault("data", {})
    sources = data.setdefault("sources", {})

    if storage_options:
        sources[name] = {"url": base_url, "storage_options": storage_options}
    else:
        sources[name] = base_url

    _save_user_config(config)
    _SOURCES[name] = sources[name]


def register_dataset(name: str, source: str, file: str) -> None:
    """Register a custom dataset under an existing source.

    Persists to the user config file at ``<config_dir>/drevalpy.json``.

    :param name: Dataset name (used with ``load_mudataset``).
    :param source: Source name (must be registered).
    :param file: Filename of the .h5mu file at the source URL.
    :raises KeyError: If the source is not registered.
    """
    global _DATASETS
    if source not in _SOURCES:
        raise KeyError(f"Source '{source}' not registered. Register it first with register_source().")

    config = _load_user_config()
    data = config.setdefault("data", {})
    datasets = data.setdefault("datasets", {})
    datasets[name] = {"source": source, "file": file}

    _save_user_config(config)
    _DATASETS[name] = {"source": source, "file": file}


def unregister_dataset(name: str) -> None:
    """Remove a custom dataset registration.

    :param name: Dataset name to remove.
    :raises KeyError: If the dataset is not in the custom registry.
    """
    global _DATASETS
    config = _load_user_config()
    datasets = config.get("data", {}).get("datasets", {})
    if name not in datasets:
        raise KeyError(f"Dataset '{name}' not in custom registry.")

    del datasets[name]
    _save_user_config(config)
    _DATASETS.pop(name, None)


def unregister_source(name: str) -> None:
    """Remove a custom source registration.

    :param name: Source name to remove.
    :raises KeyError: If the source is not in the custom registry.
    :raises ValueError: If datasets still reference this source.
    """
    global _SOURCES
    config = _load_user_config()
    sources = config.get("data", {}).get("sources", {})
    if name not in sources:
        raise KeyError(f"Source '{name}' not in custom registry.")

    datasets = config.get("data", {}).get("datasets", {})
    referencing = [ds for ds, entry in datasets.items() if entry.get("source") == name]
    if referencing:
        raise ValueError(f"Cannot remove source '{name}': still referenced by datasets {referencing}")

    del sources[name]
    _save_user_config(config)
    _SOURCES.pop(name, None)
