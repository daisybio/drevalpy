"""Load built-in and custom MuDatasets."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import requests

from ._paths import get_default_data_dir, resolve_h5mu_path
from .mudataset import MuDataset

_REGISTRY_JSON = "available_datasets.json"


def _load_registry() -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Load sources and dataset entries from the packaged registry.

    :returns: Tuple of (sources mapping name->base_url, datasets mapping name->entry).
    """
    registry_path = resources.files(__package__).joinpath(_REGISTRY_JSON)
    with registry_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return raw["sources"], raw["datasets"]


_SOURCES, _DATASETS = _load_registry()


def list_builtin_datasets() -> list[str]:
    """List built-in dataset names from the packaged registry.

    :returns: Sorted dataset names registered for ``load_mudataset``.
    """
    return sorted(_DATASETS)


def is_builtin_dataset(name: str) -> bool:
    """Return whether ``name`` is a built-in dataset.

    :param name: Dataset name to look up in the registry.
    :returns: ``True`` when ``name`` is registered as a built-in dataset.
    """
    return name in _DATASETS


def _download_h5mu(name: str) -> Path:
    """Download the .h5mu file for a built-in dataset.

    :param name: Dataset name from the registry.
    :returns: Local path to the downloaded file.
    :raises KeyError: If the dataset is not in the registry.
    :raises requests.HTTPError: If the download fails.
    """
    entry = _DATASETS[name]
    base_url = _SOURCES[entry["source"]]
    file_url = f"{base_url}/{entry['file']}"

    data_dir = get_default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / entry["file"]

    print(f"Downloading {name} from {file_url}...")
    response = requests.get(file_url, timeout=300, stream=True)
    response.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 64):
            if chunk:
                f.write(chunk)
    print(f"Saved to {dest}")
    return dest


def load_mudataset(dataset_name: str) -> MuDataset:
    """Load a built-in or custom dataset as a MuDataset from its .h5mu file.

    Resolution order:

    1. If the .h5mu exists at the standard cache path, load it directly.
    2. If *dataset_name* is built-in, download the .h5mu if needed and load.
    3. If *dataset_name* is a path to an existing .h5mu file, load it directly.

    :param dataset_name: Built-in dataset name, or path to a .h5mu file.
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
