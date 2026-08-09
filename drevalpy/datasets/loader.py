"""Load built-in and custom drug-response datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from ._paths import get_default_data_dir, resolve_h5mu_path
from .mudataset import MuDataset

_REGISTRY_JSON = "available_datasets.json"


@dataclass(frozen=True)
class _SourceConfig:
    kind: str
    ensure_artifacts: tuple[str, ...]
    base_url: str | None = None


@dataclass(frozen=True)
class BuiltinDatasetEntry:
    """Built-in dataset metadata loaded from the packaged registry."""

    name: str
    source: str
    response_file: str
    h5mu_file: str | None = None
    tissue_override: str | None = None


def _load_registry() -> tuple[str, dict[str, _SourceConfig], dict[str, BuiltinDatasetEntry]]:
    """Loads the registry of built-in datasets from the packaged registry.json file.

    :returns: A tuple containing the default measure, the sources, and the registry.
    """
    registry_path = resources.files(__package__).joinpath(_REGISTRY_JSON)
    with registry_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)

    default_measure: str = raw["default_measure"]
    sources = {
        name: _SourceConfig(
            kind=cfg["kind"],
            ensure_artifacts=tuple(cfg.get("ensure_artifacts", [])),
            base_url=cfg.get("base_url"),
        )
        for name, cfg in raw["sources"].items()
    }
    registry = {
        entry["name"]: BuiltinDatasetEntry(
            name=entry["name"],
            source=entry["source"],
            response_file=entry["response_file"],
            h5mu_file=entry.get("h5mu_file"),
            tissue_override=entry.get("tissue_override"),
        )
        for entry in raw["datasets"]
    }
    return default_measure, sources, registry


_DEFAULT_MEASURE, _SOURCES, _REGISTRY = _load_registry()


def list_builtin_datasets() -> list[str]:
    """List built-in dataset names from the packaged registry.

    :returns: Sorted dataset names registered for ``load_mudataset``.
    """
    return sorted(_REGISTRY)


def is_builtin_dataset(name: str) -> bool:
    """Return whether ``name`` is a built-in dataset.

    :param name: Dataset name to look up in the registry.
    :returns: ``True`` when ``name`` is registered as a built-in dataset.
    """
    return name in _REGISTRY


def get_builtin_dataset_entry(name: str) -> BuiltinDatasetEntry | None:
    """Return registry metadata for a built-in dataset.

    :param name: Built-in dataset name.
    :returns: Registry entry for *name*, or ``None`` when the name is unknown.
    """
    return _REGISTRY.get(name)


def load_mudataset(dataset_name: str) -> MuDataset:
    """Load a built-in or custom dataset as a MuDataset from its .h5mu file.

    Resolution order:

    1. If the .h5mu exists at the standard cache path, load it directly.
    2. If *dataset_name* is built-in and has an ``h5mu_file`` entry, download if
       needed and load.
    3. If *dataset_name* is a path to an existing .h5mu file, load it directly.

    :param dataset_name: Built-in dataset name, or path to a .h5mu file.
    :returns: Loaded MuDataset.
    :raises FileNotFoundError: If the .h5mu file cannot be found or downloaded.
    """
    h5mu_path = resolve_h5mu_path(dataset_name)
    if h5mu_path.is_file():
        return MuDataset.from_file(h5mu_path)

    entry = _REGISTRY.get(dataset_name)
    if entry is not None and entry.h5mu_file is not None:
        data_dir = get_default_data_dir()
        candidate = data_dir / entry.h5mu_file
        if candidate.is_file():
            return MuDataset.from_file(candidate)

    candidate_path = Path(dataset_name)
    if candidate_path.is_file() and candidate_path.suffix == ".h5mu":
        return MuDataset.from_file(candidate_path)

    raise FileNotFoundError(
        f"Cannot locate .h5mu for dataset '{dataset_name}'. Checked: {h5mu_path}, registry entry, and direct path."
    )
