"""Load built-in and custom drug-response datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import pandas as pd

from ._paths import get_default_data_dir, resolve_h5mu_path
from .mudataset import MuDataset
from .utils import (
    ALLOWED_MEASURES,
    CELL_LINE_IDENTIFIER,
    DRUG_IDENTIFIER,
    TISSUE_IDENTIFIER,
    download_dataset,
    download_from_url,
    unzip_data,
)

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


# ------------------------------------------------------------------
# Legacy CSV loading (kept for component infrastructure and tests)
# ------------------------------------------------------------------


def check_measure(measure_queried: str, measures_data: list[str], dataset_name: str) -> None:
    """Validate that a response measure exists in a dataset table.

    :param measure_queried: Column name requested for loading.
    :param measures_data: Column names present in the response table.
    :param dataset_name: Dataset name included in error messages.
    :raises ValueError: If *measure_queried* is not among *measures_data*.
    """
    measures_available = set(ALLOWED_MEASURES).intersection(set(measures_data))
    if measure_queried not in measures_data:
        raise ValueError(
            f"Measure '{measure_queried}' not found in dataset {dataset_name}."
            f"Available measures are: {', '.join(measures_available)}."
        )


def _ensure_zenodo_artifacts(entry: BuiltinDatasetEntry, source: _SourceConfig) -> None:
    path_data = get_default_data_dir()
    response_path = path_data / entry.response_file
    if not response_path.is_file():
        download_dataset(entry.name, redownload=True)

    meta_path = path_data / "meta" / "tissue_mapping.csv"
    if "meta" in source.ensure_artifacts and not meta_path.is_file():
        download_dataset("meta", redownload=True)


def _download_nfcore_zip(path_data: Path, artifact_name: str, base_url: str) -> None:
    file_url = f"{base_url}/{artifact_name}.zip"
    file_path = path_data / f"{artifact_name}.zip"
    response = download_from_url(dataset_name=artifact_name, file_url=file_url)
    unzip_data(path_to_zip=file_path, response=response, data_path=path_data)


def _ensure_nfcore_artifacts(entry: BuiltinDatasetEntry, source: _SourceConfig) -> None:
    path_data = get_default_data_dir()
    if source.base_url is None:
        raise ValueError(f"nfcore source for dataset {entry.name} is missing base_url")
    base_url = source.base_url
    path_data.mkdir(parents=True, exist_ok=True)
    for artifact in source.ensure_artifacts:
        artifact_path = path_data / artifact
        if not artifact_path.exists():
            _download_nfcore_zip(path_data, artifact, base_url)

    _download_nfcore_zip(path_data, entry.name, base_url)


def _ensure_builtin_artifacts(entry: BuiltinDatasetEntry) -> None:
    source = _SOURCES[entry.source]
    if source.kind == "zenodo":
        _ensure_zenodo_artifacts(entry, source)
    else:
        _ensure_nfcore_artifacts(entry, source)


def _read_response_csv(path: Path) -> pd.DataFrame:
    response_data = pd.read_csv(path, dtype={"pubchem_id": str, "cell_line_name": str})
    response_data[DRUG_IDENTIFIER] = response_data[DRUG_IDENTIFIER].str.replace(",", "")
    return response_data


def _load_builtin(entry: BuiltinDatasetEntry, measure: str):
    from .dataset import DrugResponseDataset

    data_root = get_default_data_dir()
    _ensure_builtin_artifacts(entry)
    response_path = data_root / entry.response_file
    response_data = _read_response_csv(response_path)
    check_measure(measure, list(response_data.columns), entry.name)
    if entry.tissue_override is not None:
        response_data[TISSUE_IDENTIFIER] = entry.tissue_override
    return DrugResponseDataset(
        response=response_data[measure].values,
        cell_line_ids=response_data[CELL_LINE_IDENTIFIER].values,
        drug_ids=response_data[DRUG_IDENTIFIER].values,
        tissues=response_data[TISSUE_IDENTIFIER].values,
        dataset_name=entry.name,
    )


def load_custom(
    path_data: str | Path, dataset_name: str = "custom", measure: str = "response", tissue_column: str | None = None
):
    """Load a custom drug-response table from CSV.

    :param path_data: Path to the CSV file or directory containing ``{dataset_name}.csv``.
    :param dataset_name: Label stored on the returned dataset.
    :param measure: Column name for the response values to predict.
    :param tissue_column: Optional tissue column name; ``None`` skips tissue loading.
    :returns: ``DrugResponseDataset`` with response, identifiers, and optional tissues.
    """
    from .dataset import DrugResponseDataset

    return DrugResponseDataset.from_csv(
        input_file=path_data, dataset_name=dataset_name, measure=measure, tissue_column=tissue_column
    )


def load_response_dataset(
    dataset_name: str,
    measure: str = "response",
    curve_curator: bool = False,
    cores: int = 1,
    tissue_column: str | None = None,
    normalize: bool = False,
):
    """Load a built-in or custom drug-response dataset (legacy CSV path).

    Built-in names resolve through the packaged registry and download artifacts
    on demand. Custom datasets are read from
    ``<cache_dir>/<dataset_name>/<dataset_name>.csv``.

    :param dataset_name: Built-in registry name or custom dataset folder name.
    :param measure: Response column name.
    :param curve_curator: Fit CurveCurator from raw viability data for custom sets.
    :param cores: Worker count for CurveCurator fitting.
    :param tissue_column: Tissue column for custom CSV loads only.
    :param normalize: Normalize responses during CurveCurator fitting.
    :returns: ``DrugResponseDataset`` with response values and identifiers.
    :raises FileNotFoundError: If a custom dataset CSV cannot be found.
    """
    from .curvecurator import fit_curves

    data_dir = get_default_data_dir()
    if curve_curator:
        measure += "_curvecurator"
        input_file = data_dir / dataset_name / f"{dataset_name}_raw.csv"
    else:
        input_file = data_dir / dataset_name / f"{dataset_name}.csv"

    entry = _REGISTRY.get(dataset_name)
    if entry is not None:
        return _load_builtin(entry, measure=measure)

    if input_file.is_file():
        if curve_curator:
            fit_curves(
                input_file=input_file,
                output_dir=input_file.parent,
                dataset_name=dataset_name,
                cores=cores,
                normalize=normalize,
            )
        return load_custom(
            path_data=data_dir / dataset_name / f"{dataset_name}.csv",
            dataset_name=dataset_name,
            measure=measure,
            tissue_column=tissue_column,
        )
    raise FileNotFoundError(f"Custom dataset does not exist at given path: {input_file}")


def load_dataset(
    dataset_name: str,
    measure: str = "response",
    curve_curator: bool = False,
    cores: int = 1,
    tissue_column: str | None = None,
    normalize: bool = False,
):
    """Backward-compatible alias for ``load_response_dataset``.

    :param dataset_name: Dataset name or custom study folder name.
    :param measure: Response measure column to load.
    :param curve_curator: Whether to fit curves via CurveCurator when needed.
    :param cores: Parallel cores for CurveCurator fitting.
    :param tissue_column: Optional tissue annotation column.
    :param normalize: Whether to normalize responses after loading.
    :returns: Loaded drug-response dataset.
    """
    return load_response_dataset(
        dataset_name=dataset_name,
        measure=measure,
        curve_curator=curve_curator,
        cores=cores,
        tissue_column=tissue_column,
        normalize=normalize,
    )
