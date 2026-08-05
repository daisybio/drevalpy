"""Load built-in and custom drug-response datasets."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import pandas as pd

from .curvecurator import fit_curves
from .dataset import DrugResponseDataset
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
_META_TISSUE_MAPPING = os.path.join("meta", "tissue_mapping.csv")


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
    tissue_override: str | None = None


def _load_registry() -> tuple[str, dict[str, _SourceConfig], dict[str, BuiltinDatasetEntry]]:
    """Loads the registry of built-in datasets from the packaged registry.json file.

    This function only runs once when the module is imported.
    After that, the registry is cached in the module's namespace.

    :returns: A tuple containing the default measure, the sources, and the registry.
    :rtype: tuple[str, dict[str, _SourceConfig], dict[str, BuiltinDatasetEntry]]
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
            tissue_override=entry.get("tissue_override"),
        )
        for entry in raw["datasets"]
    }
    return default_measure, sources, registry


_DEFAULT_MEASURE, _SOURCES, _REGISTRY = _load_registry()


def list_builtin_datasets() -> list[str]:
    """List built-in dataset names from the packaged registry.

    :returns: Sorted dataset names registered for ``load_dataset``.
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


def _ensure_zenodo_artifacts(path_data: str, entry: BuiltinDatasetEntry, source: _SourceConfig) -> None:
    response_path = Path(path_data) / entry.response_file
    if not response_path.is_file():
        download_dataset(entry.name, path_data, redownload=True)

    meta_path = Path(path_data) / _META_TISSUE_MAPPING
    if "meta" in source.ensure_artifacts and not meta_path.is_file():
        download_dataset("meta", path_data, redownload=True)


def _download_nfcore_zip(path_data: str, artifact_name: str, base_url: str) -> None:
    file_url = f"{base_url}/{artifact_name}.zip"
    file_path = Path(path_data) / f"{artifact_name}.zip"
    response = download_from_url(dataset_name=artifact_name, file_url=file_url)
    unzip_data(path_to_zip=file_path, response=response, data_path=path_data)


def _ensure_nfcore_artifacts(path_data: str, entry: BuiltinDatasetEntry, source: _SourceConfig) -> None:
    if source.base_url is None:
        raise ValueError(f"nfcore source for dataset {entry.name} is missing base_url")
    base_url = source.base_url
    Path(path_data).mkdir(parents=True, exist_ok=True)
    for artifact in source.ensure_artifacts:
        artifact_path = Path(path_data) / artifact
        if not artifact_path.exists():
            _download_nfcore_zip(path_data, artifact, base_url)

    _download_nfcore_zip(path_data, entry.name, base_url)


def _ensure_builtin_artifacts(path_data: str, entry: BuiltinDatasetEntry) -> None:
    source = _SOURCES[entry.source]
    if source.kind == "zenodo":
        _ensure_zenodo_artifacts(path_data, entry, source)
    else:
        _ensure_nfcore_artifacts(path_data, entry, source)


def _read_response_csv(path: Path) -> pd.DataFrame:
    response_data = pd.read_csv(path, dtype={"pubchem_id": str, "cell_line_name": str})
    response_data[DRUG_IDENTIFIER] = response_data[DRUG_IDENTIFIER].str.replace(",", "")
    return response_data


def _load_builtin(entry: BuiltinDatasetEntry, path_data: str, measure: str) -> DrugResponseDataset:
    _ensure_builtin_artifacts(path_data, entry)
    response_path = Path(path_data) / entry.response_file
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
) -> DrugResponseDataset:
    """Load a custom drug-response table from CSV.

    :param path_data: Path to the CSV file or directory containing ``{dataset_name}.csv``.
    :param dataset_name: Label stored on the returned ``DrugResponseDataset``.
    :param measure: Column name for the response values to predict.
    :param tissue_column: Optional tissue column name; ``None`` skips tissue loading.
    :returns: ``DrugResponseDataset`` with response, identifiers, and optional tissues.
    """
    return DrugResponseDataset.from_csv(
        input_file=path_data, dataset_name=dataset_name, measure=measure, tissue_column=tissue_column
    )


def load_dataset(
    dataset_name: str,
    path_data: str = "data",
    measure: str = "response",
    curve_curator: bool = False,
    cores: int = 1,
    tissue_column: str | None = None,
    normalize: bool = False,
) -> DrugResponseDataset:
    """Load a built-in or custom drug-response dataset.

    Built-in names resolve through the packaged registry and download artifacts
    on demand. Custom datasets are read from
    ``<path_data>/<dataset_name>/<dataset_name>.csv``.

    :param dataset_name: Built-in registry name or custom dataset folder name.
    :param path_data: Parent directory for downloaded or custom datasets.
    :param measure: Response column name; ``"_curvecurator"`` is appended when ``curve_curator`` is ``True``.
    :param curve_curator: Fit CurveCurator from raw viability data for custom sets.
    :param cores: Worker count for CurveCurator fitting.
    :param tissue_column: Tissue column for custom CSV loads only.
    :param normalize: Normalize responses to ``[0, 1]`` during CurveCurator fitting.
    :returns: ``DrugResponseDataset`` with response values and identifiers.
    :raises FileNotFoundError: If a custom dataset CSV cannot be found.
    """
    if curve_curator:
        measure += "_curvecurator"
        input_file = Path(path_data).resolve() / dataset_name / f"{dataset_name}_raw.csv"
    else:
        input_file = Path(path_data).resolve() / dataset_name / f"{dataset_name}.csv"

    entry = _REGISTRY.get(dataset_name)
    if entry is not None:
        return _load_builtin(entry, path_data, measure=measure)

    if input_file.is_file():
        if curve_curator:
            fit_curves(
                input_file=str(input_file),
                output_dir=str(input_file.parent),
                dataset_name=dataset_name,
                cores=cores,
                normalize=normalize,
            )
        return load_custom(
            path_data=Path(path_data) / dataset_name / f"{dataset_name}.csv",
            dataset_name=dataset_name,
            measure=measure,
            tissue_column=tissue_column,
        )
    raise FileNotFoundError(f"Custom dataset does not exist at given path: {input_file}")
