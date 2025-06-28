"""Contains functions to load the GDSC1, GDSC2, CCLE, and Toy datasets."""

import os
from pathlib import Path
from typing import Callable

import pandas as pd

from .curvecurator import fit_curves
from .dataset import DrugResponseDataset
from .utils import ALLOWED_MEASURES, CELL_LINE_IDENTIFIER, DRUG_IDENTIFIER, TISSUE_IDENTIFIER, download_dataset


def check_measure(measure_queried: str, measures_data: list[str], dataset_name: str) -> None:
    """
    Check if the queried measure is in the dataset.

    :param measure_queried: The measure to check.
    :param measures_data: The measures in the dataset.
    :param dataset_name: The name of the dataset.
    :raises ValueError: If the measure is not found in the dataset.
    """
    measures_available = set(ALLOWED_MEASURES).intersection(set(measures_data))
    if measure_queried not in measures_data:
        raise ValueError(
            f"Measure '{measure_queried}' not found in dataset {dataset_name}."
            f"Available measures are: {', '.join(measures_available)}."
        )


def load_gdsc1(
    path_data: str = "data",
    measure: str = "LN_IC50_curvecurator",
    file_name: str = "GDSC1.csv",
    dataset_name: str = "GDSC1",
) -> DrugResponseDataset:
    """
    Loads the GDSC1 dataset.

    :param path_data: Path to the dataset.
    :param file_name: File name of the dataset.
    :param measure: The name of the column containing the measure to predict, default = "LN_IC50_curvecurator"

    :param dataset_name: Name of the dataset.
    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs.
    """
    path = os.path.join(path_data, dataset_name, file_name)
    if not os.path.exists(path):
        download_dataset(dataset_name, path_data, redownload=True)

    response_data = pd.read_csv(path, dtype={"pubchem_id": str})
    response_data[DRUG_IDENTIFIER] = response_data[DRUG_IDENTIFIER].str.replace(",", "")
    check_measure(measure, list(response_data.columns), dataset_name)
    return DrugResponseDataset(
        response=response_data[measure].values,
        cell_line_ids=response_data[CELL_LINE_IDENTIFIER].values,
        drug_ids=response_data[DRUG_IDENTIFIER].values,
        tissues=response_data[TISSUE_IDENTIFIER].values,
        dataset_name=dataset_name,
    )


def load_gdsc2(path_data: str = "data", measure: str = "LN_IC50_curvecurator", file_name: str = "GDSC2.csv"):
    """
    Loads the GDSC2 dataset.

    :param path_data: Path to the dataset.
    :param file_name: File name of the dataset.
    :param measure: The name of the column containing the measure to predict, default = "LN_IC50_curvecurator"

    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs.
    """
    return load_gdsc1(path_data=path_data, measure=measure, file_name=file_name, dataset_name="GDSC2")


def load_ccle(
    path_data: str = "data", measure: str = "LN_IC50_curvecurator", file_name: str = "CCLE.csv"
) -> DrugResponseDataset:
    """
    Loads the CCLE dataset.

    :param path_data: Path to the dataset.
    :param file_name: File name of the dataset.
    :param measure: The name of the column containing the measure to predict, default = "LN_IC50_curvecurator"

    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs.
    """
    dataset_name = "CCLE"
    path = os.path.join(path_data, dataset_name, file_name)
    if not os.path.exists(path):
        download_dataset(dataset_name, path_data, redownload=True)

    response_data = pd.read_csv(path, dtype={"pubchem_id": str})
    response_data[DRUG_IDENTIFIER] = response_data[DRUG_IDENTIFIER].str.replace(",", "")
    check_measure(measure, list(response_data.columns), dataset_name)
    return DrugResponseDataset(
        response=response_data[measure].values,
        cell_line_ids=response_data[CELL_LINE_IDENTIFIER].values,
        drug_ids=response_data[DRUG_IDENTIFIER].values,
        tissues=response_data[TISSUE_IDENTIFIER].values,
        dataset_name=dataset_name,
    )


def _load_toy(
    path_data: str = "data", measure: str = "LN_IC50_curvecurator", dataset_name="TOYv1"
) -> DrugResponseDataset:
    """
    Loads small Toy dataset, subsampled from CTRPv2 or GDSC2.

    :param path_data: Path to the dataset.
    :param measure: The name of the column containing the measure to predict, default = "LN_IC50_curvecurator"
    :param dataset_name: Name of the dataset. Either "TOYv1" or "TOYv2".

    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs.
    """
    path = os.path.join(path_data, dataset_name, f"{dataset_name}.csv")
    if not os.path.exists(path):
        download_dataset(dataset_name, path_data, redownload=True)
    response_data = pd.read_csv(path, dtype={"pubchem_id": str})
    check_measure(measure, list(response_data.columns), dataset_name)
    return DrugResponseDataset(
        response=response_data[measure].values,
        cell_line_ids=response_data[CELL_LINE_IDENTIFIER].values,
        drug_ids=response_data[DRUG_IDENTIFIER].values,
        tissues=response_data[TISSUE_IDENTIFIER].values,
        dataset_name=dataset_name,
    )


def load_toyv1(path_data: str = "data", measure: str = "LN_IC50_curvecurator") -> DrugResponseDataset:
    """
    Loads small Toy dataset, subsampled from CTRPv2.

    :param path_data: Path to the dataset.
    :param measure: The name of the column containing the measure to predict, default = "LN_IC50_curvecurator"

    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs.
    """
    return _load_toy(path_data, measure, "TOYv1")


def load_toyv2(path_data: str = "data", measure: str = "LN_IC50_curvecurator") -> DrugResponseDataset:
    """
    Loads small Toy dataset, subsampled from GDSC2. Can be used to test cross study prediction.

    :param path_data: Path to the dataset.
    :param measure: The name of the column containing the measure to predict, default = "LN_IC50_curvecurator"

    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs.
    """
    return _load_toy(path_data, measure, "TOYv2")


def _load_ctrpv(version: str, path_data: str = "data", measure: str = "LN_IC50_curvecurator") -> DrugResponseDataset:
    """
    Load CTRPv1 dataset.

    :param version: The version of the CTRP dataset to load.
    :param path_data: Path to location of CTRPv1 dataset
    :param measure: The name of the column containing the measure to predict, default = "LN_IC50_curvecurator"

    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs
    """
    dataset_name = "CTRPv" + version
    path = os.path.join(path_data, dataset_name, f"{dataset_name}.csv")
    if not os.path.exists(path):
        download_dataset(dataset_name, path_data, redownload=True)
    response_data = pd.read_csv(path, dtype={"pubchem_id": str})
    check_measure(measure, list(response_data.columns), dataset_name)

    return DrugResponseDataset(
        response=response_data[measure].values,
        cell_line_ids=response_data[CELL_LINE_IDENTIFIER].values,
        drug_ids=response_data[DRUG_IDENTIFIER].values,
        tissues=response_data[TISSUE_IDENTIFIER].values,
        dataset_name=dataset_name,
    )


def load_ctrpv1(path_data: str = "data", measure: str = "LN_IC50_curvecurator") -> DrugResponseDataset:
    """
    Load CTRPv2 dataset.

    :param path_data: Path to location of CTRPv2 dataset
    :param measure: The name of the column containing the measure to predict, default = "LN_IC50_curvecurator"

    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs
    """
    return _load_ctrpv("1", path_data, measure)


def load_ctrpv2(path_data: str = "data", measure: str = "LN_IC50_curvecurator") -> DrugResponseDataset:
    """
    Load CTRPv2 dataset.

    :param path_data: Path to location of CTRPv2 dataset
    :param measure: The name of the column containing the measure to predict, default: LN_IC50_curvecurator

    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs
    """
    return _load_ctrpv("2", path_data, measure)


def load_custom(
    path_data: str | Path, dataset_name: str = "custom", measure: str = "response", tissue_column: str | None = None
) -> DrugResponseDataset:
    """
    Load custom dataset.

    :param path_data: Path to location of custom dataset
    :param dataset_name: Name of the dataset.
    :param measure: The name of the column containing the measure to predict, default = "response"
    :param tissue_column: The name of the column containing the tissue type. If None, no tissue information is loaded.

    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs
    """
    return DrugResponseDataset.from_csv(
        input_file=path_data, dataset_name=dataset_name, measure=measure, tissue_column=tissue_column
    )


# Used in pipeline
AVAILABLE_DATASETS: dict[str, Callable] = {
    "GDSC1": load_gdsc1,
    "GDSC2": load_gdsc2,
    "CCLE": load_ccle,
    "TOYv1": load_toyv1,
    "TOYv2": load_toyv2,
    "CTRPv1": load_ctrpv1,
    "CTRPv2": load_ctrpv2,
}


def load_dataset(
    dataset_name: str,
    path_data: str = "data",
    measure: str = "response",
    curve_curator: bool = False,
    cores: int = 1,
    tissue_column: str | None = None,
) -> DrugResponseDataset:
    """
    Load a dataset based on the dataset name.

    :param dataset_name: The name of the dataset to load. Can be one of ('GDSC1', 'GDSC2', 'CCLE', 'TOYv1', or 'TOYv2')
        to download provided datasets, or any other name to allow for custom datasets.
    :param path_data: The parent path in which custom or downloaded datasets should be located, or in which raw
        viability data is to be found for fitting with CurveCurator (see param curve_curator for details).
        The location of the datasets are resolved by <path_data>/<dataset_name>/<dataset_name>.csv.
    :param measure: The name of the column containing the measure to predict, default = "response".
        If curve_curator is True, this measure is appended with "_curvecurator", e.g. "response_curvecurator" to
        distinguish between measures provided by the original source of a dataset, or the measures fit by
        CurveCurator.
    :param curve_curator: If True, the measure is appended with "_curvecurator".
        If a custom dataset_name was provided, this will invoke the fitting procedure of raw viability data,
        which is expected to exist at <path_data>/<dataset_name>/<dataset_name>_raw.csv. The fitted dataset will
        be stored in the same folder, in a file called <dataset_name>.csv
    :param cores: Number of cores to use for CurveCurator fitting. Only used when curve_curator is True, default = 1
    :param tissue_column: The name of the column containing the tissue type. If None, no tissue information is loaded.
        This is only used when loading a custom dataset. Default = None.
    :return: A DrugResponseDataset containing response, cell line IDs, drug IDs, and dataset name.
    :raises FileNotFoundError: If the custom dataset or raw viability data could not be found at the given path.
    """
    if curve_curator:
        measure += "_curvecurator"
        input_file = Path(path_data) / dataset_name / f"{dataset_name}_raw.csv"
    else:
        input_file = Path(path_data) / dataset_name / f"{dataset_name}.csv"

    if dataset_name in AVAILABLE_DATASETS:
        return AVAILABLE_DATASETS[dataset_name](path_data, measure=measure)

    if input_file.is_file():
        if curve_curator:
            fit_curves(
                input_file=input_file,
                output_dir=input_file.parent,
                dataset_name=dataset_name,
                cores=cores,
            )
        return load_custom(
            path_data=Path(path_data) / dataset_name / f"{dataset_name}.csv",
            dataset_name=dataset_name,
            measure=measure,
            tissue_column=tissue_column,
        )
    raise FileNotFoundError(f"Custom dataset does not exist at given path: {input_file}")
