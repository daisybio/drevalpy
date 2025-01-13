"""Contains functions to load the GDSC1, GDSC2, CCLE, and Toy datasets."""

import os
from pathlib import Path
from typing import Callable

import pandas as pd

from ..pipeline_function import pipeline_function
from .curvecurator import fit_curves
from .dataset import DrugResponseDataset
from .utils import download_dataset


def load_gdsc1(
    path_data: str = "data",
    measure: str = "LN_IC50",
    file_name: str = "response_GDSC1.csv",
    dataset_name: str = "GDSC1",
) -> DrugResponseDataset:
    """
    Loads the GDSC1 dataset.

    :param path_data: Path to the dataset.
    :param file_name: File name of the dataset.
    :param measure: The name of the column containing the measure to predict, default = "LN_IC50"

    :param dataset_name: Name of the dataset.
    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs.
    """
    path = os.path.join(path_data, dataset_name, file_name)
    if not os.path.exists(path):
        download_dataset(dataset_name, path_data, redownload=True)

    response_data = pd.read_csv(path)
    response_data["DRUG_NAME"] = response_data["DRUG_NAME"].str.replace(",", "")

    return DrugResponseDataset(
        response=response_data[measure].values,
        cell_line_ids=response_data["CELL_LINE_NAME"].values,
        drug_ids=response_data["DRUG_NAME"].values,
        dataset_name=dataset_name,
    )


def load_gdsc2(path_data: str = "data", measure: str = "LN_IC50", file_name: str = "response_GDSC2.csv"):
    """
    Loads the GDSC2 dataset.

    :param path_data: Path to the dataset.
    :param file_name: File name of the dataset.
    :param measure: The name of the column containing the measure to predict, default = "LN_IC50"

    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs.
    """
    return load_gdsc1(path_data=path_data, measure=measure, file_name=file_name, dataset_name="GDSC2")


def load_ccle(
    path_data: str = "data", measure: str = "LN_IC50", file_name: str = "response_CCLE.csv"
) -> DrugResponseDataset:
    """
    Loads the CCLE dataset.

    :param path_data: Path to the dataset.
    :param file_name: File name of the dataset.
    :param measure: The name of the column containing the measure to predict, default = "LN_IC50"

    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs.
    """
    dataset_name = "CCLE"
    path = os.path.join(path_data, dataset_name, file_name)
    if not os.path.exists(path):
        download_dataset(dataset_name, path_data, redownload=True)

    response_data = pd.read_csv(path)
    response_data["DRUG_NAME"] = response_data["DRUG_NAME"].str.replace(",", "")

    return DrugResponseDataset(
        response=response_data[measure].values,
        cell_line_ids=response_data["CELL_LINE_NAME"].values,
        drug_ids=response_data["DRUG_NAME"].values,
        dataset_name=dataset_name,
    )


def load_toy(path_data: str = "data", measure: str = "response") -> DrugResponseDataset:
    """
    Loads small Toy dataset, subsampled from GDSC1.

    :param path_data: Path to the dataset.
    :param measure: The name of the column containing the measure to predict, default = "response"

    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs.
    """
    dataset_name = "Toy_Data"
    measure = "response"  # overwrite this explicitly to avoid problems, should be changed in the future
    path = os.path.join(path_data, dataset_name, "toy_data.csv")
    if not os.path.exists(path):
        download_dataset(dataset_name, path_data, redownload=True)
    response_data = pd.read_csv(path)

    return DrugResponseDataset(
        response=response_data[measure].values,
        cell_line_ids=response_data["cell_line_id"].values,
        drug_ids=response_data["drug_id"].values,
        dataset_name=dataset_name,
    )


def load_custom(path_data: str | Path, measure: str = "response") -> DrugResponseDataset:
    """
    Load custom dataset.

    :param path_data: Path to location of custom dataset
    :param measure: The name of the column containing the measure to predict, default = "response"

    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs
    """
    return DrugResponseDataset.from_csv(path_data, measure=measure)


AVAILABLE_DATASETS: dict[str, Callable] = {
    "GDSC1": load_gdsc1,
    "GDSC2": load_gdsc2,
    "CCLE": load_ccle,
    "Toy_Data": load_toy,
}


@pipeline_function
def load_dataset(
    dataset_name: str, path_data: str = "data", measure: str = "response", curve_curator: bool = False, cores: int = 1
) -> DrugResponseDataset:
    """
    Load a dataset based on the dataset name.

    :param dataset_name: The name of the dataset to load. Can be one of ('GDSC1', 'GDSC2', 'CCLE', or 'Toy_Data')
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
    :return: A DrugResponseDataset containing response, cell line IDs, drug IDs, and dataset name.
    :raises FileNotFoundError: If the custom dataset or raw viability data could not be found at the given path.
    """
    if curve_curator:
        measure += "_curvecurator"
        input_file = Path(path_data) / dataset_name / f"{dataset_name}_raw.csv"
    else:
        input_file = Path(path_data) / dataset_name / f"{dataset_name}.csv"

    if dataset_name in AVAILABLE_DATASETS:
        return AVAILABLE_DATASETS[dataset_name](path_data)

    if input_file.is_file():
        if curve_curator:
            fit_curves(
                input_file=input_file,
                output_dir=input_file.parent,
                dataset_name=dataset_name,
                cores=cores,
            )
        return load_custom(Path(path_data) / dataset_name / f"{dataset_name}.csv", measure=measure)
    raise FileNotFoundError(f"Custom dataset does not exist at given path: {input_file}")
