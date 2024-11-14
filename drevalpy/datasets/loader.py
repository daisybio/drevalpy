"""Contains functions to load the GDSC1, GDSC2, CCLE, and Toy datasets."""

import os
from typing import Callable

import pandas as pd

from ..pipeline_function import pipeline_function
from .dataset import DrugResponseDataset
from .utils import download_dataset


def load_gdsc1(
    path_data: str = "data", file_name: str = "response_GDSC1.csv", dataset_name: str = "GDSC1"
) -> DrugResponseDataset:
    """
    Loads the GDSC1 dataset.

    :param path_data: Path to the dataset.
    :param file_name: File name of the dataset.
    :param dataset_name: Name of the dataset.
    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs.
    """
    path = os.path.join(path_data, dataset_name, file_name)
    if not os.path.exists(path):
        download_dataset(dataset_name, path_data, redownload=True)

    response_data = pd.read_csv(path)
    response_data["DRUG_NAME"] = response_data["DRUG_NAME"].str.replace(",", "")

    return DrugResponseDataset(
        response=response_data["LN_IC50"].values,
        cell_line_ids=response_data["CELL_LINE_NAME"].values,
        drug_ids=response_data["DRUG_NAME"].values,
        dataset_name=dataset_name,
    )


def load_gdsc2(path_data: str = "data", file_name: str = "response_GDSC2.csv"):
    """
    Loads the GDSC2 dataset.

    :param path_data: Path to the dataset.
    :param file_name: File name of the dataset.
    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs.
    """
    return load_gdsc1(path_data=path_data, file_name=file_name, dataset_name="GDSC2")


def load_ccle(path_data: str = "data", file_name: str = "response_CCLE.csv") -> DrugResponseDataset:
    """
    Loads the CCLE dataset.

    :param path_data: Path to the dataset.
    :param file_name: File name of the dataset.
    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs.
    """
    dataset_name = "CCLE"
    path = os.path.join(path_data, dataset_name, file_name)
    if not os.path.exists(path):
        download_dataset(dataset_name, path_data, redownload=True)

    response_data = pd.read_csv(path)
    response_data["DRUG_NAME"] = response_data["DRUG_NAME"].str.replace(",", "")

    return DrugResponseDataset(
        response=response_data["LN_IC50"].values,
        cell_line_ids=response_data["CELL_LINE_NAME"].values,
        drug_ids=response_data["DRUG_NAME"].values,
        dataset_name=dataset_name,
    )


def load_toy(path_data: str = "data") -> DrugResponseDataset:
    """
    Loads small Toy dataset, subsampled from GDSC1.

    :param path_data: Path to the dataset.
    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs.
    """
    dataset_name = "Toy_Data"
    path = os.path.join(path_data, dataset_name, "toy_data.csv")
    if not os.path.exists(path):
        download_dataset(dataset_name, path_data, redownload=True)
    response_data = pd.read_csv(path)

    return DrugResponseDataset(
        response=response_data["response"].values,
        cell_line_ids=response_data["cell_line_id"].values,
        drug_ids=response_data["drug_id"].values,
        dataset_name=dataset_name,
    )


AVAILABLE_DATASETS: dict[str, Callable] = {
    "GDSC1": load_gdsc1,
    "GDSC2": load_gdsc2,
    "CCLE": load_ccle,
    "Toy_Data": load_toy,
}


@pipeline_function
def load_dataset(dataset_name: str, path_data: str = "data") -> DrugResponseDataset:
    """
    Load a dataset based on the dataset name.

    :param dataset_name: The name of the dataset to load ('GDSC1', 'GDSC2', 'CCLE', or 'Toy_Data').
    :param path_data: The path to the dataset.
    :return: A DrugResponseDataset containing response, cell line IDs, drug IDs, and dataset name.
    :raises ValueError: If the dataset name is unknown.
    """
    if dataset_name in AVAILABLE_DATASETS:
        return AVAILABLE_DATASETS[dataset_name](path_data)  # type: ignore
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
