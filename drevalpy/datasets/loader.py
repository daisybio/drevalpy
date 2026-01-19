"""Contains functions to load the GDSC1, GDSC2, CCLE, and Toy datasets."""

import os
from pathlib import Path
from typing import Callable

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


def _load_zenodo_dataset(
    path_data: str = "data",
    measure: str = "LN_IC50_curvecurator",
    file_name: str = "dataset_name.csv",
    dataset_name: str = "dataset_name",
) -> DrugResponseDataset:
    """
    Parent function to load_gdsc1, load_gdsc2, ...

    :param path_data: Path to the dataset.
    :param file_name: File name of the dataset, e.g., GDSC1.csv
    :param measure: File name of the dataset, default = "LN_IC50_curvecurator".
    :param dataset_name: Name of the dataset, e.g., GDSC1.
    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs.
    """
    path = os.path.join(path_data, dataset_name, file_name)
    if not os.path.exists(path):
        download_dataset(dataset_name, path_data, redownload=True)
    # tissue mapping is not in TOY play dataset
    meta_path = os.path.join(path_data, "meta", "tissue_mapping.csv")
    if not os.path.exists(meta_path):
        download_dataset("meta", path_data, redownload=True)

    response_data = pd.read_csv(path, dtype={"pubchem_id": str, "cell_line_name": str})
    response_data[DRUG_IDENTIFIER] = response_data[DRUG_IDENTIFIER].str.replace(",", "")
    check_measure(measure, list(response_data.columns), dataset_name)
    if dataset_name == "BeatAML2":
        # only has AML patients = blood
        response_data[TISSUE_IDENTIFIER] = "Blood"
    elif dataset_name == "PDX_Bruna":
        # only has breast cancer patients
        response_data[TISSUE_IDENTIFIER] = "Breast"
    return DrugResponseDataset(
        response=response_data[measure].values,
        cell_line_ids=response_data[CELL_LINE_IDENTIFIER].values,
        drug_ids=response_data[DRUG_IDENTIFIER].values,
        tissues=response_data[TISSUE_IDENTIFIER].values,
        dataset_name=dataset_name,
    )


def load_gdsc1(
    path_data: str = "data",
    measure: str = "LN_IC50_curvecurator",
) -> DrugResponseDataset:
    """
    Loads the GDSC1 dataset.

    :param path_data: Path to the dataset.
    :param measure: The name of the column containing the measure to predict, default = "LN_IC50_curvecurator"
    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs.
    """
    return _load_zenodo_dataset(path_data=path_data, measure=measure, file_name="GDSC1.csv", dataset_name="GDSC1")


def load_gdsc2(
    path_data: str = "data",
    measure: str = "LN_IC50_curvecurator",
):
    """
    Loads the GDSC2 dataset.

    :param path_data: Path to the dataset.
    :param measure: The name of the column containing the measure to predict, default = "LN_IC50_curvecurator"

    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs.
    """
    return _load_zenodo_dataset(path_data=path_data, measure=measure, file_name="GDSC2.csv", dataset_name="GDSC2")


def load_ccle(
    path_data: str = "data",
    measure: str = "LN_IC50_curvecurator",
) -> DrugResponseDataset:
    """
    Loads the CCLE dataset.

    :param path_data: Path to the dataset.
    :param measure: The name of the column containing the measure to predict, default = "LN_IC50_curvecurator"

    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs.
    """
    return _load_zenodo_dataset(path_data=path_data, measure=measure, file_name="CCLE.csv", dataset_name="CCLE")


def _load_test_data(
    path_data: str = "data", measure: str = "LN_IC50_curvecurator", dataset_name: str = "TOYv1"
) -> DrugResponseDataset:
    # ensure that path_data exists
    Path(path_data).mkdir(parents=True, exist_ok=True)
    test_data_path = "https://github.com/nf-core/test-datasets/raw/refs/heads/drugresponseeval/test_data"
    # first get meta
    meta_path = os.path.join(path_data, "meta")
    if not os.path.exists(meta_path):
        file_url = f"{test_data_path}/meta.zip"
        file_path = Path(path_data) / "meta.zip"
        response_meta = download_from_url(dataset_name="meta", file_url=file_url)
        unzip_data(path_to_zip=file_path, response=response_meta, data_path=path_data)
    # get raw test data
    raw_data_path = os.path.join(path_data, "CTRPv2_sample_test")
    if not os.path.exists(raw_data_path):
        file_url = f"{test_data_path}/CTRPv2_sample_test.zip"
        file_path = Path(path_data) / "CTRPv2_sample_test.zip"
        response_raw = download_from_url(dataset_name="CTRPv2_sample_test", file_url=file_url)
        unzip_data(path_to_zip=file_path, response=response_raw, data_path=path_data)
    file_url = f"{test_data_path}/{dataset_name}.zip"
    file_path = Path(path_data) / f"{dataset_name}.zip"
    response = download_from_url(dataset_name=dataset_name, file_url=file_url)
    unzip_data(path_to_zip=file_path, response=response, data_path=path_data)

    file_name = Path(path_data) / dataset_name / f"{dataset_name}.csv"
    response_data = pd.read_csv(file_name, dtype={"pubchem_id": str, "cell_line_name": str})
    response_data[DRUG_IDENTIFIER] = response_data[DRUG_IDENTIFIER].str.replace(",", "")
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
    return _load_test_data(path_data=path_data, measure=measure, dataset_name="TOYv1")


def load_toyv2(path_data: str = "data", measure: str = "LN_IC50_curvecurator") -> DrugResponseDataset:
    """
    Loads small Toy dataset, subsampled from GDSC2. Can be used to test cross study prediction.

    :param path_data: Path to the dataset.
    :param measure: The name of the column containing the measure to predict, default = "LN_IC50_curvecurator"

    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs.
    """
    return _load_test_data(path_data=path_data, measure=measure, dataset_name="TOYv2")


def load_ctrpv1(path_data: str = "data", measure: str = "LN_IC50_curvecurator") -> DrugResponseDataset:
    """
    Load CTRPv1 dataset.

    :param path_data: Path to the location of CTRPv1 dataset
    :param measure: The name of the column containing the measure to predict, default = "LN_IC50_curvecurator"

    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs
    """
    return _load_zenodo_dataset(path_data=path_data, measure=measure, file_name="CTRPv1.csv", dataset_name="CTRPv1")


def load_ctrpv2(path_data: str = "data", measure: str = "LN_IC50_curvecurator") -> DrugResponseDataset:
    """
    Load CTRPv2 dataset.

    :param path_data: Path to the location of CTRPv2 dataset
    :param measure: The name of the column containing the measure to predict, default: LN_IC50_curvecurator

    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs
    """
    return _load_zenodo_dataset(path_data=path_data, measure=measure, file_name="CTRPv2.csv", dataset_name="CTRPv2")


def load_beataml2(
    path_data: str = "data",
    measure: str = "LN_IC50_curvecurator",
) -> DrugResponseDataset:
    """
    Loads the BeatAML2 dataset.

    :param path_data: Path to the dataset.
    :param measure: The name of the column containing the measure to predict, default: LN_IC50_curvecurator

    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs.
    """
    return _load_zenodo_dataset(path_data=path_data, measure=measure, file_name="BeatAML2.csv", dataset_name="BeatAML2")


def load_pdx_bruna(
    path_data: str = "data",
    measure: str = "LN_IC50_curvecurator",
) -> DrugResponseDataset:
    """
    Loads the PDX_Bruna dataset.

    :param path_data: Path to the dataset.
    :param measure: The name of the column containing the measure to predict, default: LN_IC50_curvecurator

    :return: DrugResponseDataset containing response, cell line IDs, and drug IDs.
    """
    return _load_zenodo_dataset(
        path_data=path_data, measure=measure, file_name="PDX_Bruna.csv", dataset_name="PDX_Bruna"
    )


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
    "BeatAML2": load_beataml2,
    "PDX_Bruna": load_pdx_bruna,
}


def load_dataset(
    dataset_name: str,
    path_data: str = "data",
    measure: str = "response",
    curve_curator: bool = False,
    cores: int = 1,
    tissue_column: str | None = None,
    normalize: bool = False,
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
    :param normalize: Whether to normalize the response values to [0, 1] for curvecurator. Default = False.
        Only used for custom datasets when curve_curator is True.
    :return: A DrugResponseDataset containing response, cell line IDs, drug IDs, and dataset name.
    :raises FileNotFoundError: If the custom dataset or raw viability data could not be found at the given path.
    """
    if curve_curator:
        measure += "_curvecurator"
        input_file = Path(path_data).resolve() / dataset_name / f"{dataset_name}_raw.csv"
    else:
        input_file = Path(path_data).resolve() / dataset_name / f"{dataset_name}.csv"

    if dataset_name in AVAILABLE_DATASETS:
        return AVAILABLE_DATASETS[dataset_name](path_data, measure=measure)

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
