"""Utility functions for datasets."""

import zipfile
from pathlib import Path

import requests
from requests import Response

from ._paths import get_default_data_dir

DRUG_IDENTIFIER = "pubchem_id"
CELL_LINE_IDENTIFIER = "cell_line_name"
TISSUE_IDENTIFIER = "tissue"
ALLOWED_MEASURES = ["LN_IC50", "EC50", "IC50", "pEC50", "AUC", "response"]
ALLOWED_MEASURES.extend([f"{m}_curvecurator" for m in ALLOWED_MEASURES])


def unzip_data(path_to_zip: Path, response: Response, data_path: str | Path):
    """Unzips the downloaded data.

    :param path_to_zip: Path to the zip file to be unzipped.
    :param response: HTML response containing response.content
    :param data_path: Where the unzipped directory should be stored
    """
    with open(path_to_zip, "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile(path_to_zip, "r") as z:
        for member in z.infolist():
            if not member.filename.startswith("__MACOSX/"):
                z.extract(member, Path(data_path))
    path_to_zip.unlink()  # Remove zip file after extraction


def download_from_url(dataset_name: str, file_url: str) -> Response:
    """Download a file from a given URL.

    :param dataset_name: how the dataset is called
    :param file_url: exact URL to the zip file
    :return: HTML response containing response.content
    :raises HTTPError: if the download fails
    """
    print(f"Downloading {dataset_name} from {file_url}...")
    response = requests.get(file_url, timeout=120)
    if response.status_code != 200:
        raise requests.exceptions.HTTPError(f"Error downloading file: {response.status_code}")
    return response


def download_dataset(
    dataset_name: str,
    redownload: bool = False,
):
    """Download the latest dataset from Zenodo.

    :param dataset_name: dataset name, from "GDSC1", "GDSC2", "CCLE", "CTRPv1", "CTRPv2", "TOYv1", "TOYv2", "meta"
    :param redownload: whether to redownload the data
    :raises HTTPError: if the download fails
    """
    data_path = get_default_data_dir()
    file_name = f"{dataset_name}.zip"
    file_path = Path(data_path) / file_name
    extracted_folder_path = file_path.with_suffix("")
    timeout = 120
    # Check if the extracted data exists and skip download if not redownloading
    if extracted_folder_path.exists() and not redownload:
        print(f"{dataset_name} is already extracted, skipping download.")
    else:
        url = "https://zenodo.org/doi/10.5281/zenodo.12633909"
        # Fetch the latest record
        response = requests.get(url, timeout=timeout)
        if response.status_code != 200:
            raise requests.exceptions.HTTPError(f"Error fetching record: {response.status_code}")
        latest_url = response.links["linkset"]["url"]
        response = requests.get(latest_url, timeout=timeout)
        if response.status_code != 200:
            raise requests.exceptions.HTTPError(f"Error fetching record: {response.status_code}")
        data = response.json()

        # Ensure the save path exists
        extracted_folder_path.parent.mkdir(exist_ok=True, parents=True)

        # Download each file
        name_to_url = {file["key"]: file["links"]["self"] for file in data["files"]}
        file_url = name_to_url[file_name]

        response = download_from_url(dataset_name=dataset_name, file_url=file_url)
        unzip_data(path_to_zip=file_path, response=response, data_path=data_path)

        print(f"{dataset_name} data downloaded and extracted to {data_path}")
