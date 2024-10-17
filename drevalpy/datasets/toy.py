"""Toy dataset."""

import pickle
from pathlib import Path

from .dataset import DrugResponseDataset
from .utils import download_dataset


class Toy(DrugResponseDataset):
    """
    Toy dataset, subsampled from GDSC1. Used for testing purposes.

    This datasets contains 40 random drugs and 100 random cell lines.
    Methylation features were subsampled to 250.
    """

    def __init__(
        self,
        path_data: str | Path = "data",
        file_name: str = "toy_data_drp_dataset.pkl",
        dataset_name: str = "Toy_Data",
    ):
        """
        Initialization method for Toy dataset.

        :param path_data: path to the dataset
        """
        path = Path(path_data) / dataset_name / file_name
        if not path.exists():
            path.mkdir(parents=True)
            download_dataset(dataset_name, path_data, redownload=True)
        with open(path, "rb") as f:
            response_data = pickle.load(f)

        response_data.drug_ids = [di.replace(",", "") for di in response_data.drug_ids]
        super().__init__(
            response=response_data.response,
            cell_line_ids=response_data.cell_line_ids,
            drug_ids=response_data.drug_ids,
            dataset_name=dataset_name,
        )
