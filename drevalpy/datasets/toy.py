"""Toy dataset."""

import os

import pandas as pd

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
        path_data: str = "data",
        file_name: str = "toy_data.csv",
        dataset_name: str = "Toy_Data",
    ):
        """
        Initialization method for Toy dataset.

        :param path_data: path to the dataset
        """
        path = os.path.join(path_data, dataset_name, file_name)
        if not os.path.exists(path):
            download_dataset(dataset_name, path_data, redownload=True)
        response_data = pd.read_csv(path)
        super().__init__(
            response=response_data.response,
            cell_line_ids=response_data.cell_line_id,
            drug_ids=response_data.drug_id,
            dataset_name=dataset_name,
        )
