from dreval.datasets.dataset import DrugResponseDataset
import pandas as pd


class GDSC1(DrugResponseDataset):
    """
    GDSC1 dataset.
    """

    def __init__(self, path_data: str = "data/", file_name: str = "response_GDSC1.csv", dataset_name: str = "GDSC1"):
        """
        :param path: path to the dataset
        """
        response_data = pd.read_csv(path_data + f"GDSC/{file_name}")
        self.response =  response_data["LN_IC50"].values
        self.cell_line_ids = response_data["CELL_LINE_NAME"].values
        self.drug_ids = response_data["DRUG_NAME"].values
        self.predictions = None
        self.dataset_name = dataset_name