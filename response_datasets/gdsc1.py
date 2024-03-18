from suite.dataset import DrugResponseDataset
import pandas as pd


class GDSC1(DrugResponseDataset):
    """
    GDSC1 dataset.
    """

    def __init__(self, path: str = "data/GDSC/response_GDSC1.csv"):
        """
        :param path: path to the dataset
        """
        response_data = pd.read_csv(path)
        self.response =  response_data["LN_IC50"].values
        self.cell_line_ids = response_data["CELL_LINE_NAME"].values
        self.drug_ids = response_data["DRUG_NAME"].values
        self.predictions = None