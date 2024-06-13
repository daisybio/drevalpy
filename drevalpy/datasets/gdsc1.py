from drevalpy.datasets.dataset import DrugResponseDataset
import pandas as pd
import os


class GDSC1(DrugResponseDataset):
    """
    GDSC1 dataset.
    """

    def __init__(
        self,
        path_data: str = "data",
        file_name: str = "response_GDSC1.csv",
        dataset_name: str = "GDSC1",
    ):
        """
        :param path: path to the dataset
        """
        path = os.path.join(path_data, dataset_name, file_name)
        response_data = pd.read_csv(path)
        super().__init__(
            response=response_data["LN_IC50"].values,
            cell_line_ids=response_data["CELL_LINE_NAME"].values,
            drug_ids=response_data["DRUG_NAME"].values,
            dataset_name=dataset_name,
        )
