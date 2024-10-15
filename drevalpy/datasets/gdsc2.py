"""GDSC2 dataset."""

from .gdsc1 import GDSC1


class GDSC2(GDSC1):
    """GDSC2 dataset."""

    def __init__(self, path_data: str = "data", file_name: str = "response_GDSC2.csv"):
        """
        Initialization method for GDSC2 dataset.

        :param path_data: path to the dataset
        """
        super().__init__(path_data=path_data, file_name=file_name, dataset_name="GDSC2")
