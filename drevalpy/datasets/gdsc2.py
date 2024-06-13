from drevalpy.datasets.gdsc1 import GDSC1


class GDSC2(GDSC1):
    """
    GDSC2 dataset.
    """

    def __init__(self, path_data: str = "data/", file_name: str = "response_GDSC2.csv"):
        super().__init__(path_data=path_data, file_name=file_name, dataset_name="GDSC2")
