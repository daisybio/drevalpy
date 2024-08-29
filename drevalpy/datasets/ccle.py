"""
CCLE dataset.
"""

from .gdsc1 import GDSC1


class CCLE(GDSC1):
    """
    CCLE dataset.
    """

    def __init__(
        self,
        path_data: str = "data",
        file_name: str = "response_CCLE.csv",
    ):
        """
        :param path: path to the dataset
        """
        super().__init__(
            path_data=path_data,
            file_name=file_name,
            dataset_name="CCLE",
        )
