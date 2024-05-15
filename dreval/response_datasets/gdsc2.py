from .gdsc1 import GDSC1

class GDSC2(GDSC1):
    """
    GDSC2 dataset.
    """

    def __init__(self, path: str = "data/GDSC/response_GDSC2.csv"):
        """
        :param path: path to the dataset
        """
        super().__init__(path)